#%%
import regex as re
import pandas as pd
import sys, time, datetime, os, calendar
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu, binom_test
from config import system_name, data_file_format, scope_name, date_model, pre_defined_systems, parallel, verbose
from tqdm import tqdm
from dateutil import tz
import multiprocessing

try:
    time_zone = pre_defined_systems[system_name]['timezone_dict'][scope_name]
except KeyError:
    time_zone = None


def both(a,b):
    return np.bitwise_and(a,b)

def create_dummy_data(data_date, date_model, system_name, time_zone=time_zone):
    min_data_date = int(np.min(data_date))
    max_data_date = int(np.max(data_date))
    offset = 2
    sampling_interval = int(date_model[0])
    st_dt_str = str(date_model[1])
    end_dt_str = str(date_model[2])
    st_time = datetime.datetime(*time.strptime(st_dt_str, '%Y-%m-%d')[:3], tzinfo=tz.gettz(time_zone)).timestamp()
    end_time = datetime.datetime(*time.strptime(end_dt_str, '%Y-%m-%d')[:3], tzinfo=tz.gettz(time_zone)).timestamp()
    dates = set(np.arange(st_time, end_time, sampling_interval).astype(int))
    data_date_set = set(data_date)
    all_dates = list(dates.union(data_date_set))
    dummy_data = pd.DataFrame({'date':all_dates})

    if preprocess_script_exists:
        dummy_data = preprocess_data.preprocess(dummy_data, dummy = True, time_zone=time_zone)
    dummy_data['available'] = [True if (i>=min_data_date and i<=(max_data_date + 86400)) else False for i in dummy_data.day_date]
    return dummy_data

def convert_epoch_time_to_day_of_the_week(epoch, time_zone=time_zone):
    local_time_object = datetime.datetime.fromtimestamp(epoch,tz=tz.gettz(time_zone))
    return local_time_object.weekday()

def convert_epoch_time_to_year(epoch):
    local_time_object = datetime.datetime.fromtimestamp(epoch,tz=tz.gettz(time_zone))
    return local_time_object.year

def convert_epoch_time_to_month(epoch):
    local_time_object = datetime.datetime.fromtimestamp(epoch,tz=tz.gettz(time_zone))
    return local_time_object.month

def convert_epoch_time_to_day_date(epoch, time_zone=time_zone):
    local_time_object = datetime.datetime.fromtimestamp(epoch,tz=tz.gettz(time_zone))
    td =  datetime.timedelta(hours=local_time_object.hour, minutes = local_time_object.minute, seconds = local_time_object.second)
    local_day_begin = local_time_object - td
    day_date_epoch = local_day_begin.timestamp()
    return day_date_epoch

def convert_epoch_time_to_day_time(epoch, time_zone=time_zone):
    local_time_object = datetime.datetime.fromtimestamp(epoch,tz=tz.gettz(time_zone))
    day_time = local_time_object.hour + local_time_object.minute/60
    return day_time

def check_if_exist(fil):
    if not os.path.exists(fil):
        print("please check if {} exists. if not, refer to the documentation".format(fil))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!aborting score_insight_lib.py")
        sys.exit(1)

def read_file(prefix, system_name, input_file_type):
    if input_file_type == "csv":
        in_file = "{}{}.{}".format(prefix, system_name, input_file_type)
        check_if_exist(in_file)
        df = pd.read_csv(in_file)
    elif input_file_type == "xlsx":
        in_file = "{}{}.{}".format(prefix, system_name, input_file_type)
        check_if_exist(in_file)
        df = pd.read_excel(in_file,engine='openpyxl')
    elif input_file_type == "pickle":
        in_file = "{}{}.{}".format(prefix, system_name, input_file_type)
        check_if_exist(in_file)
        df = pd.read_pickle(in_file)
    else:
        df = sys.exit(1)
    return df

def read_data(system_name, input_file_type, scp_name=""):
    if scp_name and scp_name!="":
        scp_name += "_"
        df = read_file('systems/{}/data_{}'.format(system_name, scp_name), system_name, input_file_type)
    else:
        df = read_file('systems/{}/data_'.format(system_name), system_name, input_file_type)
    if preprocess_script_exists:
        df = preprocess_data.preprocess(df, dummy = False, time_zone=time_zone)
    return df

def read_library(system_name, input_file_type):
    df = read_file('systems/{}/statement_library_'.format(system_name), system_name, input_file_type)
    df['meanA'] = 0.0
    df['meanB'] = 0.0
    df['comparison'] = "<<unknown>>"
    df['score'] = 0.0
    df['percentage'] = 0.0
    df['pvalue']=1.0
    df['score_type'] = "unknown"
    return df

def prev_month_year(x,y): 
    if x>1:
        return (x-1,y) 
    else:
        return (12,y-1)

def get_now_details(now):
    time_now = now.timestamp()
    now_details = {'THIS_MONTH_INT': now.month, 'THIS_MONTH_YEAR_INT':now.year, 'THIS_YEAR_INT':now.year, 'PREV_YEAR_INT':now.year-1, 'PREV_MONTH_INT':prev_month_year(now.month,now.year)[0],
                   'PREV_MONTH_YEAR_INT':prev_month_year(now.month,now.year)[1], 'TODAY_INT': int(time_now - (now.hour*3600 + now.minute*60 + now.second)),
                   'YESTERDAY_INT':int(time_now - (1*24*3600 + now.hour*3600 + now.minute*60 + now.second)),
                   'D_YESTERDAY_INT':int(time_now - (2*24*3600 + now.hour*3600 + now.minute*60 + now.second)),
                   'THIS_WEEK_INT':int(time_now - (now.isoweekday()*24*3600 + now.hour*3600 + now.minute*60 + now.second)),
                   'PREV_WEEK_INT':int(time_now - ((7 + now.isoweekday())*24*3600 + now.hour*3600 + now.minute*60 + now.second)),
                   'D_PREV_WEEK_INT':int(time_now - ((14 + now.isoweekday())*24*3600 + now.hour*3600 + now.minute*60 + now.second))}
    return now_details

def get_percentage(meanA, meanB):
    if meanB != 0:
        percentage = abs(((meanA/meanB) - 1) * 100)
    else:
        percentage = meanA
    return percentage

def get_difference(meanA, meanB):
    return abs(meanA - meanB)

def get_comparison(meanA,meanB,comparison_dict):
    comp_list = eval(comparison_dict)
    if meanA > meanB:
        comparison_ = comp_list[2]
    elif meanA < meanB:
        comparison_ = comp_list[0]
    else:
        comparison_ = comp_list[1]
    return comparison_

def get_completeness(countA, countA_):
    if countA_!= 0:
        res = 1-((countA_ - countA)/countA_)
    else:
        res = 0
    return res

def flexi_eval(sinp):
    try:
        ev = eval(sinp)
        return ev
    except:
        return sinp

def eval_if_str(str1):
    if type(str1) == str:
        return eval(str1)
    else:
        return str1

def get_difference_weight(diff, meas, measurement_tolerance):
    x = diff
    metric_parameters = measurement_tolerance[meas]/6
    a = 1/metric_parameters
    z = 1/(1 + np.exp(-a*x))
    return z


def get_stats(params): 
    (irow, (data, dummy_data, measurement_benchmark_text, benchmark_comparison_phrase, measurement_comparison_phrase, measurement_tolerance, now_details, measurement_benchmark)) = params
    inter = irow['intermediate']
    test = irow['scoring_type']
    insight_text = irow['insight_text']
    qA,qB = irow['queryA'], irow['queryB']
    dqA,dqB = irow['dummyqueryA'], irow['dummyqueryB']
    distA, countA = get_data(qA, data, dummy_data, now_details)
    distA_,countA_ = get_data(dqA, data, dummy_data, now_details)
    if test == 'benchmark':
        distB = eval(qB) 
        countB,countB_ = 1,1
    else:
        distB,countB = get_data(qB, data, dummy_data, now_details) 
        _,countB_ = get_data(dqB, data, dummy_data, now_details)
    metric_name = eval_if_str(inter)['measurement']
    comparison_dict = measurement_comparison_phrase[metric_name]   
    if dqA == dqB:
        max_complete = True #consider the maximum of completeness
    else:
        max_complete = False
    completeness_A = get_completeness(countA, countA_)
    completeness_B = get_completeness(countB, countB_)
    row = {}
    row['meanA'] = 0.0
    row['meanB'] = 0.0
    row['countA'] = countA
    row['countB'] = countB
    row['countA_'] = countA_
    row['countB_'] = countB_
    row['comparison'] = "<<unknown>>"
    row['comparison_pre'] = "<<unknown>>"
    row['comparison_post'] = "<<unknown>>"
    row['score'] = 0.0
    row['percentage'] = 0.0
    row['difference'] = 0.0
    row['pvalue']=1.0
    row['distA'] = distA
    row['distB'] = distB
    if max_complete:
        row['completeness_factor'] = max(completeness_A, completeness_B)
    else:
        row['completeness_factor'] = completeness_A * completeness_B
    row['pscore_nonmissing']=0
    row['pscore_nonmissing_sigmoid']=0
    row['score_type'] = "unknown"
    if test == 'ks':
        if 'TODAY' in (qA+qB):
            if countA > 0 and countB > 0:
                meanA = distA.mean(skipna = True)
                meanB = distB.mean(skipna = True)
                row['meanA'] = meanA
                row['meanB'] = meanB
                row['percentage'] = get_percentage(meanA, meanB)
                row['difference'] = get_difference(meanA, meanB)
                row['comparison'] = get_comparison(meanA, meanB, comparison_dict)
                row['score']=1
                row['pvalue']=0
                row['score_type'] = 'singular'
            else:
                row['score']=0
                row['pvalue']=1
                row['score_type'] = 'irregular'
        else:
            if countA >= 10 and countB >= 10:
                meanA = distA.mean(skipna = True)
                meanB = distB.mean(skipna = True)
                row['meanA'] = meanA
                row['meanB'] = meanB
                row['comparison'] = get_comparison(meanA, meanB, comparison_dict)
                row['percentage'] = get_percentage(meanA, meanB)
                row['difference'] = get_difference(meanA, meanB)
                if test == 'ks':
                    score,pval = ks_2samp(distA,distB)
                else:
                    score,pval = mannwhitneyu(distA,distB)
                score = 1-pval
                row['score']=score
                row['pvalue']=pval
                row['score_type'] = 'regular'
            elif countA < 10  or countB < 10:
                meanA,meanB = 0,0
                if countA>0:
                    meanA = distA.mean(skipna = True)
                if countB>0:
                    meanB = distB.mean(skipna = True)
                row['meanA'] = meanA
                row['meanB'] = meanB
                row['percentage'] = get_percentage(meanA, meanB)
                row['difference'] = get_difference(meanA, meanB)
                row['score_type'] = 'irregular'
    elif test == "benchmark":
        distA = distA.dropna()
        if countA > 0:
            meanA = distA.mean(skipna = True)
            measurement = meanA
            conditions = eval(distB)
            expressions = [(ind,j) for (ind,(i,j)) in enumerate(zip(conditions,eval(benchmark_comparison_phrase[metric_name]))) if i]
            assert(len(expressions)==1)
            row['comparison'] = expressions[0][1]
            row['score_type'] = 'benchmark'
            cond_binomial = eval(distB.replace('measurement', 'distA'))[expressions[0][0]]
            row['meanA'] = distA.mean(skipna = True)
            benchmark_text_exp = flexi_eval(measurement_benchmark_text[metric_name])
            if type(benchmark_text_exp) == list:
                insight_text = insight_text.replace(str(benchmark_text_exp),benchmark_text_exp[expressions[0][0]])
                benchmark_text_exp = benchmark_text_exp[expressions[0][0]]
            row['meanB'] = benchmark_text_exp
            nums = re.findall('([0-9]+[.]*[0-9]*)', benchmark_text_exp)
            if len(nums)==1:
                row['percentage'] = get_percentage(row['meanA'], float(nums[0]))
                row['difference'] = get_difference(row['meanA'], float(nums[0]))
            lp = cond_binomial.sum()
            ln = len(cond_binomial) - lp
            row['pvalue']= binom_test([lp, ln])
            row['score']=1-(row['pvalue'])
    elif test == "autobenchmark":
        metric_name = eval_if_str(inter)['measurement']
        distA = distA.dropna()
        if countA > 0:
            meanA = distA.mean(skipna = True)
            meanB = distB.mean(skipna = True)
            row['comparison'] = get_comparison(meanA, meanB, comparison_dict)
            row['score_type'] = 'autobenchmark'
            row['meanA'] = meanA
            row['meanB'] = meanB
            if row['comparison'] == "greater than":
                cond_binomial = distA > meanB
            elif row['comparison'] == "lower than":
                cond_binomial = distA < meanB
            else:
                cond_binomial = (distA == meanB)
            row['percentage'] = get_percentage(meanA, meanB)
            row['difference'] = get_difference(meanA, meanB)
            lp = cond_binomial.sum()
            ln = len(cond_binomial) - lp
            row['pvalue']= binom_test([lp, ln])
            row['score']=1-(row['pvalue'])
    elif test == "count":  
        row['meanA'] = distA.dropna().mean(skipna = True)
        row['meanB'] = distB.dropna().mean(skipna = True)
        row['comparison'] = get_comparison(countA, countB, comparison_dict)
        row['score_type'] = 'count'
        row['percentage'] = get_percentage(countA, countB)
        row['difference'] = get_difference(countA, countB)
        lp = countA
        ln = countB
        row['pvalue']= binom_test([lp, ln])
        row['score']=1-(row['pvalue'])
        row['completeness_factor'] = 1
    row['pscore_nonmissing'] = (row['completeness_factor'] * (1-row['pvalue']))
    row['difference_weight'] = get_difference_weight(row['difference'], metric_name, measurement_tolerance)
    row['pscore_final'] = row['pscore_nonmissing'] * (row['difference_weight'])
    row['meanB_'] = row['meanB']
    if isinstance(row['meanB_'],str):
        meanB_condidate = re.findall('([0-9]+[.]*[0-9]*)', row['meanB_'])
        if row['meanB_']!='':
            row['meanB_'] = float(re.findall('([0-9]+[.]*[0-9]*)', row['meanB_'])[0])
        else: 
            row['meanB_'] = 0.0
    row['distributional_feature'] = [row['meanA'], row['meanB_']]
    row['neural_relevance_score'] = 0
    if row['comparison']!="<<unknown>>":
        splits = row['comparison'].split(' ')
        if len(splits) == 2:
            row['comparison_pre'], row['comparison_post']  = row['comparison'].split(' ')
    row['rough_insight_text'] = insight_text.replace("""{{comparison}}""",row['comparison']).replace("""{{comparison_pre}}""",row['comparison_pre']).replace("""{{comparison_post}}""",row['comparison_post'])
    return row

def get_categorisation_if_exist(sys_definition_file):
    try:
        categorisation_def = pd.read_excel(sys_definition_file, sheet_name=['categorisation'],engine='openpyxl')
        categorisation_def['categorisation'] = categorisation_def['categorisation'].dropna(axis=0, how='all')
    except:
        categorisation_def = None
    return categorisation_def

def inject_categorical_columns(dat, cat):
    if cat is not None:
        cat_def = cat['categorisation']
        for ind, row in cat_def.iterrows():
            ranges = row['ranges']
            meas = row['measurement']
            options = row['categories']
            as_col = row['as']
            dat[as_col] = dat[meas]
            for ((low, high),category_str) in zip(eval(ranges),eval(options)):
                dat.loc[(dat[meas]>=low) & (dat[meas]<=high),as_col] = category_str
    return dat

def exception_print(e,remedy,exit=True):
    print('original error message:',e)
    print('possible resolution:')
    print(remedy)
    print('aborting...')
    if exit:
        sys.exit(0)

def get_data(qA, data, dummy_data, now_details):
    for k,v in now_details.items():
        command = "{}={}".format(k,v)
        exec(command)
    try:
        distA = eval(qA)
    except KeyError as e: 
        print(qA)
        remedy = "please check the names of columns in the system_definition and adata file match."
        exception_print(e,remedy, exit=False)
        distA = pd.Series([],dtype=float)
    distA = distA.dropna()
    distA = distA[distA!=0]
    countA = len(distA)
    return distA, countA


def score_one_row(params):
    row = params[0]
    srow = get_stats(params)
    for k,v in srow.items():
        row[k] = v
    return row
# %%
if __name__ == "__main__":


    preprocess_script_path = 'systems.{}.preprocess_data'.format(system_name)
    preprocess_location = preprocess_script_path.replace('.','/')+'.py'
    if os.path.exists(preprocess_location):
        print('detected preprocessing script') if verbose else None
        import importlib
        preprocess_data = importlib.import_module(preprocess_script_path)
        preprocess_script_exists = True
    else:
        preprocess_script_exists = False

    #cred_df = pd.read_excel('user_info.xlsx', engine='openpyxl')
    #cred_df = cred_df.dropna(subset=['active'])


    input_file_type = data_file_format
    print(scope_name) if verbose else None
    if scope_name:
        print("scoring insights of scope: {}".format(scope_name)) if verbose else None
    else:
        print("scoring insights of general scope") if verbose else None
    data = read_data(system_name, input_file_type, scp_name=scope_name)
    dummy_data = create_dummy_data(data.date, date_model, system_name)
    library = read_library(system_name, input_file_type)
    system_definition_file = "systems/{}/system_definition_{}.xlsx".format(system_name,system_name)
    categorisation = get_categorisation_if_exist(system_definition_file)    
    data = inject_categorical_columns(data, categorisation)
    sheet_names = ['schemas','measurements','contexts','exclusions']                    
    system_definition = pd.read_excel(system_definition_file, sheet_name=sheet_names,engine='openpyxl')
    for sheet in sheet_names:
        system_definition[sheet] = system_definition[sheet].dropna(axis=0, how='all')
    measurement_definition = system_definition['measurements']
    measurement_list = measurement_definition['measurement'].to_list()
    measurement_phrase = {k:v for k,v in zip(measurement_list, measurement_definition['phrase'].to_list())}
    measurement_benchmark = {k:v for k,v in zip(measurement_list, measurement_definition['benchmark'].to_list())}
    measurement_benchmark_text = {k:v for k,v in zip(measurement_list, measurement_definition['benchmark text'].to_list())}
    measurement_comparison_phrase = {k:v for k,v in zip(measurement_list, measurement_definition['regular comparative phrase'].to_list())}
    measurement_group = {k:v for k,v in zip(measurement_list, measurement_definition['group_by'].to_list())}
    measurement_tolerance = {k:v for k,v in zip(measurement_list, measurement_definition['tolerance'].to_list())}
    benchmark_comparison_phrase = {k:v for k,v in zip(measurement_list, measurement_definition['benchmark comparative phrase'].to_list())}
    now = datetime.datetime.fromtimestamp(data['date'].max()).replace(tzinfo=tz.gettz(time_zone))
    now_details = get_now_details(now)     
    params = data, dummy_data, measurement_benchmark_text, benchmark_comparison_phrase, measurement_comparison_phrase, measurement_tolerance, now_details, measurement_benchmark
    stat_df = []
    scored_library = None

    if parallel == False:
        for ind, row in library.iterrows():
            stat_df.append(score_one_row((row, params)))
        
    elif parallel == True:
        with multiprocessing.Pool(20) as p:
            stat_df.append(p.map(score_one_row, [(row, params) for (indr,row) in library.iterrows()] ))    
        stat_df = stat_df[0]
        # scored_library = pd.concat(stat_df, ignore_index = True)

    for row in stat_df:
            if scored_library is None:
                scored_library = pd.DataFrame([row])
            else:
                scored_library = pd.concat([scored_library, pd.DataFrame([row])])

    if scope_name:
        scored_library.to_excel("systems/{}/scored_insights_{}_{}.xlsx".format(system_name,scope_name,system_name))
        scored_library.to_pickle("systems/{}/scored_insights_{}_{}.pickle".format(system_name,scope_name,system_name), protocol=4)
    else:
        scored_library.to_excel("systems/{}/scored_insights_{}.xlsx".format(system_name,system_name))
        scored_library.to_pickle("systems/{}/scored_insights_{}.pickle".format(system_name,system_name), protocol=4)
# %%