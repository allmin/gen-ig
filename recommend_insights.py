import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sympy import plot
np.random.seed(123)
# from pattern.en import conjugate
from config import system_name, scope_name, unknown_phrase, tense_dict, reserved_keywords, recommendation_count
from utils.commonlib import unique, flatten, streval
from sklearn.cluster import KMeans
# import regex as re
# import os, sys
# scope_name = 'user02'
def first_caps(str1):
    """
    capitalises the first character of a string sentence.
    """
    if str1 == '':
        return str1
    return str1[0].upper() + str1[1:]


def perform_kmeans(K,X):
    if K == 0:
        return []
    kmeans = KMeans(n_clusters=K, random_state=123).fit(X)
    return kmeans.labels_


def plot_insight(row):
    FSi = 20
    if not os.path.exists(f"{output_folder}/plots/"):
        os.makedirs(f"{output_folder}/plots/")
    title = row.insight_text_final.replace(', ',',\n')
    plt.figure(figsize=(11,10))
    if row.test in ['benchmark','stat']:
        plt.boxplot([row.distA.values], vert = False)
        
        plt.yticks(ticks = [1], labels = [row.textA], wrap=True, fontsize=18, rotation='vertical', va="bottom", ha="center")
        if row.test == 'benchmark':
            plt.vlines(row.meanB_, 0.5,1.5)
            plt.text(row.meanB_,2.5,"pre-set benchmark", rotation="vertical", ha="center", va="center")
    else:        
        p1 = row.distA.values
        p2 = row.distB.values
        plt.boxplot([p2, p1], vert = False)
        plt.yticks(ticks = [1,2], labels = [first_caps(row.textB), first_caps(row.textA)], wrap=True, fontsize=18, rotation='vertical', va="bottom", ha="center")
    plt.title(title, wrap=True, fontsize=FSi)
    plt.xlabel(first_caps(row.meas_text), fontsize=FSi)
    plt.ylabel(first_caps(row.common_text), fontsize=FSi, va="bottom")
    print(f"{output_folder}/plots/plot_{row.name}.png")
    plt.savefig(f"{output_folder}/plots/plot_{row.name}.png")


def save_to_system_folder(df,prefix):
    if scope_name and scope_name != '':
        df.to_excel("systems/{}/outputs/{}_{}_{}.xlsx".format(system_name,prefix,scope_name,system_name), index=False)
        df.to_pickle("systems/{}/outputs/{}_{}_{}.pickle".format(system_name,prefix,scope_name,system_name), protocol=4)
    else:
        df.to_excel("systems/{}/outputs/{}_{}.xlsx".format(system_name,prefix,system_name), index=False)
        df.to_pickle("systems/{}/outputs/{}_{}.pickle".format(system_name,prefix,system_name), protocol=4)


if __name__ == '__main__':
    sorting_criteria = 'pscore_final'
    output_folder = f"systems/{system_name}/outputs"
    if scope_name and scope_name != '':
        scored_library = pd.read_pickle(f"{output_folder}/scored_insights_final_{scope_name}_{system_name}.pickle")
    else:
        scored_library = pd.read_pickle(f"{output_folder}/scored_insights_final_{system_name}.pickle")

    # system_definition_file = "systems/{}/system_definition_{}.xlsx".format(system_name,system_name)
    # sheet_names = ['schemas','measurements','contexts','exclusions']                    
    # system_definition = pd.read_excel(system_definition_file, sheet_name=sheet_names,engine='openpyxl')
    # measurement_definition = system_definition['measurements']
    # measurement_list = measurement_definition['measurement'].to_list()
    # measurement_phrase = {k:v for k,v in zip(measurement_list, measurement_definition['phrase'].to_list())}
    scored_library['complexity'] = [len(s.split(' ')) for s in scored_library['insight_text_final']]
    list_of_filters = unique(flatten([list(streval(i).keys()) for i in scored_library['intermediate']]))
    list_of_filters_AB = flatten([[i+'_A', i+'_B'] if i!='measurement' else [i] for i in list_of_filters])

    insight_identifier = 'schema'+scored_library['schema_num'].astype(str)+'|'+'complexity'+scored_library['complexity'].astype(str)+'|'+scored_library['comparison_pre'].astype(str)+'|'
    for col in list_of_filters_AB:
        if len(insight_identifier) != 0:
            insight_identifier = insight_identifier + '|' + scored_library[col].astype(str)

    scored_library['insight_identifier'] = insight_identifier

    vocab_path = 'systems/{}/outputs/vocab.xlsx'.format(system_name)
    if os.path.exists(vocab_path):
        use_vocab = True
        vocab = pd.read_excel(vocab_path, engine = 'openpyxl')
        vocab_keys = vocab['keys'].values
        numv = len(vocab)-1
        vocabulary = {k:v for (k,v) in zip(vocab['keys'].tolist(), vocab['indices'].tolist())}
    else:
        use_vocab = False
        vocabulary = {'TOKEN-NA':0}
        numv = 0

    if use_vocab == False:
        for i in insight_identifier.values:
            for j in i.split('|'):
                for k in j.split('$'):
                    if (k not in vocabulary) and (k!=''):
                        numv += 1
                        vocabulary[k] = numv
        vocab = pd.DataFrame({'keys':list(vocabulary.keys()),'indices':list(vocabulary.values())})
        vocab.to_excel(vocab_path,engine='openpyxl',index=False)

    else:
        for i in insight_identifier.values:
            for j in i.split('|'):
                for k in j.split('$'):
                    if (k not in vocabulary) and (k!=''):
                        numv += 1
                        vocabulary[k] = numv
        vocab.to_excel(vocab_path.replace('.xlsx','_old.xlsx'),index = False,engine='openpyxl')
        vocab = pd.DataFrame({'keys':list(vocabulary.keys()),'indices':list(vocabulary.values())})
        vocab.to_excel(vocab_path,index = False,engine='openpyxl')
    

    feature_vector = np.zeros((len(insight_identifier),len(vocabulary)))
    na_tokens = []
    for ind,i in enumerate(insight_identifier.values):
        for j in i.split('|'):
            for k in j.split('$'):
                if k == '':
                    continue
                if k in vocabulary:
                    k_ind = vocabulary[k]
                else:
                    k_ind = vocabulary['TOKEN-NA']
                    if k not in na_tokens:
                        na_tokens.append(k)
                        print("token '{}' unavailable in vocabulary replaced with NA".format(k))                        
                feature_vector[ind][k_ind] += 1
    scored_library['feature_vector'] = [i for i in feature_vector]
    
    # elliminating insignificant insights:
    scored_library = scored_library[scored_library['pvalue'] < 0.05]
    feature_vector = np.array([i for i in scored_library['feature_vector'].values])

    num_insights_needed = min(recommendation_count,len(scored_library))
    cluster_labels = perform_kmeans(K=num_insights_needed,X=feature_vector)
    if len(cluster_labels)>0:
        scored_library['cluster_labels'] = cluster_labels
        
        
    save_to_system_folder(df=scored_library,prefix='scored_insights_final_with_clusters')
    
    
    recommended_insights = pd.DataFrame()
    for i in range(num_insights_needed):
        insight_candidates = scored_library[scored_library['cluster_labels'] == i]
        insight_candidate = insight_candidates[insight_candidates[sorting_criteria] == (insight_candidates[sorting_criteria].max())]
        if len(insight_candidate) > 1:
                insight_candidate = shuffle(insight_candidate).iloc[:1]
        if len(recommended_insights) == 0:
            recommended_insights = insight_candidate
        else:
            recommended_insights = pd.concat([recommended_insights,insight_candidate])
    
    
    save_to_system_folder(df=recommended_insights,prefix='recommended_insights_final')

    recommended_insights.index = recommended_insights.index + 1
    recommended_insights.index.name = 'Number'
    recommended_insights['Insight'] = recommended_insights['insight_text_final']
    
    for num,insight_data in recommended_insights.iterrows():
        plot_insight(insight_data)
    
    insight_short = recommended_insights[['Insight']].copy()

    dest_name = "systems/{}/outputs/insights_short.xlsx".format(system_name)

    insight_short.to_excel(dest_name)
    df_volunteer = insight_short.copy()
    pd.set_option('display.max_colwidth', 100)
    df_volunteer.to_excel(dest_name, index=True, engine='xlsxwriter')

    #validation columns
    writer = pd.ExcelWriter(dest_name, engine='xlsxwriter') 
    df_volunteer.to_excel(writer, sheet_name='Sheet1', index = True)

    #Assign the workbook and worksheet
    workbook  = writer.book
    worksheet = writer.sheets['Sheet1']

    #Adding the header and Datavalidation list
    for col,title,srclist in [('C','Usefulness*',['Not Useful at all','Not Useful','Neutral','Useful','Very Useful']),('D','Accuracy*',['Incorrect', 'Correct']),('E','Readability*', ['Not clear', 'Easy to understand'])]:
        worksheet.write('{}{}'.format(col,1), title)
        for i in range(20):
            worksheet.data_validation('{}{}'.format(col,i+2), {'validate': 'list', 'source': srclist})
    
    format = workbook.add_format({'text_wrap': True})
    # Setting the format but not setting the column width.
    worksheet.set_column('B:B', 80, format)
    worksheet.set_column('C:C', 14, format)
    worksheet.set_column('D:D', 10, format)
    worksheet.set_column('E:E', 14, format)
    worksheet.write('F1', 'Textual feedback (Please provide feedback on this insight below)')
    worksheet.set_column('F:F', 30, format)
    workbook.close()


    try:
        if system_name == 'iphil' and os.getcwd().split(os.path.sep)[-1] == 'get_insights':
            prefix = 'scored_insights_final_with_clusters'
            scored_library.to_pickle("../../actions/data/{}_{}_{}.pickle".format(prefix,scope_name,system_name), protocol=4)
    except Exception as e:
        print(e)

