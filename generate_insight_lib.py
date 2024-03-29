# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:25:49 2019

@author: 310276324
"""

#on mondays, you fall sleep earlier than on other days

#%%

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import datetime, itertools
import regex as re
from scipy.stats import norm
from pattern.en import conjugate
from scipy.stats import ks_2samp, mannwhitneyu
import pdb , os, shutil, hashlib, sys
from config import system_name, reserved_keywords, tense_dict, verbose, dummy_group




def check_if_exist(fil):
    if not os.path.exists(fil):
        print("please check if {} exists. if not, refer to the documentation".format(fil))
        sys.exit(1)

def intellicat_dict(l1,l2):
    res = []
    for i in l1:
        for j in l2:
            i_copy = i.copy()
            i_copy.update(j)
            res.append(i_copy)
    return(res)


def get_consecutive_combinations(candidates): 
    index_hash = {entity:ind for (ind, entity) in enumerate(candidates)}
    max_ind = max(index_hash.values())
    first_estimate = itertools.combinations(candidates,2)
    second_estimate = []
    for a,b in first_estimate:
        a_ind = index_hash[a]
        b_ind = index_hash[b]
        diff_ind = abs(a_ind - b_ind)
        if  diff_ind == 1 or diff_ind == max_ind:
            second_estimate.append((a,b))
    if len(second_estimate) == 0 and len(candidates) > 0: #(Conditions when all the candidates are same)
        multiplier = len(candidates)
        if multiplier == 2:
            multiplier = 1
        second_estimate = [(candidates[0], candidates[0])]*multiplier
    return second_estimate

def get_all_combinations(candidates):
    return itertools.combinations(candidates,2)
        


def get_single_enumerations(context_definition, context_sheet, applicable_items, only_invertible = False):
    c_dictionary = context_definition[context_sheet]
    context_name = context_sheet[2:] if context_sheet.startswith('c_') else context_sheet
    enumerations = {'names':[], 'phrases':[], 'queries':[], 'inversion_phrases':[], 'inclusive_phrases':[], 'inverse_queries':[], 'tenses':[]}
    if context_name in applicable_items:
        select_items = applicable_items[context_name]
    else:
        select_items = None
    for ind, context in c_dictionary.iterrows():
        if (select_items != None) and (context['name'] not in select_items):
            continue
        if only_invertible:
            if context['invertible'] == False:
                continue
        context_name = context['name']
        if not isinstance(context_name,str):
            continue
        entities = eval(context['entities'])
        num_entities = len(entities)
        entity_names = [context_name+'$'+str(i) for i in eval(context['entities'])]
        entity_phrases = careful_eval(context, 'prepositional phrase', entities)
        entity_tense = careful_eval(context, 'tense', entities, 'PaPC')
        entity_queries = careful_eval(context, 'query', entities)
        inversion_phrase = careful_eval(context, 'inversion phrase', entities)
        inversion_query = careful_eval(context, 'inverse query', entities)
        enumerations['names'].extend(entity_names)
        enumerations['phrases'].extend(entity_phrases)
        enumerations['tenses'].extend(entity_tense)
        enumerations['queries'].extend(entity_queries)
        enumerations['inversion_phrases'].extend(inversion_phrase)
        enumerations['inverse_queries'].extend(inversion_query)
    return enumerations

def careful_eval(contxt, col, entities=None, default=''):
    if entities:
        num_entities = len(entities)
    else:
        num_entities =  1

    if col in contxt:
        exp = contxt[col]
    else:
        exp = [default]*num_entities

    if type(exp)!=str:
        res =  ['']
    else:
        exp = exp.replace("\xa0",' ')
        try:
            res = eval(exp)
        except:
            res = [exp]
    if res is None:
        res = ['']*num_entities
        print(f"Warning: define a inversion_phrase for all rows in the context:{contxt}")
    if len(res)!= num_entities:  
        print("unequal exp {} for number of entities: {} of type: {} ".format(res,num_entities,col))
        res =  res*num_entities
    return res
        

def get_double_enumerations(context_definition, context_sheet, applicable_items):
    c_dictionary = context_definition[context_sheet]
    unique_pairs = list(c_dictionary['pair id'].unique())
    for pair_id in unique_pairs:
        c_dictionary_subset = c_dictionary[c_dictionary['pair id'] == pair_id]

    enumerations = {'names':[], 'phrases':[], 'tenses':[], 'queries':[], 'pair_id':[], 'consecutive_constrain':[]}
    for ind, context in c_dictionary.iterrows():
        context_name = context['name']
        if not isinstance(context_name,str):
            continue
        pair_id = context['pair id']
        consecutive_constrain = context['consecutive constrain']
        entities = eval(context['entities'])
        num_entities = len(entities)
        pair_id = [context['pair id']]*num_entities
        consecutive_constrain = [context['consecutive constrain']]*num_entities
        entity_names = [context_name+'$'+str(i) for i in entities]
        entity_phrases = careful_eval(context, 'prepositional phrase', entities)
        if 'tense' in context:
            entity_tenses = careful_eval(context, 'tense', entities, 'PaPC')
        else:
            entity_tenses = ['PaPC']*len(entities)
        entity_queries = careful_eval(context, 'query', entities)
        enumerations['names'].extend(entity_names)
        enumerations['phrases'].extend(entity_phrases)
        enumerations['tenses'].extend(entity_tenses)
        enumerations['queries'].extend(entity_queries)
        enumerations['pair_id'].extend(pair_id)
        enumerations['consecutive_constrain'].extend(consecutive_constrain)
    enumeration_df = pd.DataFrame(enumerations)
    double_enumerations = {'names':[], 'phrases':[], 'tenses':[], 'queries':[]}
    for p_id in unique(enumeration_df['pair_id'].tolist()): #only pairable entities are considered
        pairable_rows = enumeration_df[enumeration_df['pair_id']==p_id]
        for con_constrain in unique(pairable_rows['consecutive_constrain']):
            selected_rows = pairable_rows[pairable_rows['consecutive_constrain'] == con_constrain]
            name_candidates = selected_rows['names'].to_list()
            phrase_candidates = selected_rows['phrases'].to_list()
            tense_candidates = selected_rows['tenses'].to_list()
            query_candidates = selected_rows['queries'].to_list()
           
            if con_constrain == False:                
                name_pairs = get_all_combinations(name_candidates)
                phrase_pairs = get_all_combinations(phrase_candidates)
                tense_pairs = get_all_combinations(tense_candidates)
                query_pairs = get_all_combinations(query_candidates)
            
            elif con_constrain == True:
                name_pairs = get_consecutive_combinations(name_candidates)
                phrase_pairs = get_consecutive_combinations(phrase_candidates)
                tense_pairs = get_consecutive_combinations(tense_candidates)
                query_pairs = get_consecutive_combinations(query_candidates)
            
            double_enumerations['names'].append(name_pairs)
            double_enumerations['phrases'].append(phrase_pairs)
            double_enumerations['tenses'].append(tense_pairs)
            double_enumerations['queries'].append(query_pairs)
    
    for key in double_enumerations.keys():
        double_enumerations[key] = flatten(double_enumerations[key])
        
    return double_enumerations

def get_inverse_query(q):
    return q.replace("<=",">").replace(">=","<").replace("==","!=")

def generate_intermediate_insight(first_context, second_context, common_context, change_type, measurements, context_definition, applicable_items):
    """
    generate intermediate insights given the common and uncommon contexts
    """

    comb_properties = ['names', 'phrases', 'tenses', 'queries']
    combinations_prop_dict = {i:[{}] for i in comb_properties}
    
    for i in common_context: #common_context situations
        context_sheet = "c_"+i
        enumerations = get_single_enumerations(context_definition, context_sheet, applicable_items)
        for p in comb_properties:
            e_props = [{i:j} for j in enumerations[p]]
            combinations_prop_dict[p] = intellicat_dict(combinations_prop_dict[p], e_props)


    if first_context[0] == second_context[0]: #comparative_context -> combinatorial insights
        first_context_l = first_context[0]
        context_sheet = "c_"+first_context_l
        enumerations = get_double_enumerations(context_definition, context_sheet, applicable_items)
        for p in comb_properties:
            e_props = [{first_context_l:j} for j in enumerations[p]]
            combinations_prop_dict[p] = intellicat_dict(combinations_prop_dict[p], e_props)

    elif change_type == "exclusion": #comparative_context -> exclusion insights
        first_context_l = first_context[0]
        second_context_l = second_context[0]
        context_sheet = "c_"+first_context_l
        enumerations = get_single_enumerations(context_definition, context_sheet, applicable_items, only_invertible=True)
        e_props_dict = {}
        e_props_dict['names'] = [{first_context_l:(e_name, '!'+e_name)} for e_name in enumerations['names']]
        e_props_dict['phrases'] = [{first_context_l:(e_phrase, e_inv_phrase)} for (e_phrase, e_inv_phrase) in zip(enumerations['phrases'], enumerations['inversion_phrases'])]
        e_props_dict['tenses'] = [{first_context_l:(e_tense, e_tense)} for e_tense in enumerations['tenses']]
        e_props_dict['queries'] = [{first_context_l:(e_query, inverse_query)} for (e_query, inverse_query) in zip(enumerations['queries'], enumerations['inverse_queries'])]
        for p in comb_properties:
            combinations_prop_dict[p] = intellicat_dict(combinations_prop_dict[p], e_props_dict[p])

    elif second_context[0] in ["measurement_benchmark","none"]: #comparative_context -> benchmark insights
        keyword = "{{" + second_context[0] + "}}"
        first_context_l = first_context[0]
        second_context_l = second_context[0]
        context_sheet = "c_"+first_context_l
        enumerations = get_single_enumerations(context_definition, context_sheet, applicable_items)
        e_props_dict = {}
        e_props_dict['names'] = [{first_context_l:(e_name, keyword)} for e_name in enumerations['names']]
        e_props_dict['phrases'] = [{first_context_l:(e_phrase, keyword)} for (e_phrase, e_inv_phrase) in zip(enumerations['phrases'], enumerations['inversion_phrases'])]
        e_props_dict['tenses'] = [{first_context_l:(e_tense, keyword)} for e_tense in enumerations['tenses']]
        e_props_dict['queries'] = [{first_context_l:(e_query, keyword)} for e_query in enumerations['queries']]
        for p in comb_properties:
            combinations_prop_dict[p] = intellicat_dict(combinations_prop_dict[p], e_props_dict[p])
    

    e_measurements = [{'measurement':measurement} for measurement in measurements]
    for p in comb_properties:
        combinations_prop_dict[p] = intellicat_dict(combinations_prop_dict[p],e_measurements)

    return combinations_prop_dict


def clear_folder(pt):
    if os.path.exists(pt):
        shutil.rmtree(pt)
    os.makedirs(pt)

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()

def append_to_file(fn,ln):
    with open(fn, 'a') as myfile:
        myfile.write(ln+'\n')

def get_cnl(row, comparison, measure='default', flip_flag=False):
    """
    gets the cnl
    """
    st = eval(row['intermediate'])
    contextA = st[:-1]
    contextB = st[:-2]
    contextB.append(st[-1])
    if measure == 'default':
        measure = row['measurement']
    meanA = row['meanA']
    meanB = row['meanB']
    c_measure = (meanA - meanB) / meanB
    contextA_str = " [SEP] ".join(get_components(contextA))
    contextB_str = " [SEP] ".join(get_components(contextB))
    cnl_text = """[MEASURE 1] {} [SEP] [CONTEXT 1] {} [SEP] {:0.2f} {} {:0.2f} [SEP] [CONTEXT 2] {}""".format(measure,contextA_str, meanA, comparison, meanB, contextB_str)
    cnl_text = cnl_text.replace('user_id', 'user').replace('°C',' degree')
    return (cnl_text)


def unique(lst):
    set1 = set(lst)
    lst = list(set1)
    return lst

def c_intersection(ls1, ls2, additional_contexts=[]):
    return [i for i in ls1 if ((i in ls2) or (i.lstrip('!') in ls2) or (i in additional_contexts)) ]

def flatten(t):
    return [item for sublist in t for item in sublist]

def first_caps(str1):
    return str1[0].upper() + str1[1:]

def flatten_and_append(lst,app):
    if type(app) == list:
        lst.extend(app)
    else:
        lst.append(app)

def flexi_eval(sinp):
    try:
        ev = eval(sinp)
        return ev
    except:
        if type(sinp)==str and (sinp.startswith("'") or sinp.startswith("!")):
            return sinp
        else:
            return "'{}'".format(sinp)

def get_query(inter_query, inter_groups):
    # groupA_list = []
    # groupB_list = []
    queryA = "data["
    queryB = "data["
    for k,v in inter_query.items():
        if k != "measurement":
            if len(v) == 2:
                queryA += '(' + v[0] + ') &'
                queryB += '(' + v[1] + ') &'
                # flatten_and_append(groupA_list,inter_groups[k][0])
                # flatten_and_append(groupB_list,inter_groups[k][1])
            else:
                queryA += '(' + v + ') &'
                queryB += '(' + v + ') &'
                # flatten_and_append(groupA_list, inter_groups[k])
                # flatten_and_append(groupB_list, inter_groups[k])
    queryA = queryA.rstrip(' &')
    queryB = queryB.rstrip(' &')
    # unique_groupA = unique(groupA_list)
    # unique_groupB = unique(groupB_list)
    meas = measurement_columns[inter_query['measurement']]
    groupAB = flexi_eval(measurement_group[inter_query['measurement']])
    groupAB = clean_grouping_for_real(groupAB) 
    agg_fun = measurement_aggregate[inter_query['measurement']]
    numeric_str = "numeric_only=True" if agg_fun!='count' else ""
    queryA += """].groupby(by={}).{}({})['{}']""".format(str(groupAB),agg_fun,numeric_str, meas)
    queryB += """].groupby(by={}).{}({})['{}']""".format(str(groupAB),agg_fun,numeric_str, meas)
    if 'measurement_benchmark' in queryB:
        queryB = "measurement_benchmark['{}']".format(inter_query['measurement'])
    if 'measurement_benchmark' in queryB:
        queryB = "measurement_benchmark['{}']".format(inter_query['measurement'])
    # queryB = queryB.replace("""data[({{measurement_benchmark}})]""","measurement_benchmark").replace(""".groupby(by=['{{measurement_benchmark}}'])."""+agg_fun+"()","")
    return queryA, queryB

def clean_grouping_for_dummy(gp):
    # doesnt include any groups with a !, 
    if isinstance(gp,str):
        gp = [gp]
    if dummy_group:
        gp = [i for i in gp if i in dummy_group]
    final_gps = []
    for g in gp:
        if g.startswith('!'):
            continue
        final_gps.append(g)
    if len(final_gps) == 1:
        final_gps = final_gps[0]
        if final_gps[0]!="'":
            final_gps = "'{}'".format(final_gps)
    return final_gps

def clean_grouping_for_real(gp):
    if isinstance(gp,str):
        gp = [gp]
    final_gps = []
    for g in gp:
        if g.startswith('!'):
            g = g[1:]
        final_gps.append(g)
    if len(final_gps) == 1:
        final_gps = final_gps[0]
        if final_gps[0]!="'":
            final_gps = "'{}'".format(final_gps)
    return final_gps

def get_dummy_query(inter_query, inter_groups):
    """
    get only the 'period' related filters
    """
    # groupA_list = []
    # groupB_list = []
    queryA = "data["
    queryB = "data["
    for k,v in inter_query.items():
        if k != "measurement":
            if len(v) == 2:
                if k.startswith('period'):
                    queryA += '(' + v[0] + ') &'
                    queryB += '(' + v[1] + ') &'
                # flatten_and_append(groupA_list,inter_groups[k][0])
                # flatten_and_append(groupB_list,inter_groups[k][1])
            else:
                if k.startswith('period'):
                    queryA += '(' + v + ') &'
                    queryB += '(' + v + ') &'
                # flatten_and_append(groupA_list,inter_groups[k])
                # flatten_and_append(groupB_list,inter_groups[k])
    queryA = queryA.rstrip(' &')
    queryB = queryB.rstrip(' &')
    meas = measurement_columns[inter_query['measurement']]
    groupAB = flexi_eval(measurement_group[inter_query['measurement']])
    groupAB = clean_grouping_for_dummy(groupAB) 
    queryA = queryA.replace('data','dummy_data')
    queryB = queryB.replace('data','dummy_data')
    # unique_groupA = unique(groupA_list)
    # unique_groupB = unique(groupB_list)
    if groupAB != []:
        queryA += """].groupby(by={}).sum()""".format(str(groupAB))
        queryB += """].groupby(by={}).sum()""".format(str(groupAB))
    else:
        queryA += ']'
        queryB += ']'
    if "measurement_benchmark" in queryB:
        queryB = "dummy_measurement_benchmark"
    # queryB = queryB.replace("""data[({{measurement_benchmark}})]""","measurement_benchmark").replace(""".groupby(by=['{{measurement_benchmark}}']).sum()""","")
    if 'dummy_data[]' in queryA:
        queryA = queryA.replace('dummy_data[]','dummy_data')
    if 'dummy_data[]' in queryB:
        queryB = queryB.replace('dummy_data[]','dummy_data')
    return queryA, queryB

def transform_tense(x, tense):
    tense_tr = tense_dict[tense]
    return conjugate(x, tense = tense_tr, person = 3, number = "singular", mood = "indicative", aspect = "imperfective", negated = False)

def tensify(inp_str,tense):
    """
    converts a string into the specified tense.
    applies only if the inp_str has a "{{tense(word)}}" segment
    """
    place_holder_format1a = re.compile("""{{tense.*?}}""")
    place_holder_format1b = re.compile("""{tense.*?}""")
    place_holder_format2 = re.compile("""\(.*?\)""")
    tense_transform_candidates = place_holder_format1a.findall(inp_str) + place_holder_format1b.findall(inp_str)
    if tense_transform_candidates:
        for tense_transform_candidate in tense_transform_candidates:
            verbs = place_holder_format2.findall(tense_transform_candidate)
            if verbs:
                for verb in verbs:
                    verb = verb.rstrip(")").lstrip("(")
                    transformed_verb = transform_tense(verb,tense)
                    inp_str = inp_str.replace(tense_transform_candidate, transformed_verb)
    return inp_str

                
def get_sentence(inter_phrase, inter_tense, template, measurement_phrase, measurement_benchmark_text):
    insight_text = template.replace("!","")
    tense_list = []
    for ((k,v),(tk,tv)) in zip(inter_phrase.items(), inter_tense.items()):
        if len(v) == 2: # Takes care of the comparative context
            find1 = "{{" + "{}:1".format(k) + "}}"
            replace1 = tensify(v[0],tv[0]) 
            tense_list.append(tv[0])
            if v[1] == """{{measurement_benchmark}}""":
                find2 = "{{" + "measurement_benchmark:2" + "}}"
                replace2 = str(measurement_benchmark_text[inter_phrase['measurement']])
            else:
                find2 = "{{" + "{}:2".format(k) + "}}"
                replace2 = tensify(v[1],tv[1]) 
            insight_text = insight_text.replace(find1,replace1).replace(find2,replace2)
        else: # Takes care of the common context
            find = "{{" + "{}".format(k) + "}}"
            if k not in reserved_keywords:
                replace = tensify(v,tv)
            elif k == "measurement":
                replace = tensify(measurement_phrase[v],tv)
            insight_text = insight_text.replace(find,replace)   
            tense_list.append(tv)
    if "SP" in tense_list:
        overall_tense = "SP"
    elif "SPa" in tense_list:
        overall_tense = "SPa"
    else:
        overall_tense = "PaPC"
    insight_text = first_caps(insight_text)
    insight_text = tensify(insight_text,overall_tense)
    return insight_text
    
def get_sentence_raw(inter_phrase, template, measurement_phrase, measurement_benchmark_text):
    insight_text = template.replace("!","")
    for k,v in inter_phrase.items():
        if len(v) == 2:
            find1 = "{{" + "{}:1".format(k) + "}}"
            replace1 = v[0]
            if v[1] == """{{measurement_benchmark}}""":
                find2 = "{{" + "measurement_benchmark:2" + "}}"
                replace2 = str(measurement_benchmark_text[inter_phrase['measurement']])
            else:
                find2 = "{{" + "{}:2".format(k) + "}}"
                replace2 = v[1]
            insight_text = insight_text.replace(find1,replace1).replace(find2,replace2)
        else:
            find = "{{" + "{}".format(k) + "}}"
            if k!="measurement":
                replace = v
            else:
                replace = measurement_phrase[v]
            insight_text = insight_text.replace(find,replace)   
    insight_text = first_caps(insight_text)
    return insight_text


def eval_dict(dict1):
    if type(dict1)!=str and np.isnan(dict1):
        return {}
    else:
        return eval(dict1)

def exception_print(e,remedy):
    print('original error message:',e)
    print('possible resolution:')
    print(remedy)
    print('aborting...')
    sys.exit(0)

def parse_schema_classes(system_definition, ignore_placeholders):
    measurement_list = system_definition['measurements']['measurement'].to_list() 
    class_dict = {'schema_num':[], 'first_context':[], 'second_context':[], 'common_context':[], 'scoring_type':[], 'change_type':[], 'measurements':[], 'schema_template':[], 'components':[], 'tag':[], 'applicable_items':[]} # empty schema collection
    for ind, schema in system_definition['schemas'].iterrows():
        schema_num = int(schema['schema num'])
        scoring_type = schema['scoring type']
        schema_template = schema['template']
        applicable_items = eval_dict(schema['applicable_items'])
        class_dict['applicable_items'].append(applicable_items)
        measurements = applicable_items['measurement'] if 'measurement' in applicable_items else measurement_list
        class_dict['schema_num'].append(schema_num)
        class_dict['measurements'].append(measurements)
        class_dict['schema_template'].append(schema_template)
        place_holder_format = re.compile("{{(.*?)}}")
        components = place_holder_format.findall(schema_template)
        components = [i for i in components if i not in ignore_placeholders]
        class_dict['components'].append(components)
        if not np.isnan(schema['silent_contexts']):
            class_dict['components'].extend(schema['silent_contexts'])
        tag = schema['tag'] if isinstance(schema['tag'],str) else ""
        class_dict['tag'].append(tag)
        class_dict['scoring_type'].append(scoring_type)
        common_context = []
        first_context = []
        second_context = []
        for component in components:
            if ":" in component:
                component, position = component.split(":")
                if position.isnumeric():
                    if position == "1":
                        first_context.append(component)
                    elif position == "2":
                        second_context.append(component)
            else:
                common_context.append(component)
        common_context = c_intersection(common_context, context_list)
        first_context = unique(c_intersection(first_context, context_list))
        second_context = unique(c_intersection(second_context, context_list, additional_contexts=['measurement_benchmark', 'none']))
        try:
            if second_context[0].startswith("!") and first_context[0] == second_context[0][1:]:
                change_type = "exclusion"
            else:
                change_type = "combination"
        except IndexError as e:
            remedy = "please check if the schema template has two comparative contexts :1 and :2 and also check if the context are defined in the 'contexts' sheet"
            exception_print(e,remedy)
            
        class_dict['common_context'].append(common_context)
        class_dict['first_context'].append(first_context)
        class_dict['second_context'].append(second_context)
        class_dict['change_type'].append(change_type)
    class_df = pd.DataFrame(class_dict)
    return class_df

def filter_list(pattern, search_space):
    search_space = [str(i) for i in search_space]
    r = re.compile(pattern)
    filtered_list = list(filter(r.match, search_space))
    return filtered_list

def remove_exclusions(statement_lib_df, exclusion_definition):
    for ind, (schema_num, target_column, pattern) in exclusion_definition[['schema num','target column','match']].iterrows():
        if schema_num != 'all':
            schema_selected = statement_lib_df[statement_lib_df['schema_num']==schema_num]
        else:
            schema_selected = statement_lib_df
        search_space = schema_selected[target_column].tolist()
        matched_rows = filter_list(pattern, search_space)
        print("removing {}...".format(matched_rows))
        for val in matched_rows:
            statement_lib_df = statement_lib_df[statement_lib_df[target_column].astype(str)!=val]
    return statement_lib_df

def create_measurement_detail_lookup(col, measurement_list, system_definition, default):
    if col in system_definition['measurements']:
        measurement_detail = {k:v for k,v in zip(measurement_list, system_definition['measurements'][col].to_list())}
    else:
        measurement_detail = {k:default for k in measurement_list}
    return measurement_detail

#%%
if __name__ == "__main__":
    if not os.path.exists('systems/{}/outputs'.format(system_name)):
        os.makedirs('systems/{}/outputs'.format(system_name))
    pick_entities_script_path = 'systems.{}.pick_entities'.format(system_name)
    pick_entities_location = pick_entities_script_path.replace('.','/')+'.py'

    if os.path.exists(pick_entities_location):
        print('detected pick_entities script') if verbose else None
        import importlib
        pick_entities = importlib.import_module(pick_entities_script_path)
        pick_entities_script_exists = True
    else:
        pick_entities_script_exists = False

    project_folder = './'
    system_definition_file = "systems/{}/system_definition_{}.xlsx".format(system_name,system_name)
    check_if_exist(system_definition_file)
    entities_file = "systems/{}/unique_entities.xlsx".format(system_name)

    if os.path.exists(entities_file):
        xls_file = pd.ExcelFile(entities_file)
        sheet_names = xls_file.sheet_names
        entity_df = pd.read_excel(entities_file,sheet_name = sheet_names,engine='openpyxl')
        entity_dict = {k:v['unique_items'].to_list() for k,v in entity_df.items()}
    sheet_names = ['schemas','measurements','contexts','exclusions']                    
    system_definition = pd.read_excel(system_definition_file, sheet_name=sheet_names,engine='openpyxl')
    system_definition['schemas'] = system_definition['schemas'].dropna(subset=['active'])
    system_definition['schemas'] = system_definition['schemas'][system_definition['schemas']['active']==1]
  
    for sheet in sheet_names:
        system_definition[sheet] = system_definition[sheet].dropna(axis=0, how='all')
    today = datetime.datetime.now()
    measurement_definition = system_definition['measurements']
    measurement_list = measurement_definition['measurement'].to_list() 
    measurement_group = create_measurement_detail_lookup('group_by', measurement_list, system_definition, 'date')
    measurement_aggregate = create_measurement_detail_lookup('aggregate', measurement_list, system_definition, 'sum')
    measurement_phrase = {k:v for k,v in zip(measurement_list, measurement_definition['phrase'].to_list())}
    measurement_columns = {k:v for k,v in zip(measurement_list, measurement_definition['column_in_data'].to_list())}
    measurement_benchmark = {k:v for k,v in zip(measurement_list, measurement_definition['benchmark'].to_list())}
    measurement_benchmark_text = {k:v for k,v in zip(measurement_list, measurement_definition['benchmark text'].to_list())}
    context_list = system_definition['contexts']['context'].to_list() 
    context_definition = pd.read_excel(system_definition_file, sheet_name=["c_{}".format(i) for i in context_list],engine='openpyxl',)
    sel_cols = measurement_list + ['date']  
    class_df = parse_schema_classes(system_definition,  ignore_placeholders = ['mean:1', 'mean:2', 'median:1', 'median:2', 'count:1', 'count:2', 'percentage'])


    """
    generating the insight statements and queries
    """

    statement_library_dict = {"schema_num":[], "insight_num":[], "intermediate":[], "first_context":[], 
                            "second_context":[], "common_context":[], "change_type":[],"inter_phrase":[], "inter_tense":[], "template":[], "insight_text":[], "queryA":[], "queryB":[], "dummyqueryA":[], "dummyqueryB":[], "scoring_type":[],"tag":[]}
    for ind,schema in tqdm(class_df.iterrows()):
        schema_num = schema['schema_num']
        scoring_type = schema['scoring_type']
        first_context = schema['first_context']
        second_context = schema['second_context']
        common_context = schema['common_context']
        change_type = schema['change_type']
        measurements = schema['measurements']
        applicable_items = schema['applicable_items']
        template = schema['schema_template']
        tag = schema['tag']
        combi_prop_dict = generate_intermediate_insight(first_context, second_context, common_context, change_type, measurements, context_definition, applicable_items)
        combi_names, combi_phrases, combi_tenses, combin_queries = combi_prop_dict.values()
        for insight_num,(inter_name, inter_phrase, inter_tense, inter_query) in enumerate(zip(combi_names, combi_phrases, combi_tenses, combin_queries)):
            # if schema_num != 5 or insight_num!=2:
            #     continue

            statement_library_dict["schema_num"].append(schema_num)
            statement_library_dict["scoring_type"].append(scoring_type)
            statement_library_dict["insight_num"].append(insight_num)
            statement_library_dict["intermediate"].append(inter_name)
            statement_library_dict["change_type"].append(change_type)
            statement_library_dict["first_context"].append(first_context)
            statement_library_dict["second_context"].append(second_context)
            statement_library_dict["common_context"].append(common_context)
            insight_text = get_sentence_raw(inter_phrase, template, measurement_phrase, measurement_benchmark_text)
            statement_library_dict["insight_text"].append(insight_text)
            queryA, queryB = get_query(inter_query, inter_groups=None)
            dummyqueryA, dummyqueryB = get_dummy_query(inter_query, inter_groups=None)
            statement_library_dict["queryA"].append(queryA)
            statement_library_dict["queryB"].append(queryB)
            statement_library_dict["dummyqueryA"].append(dummyqueryA)
            statement_library_dict["dummyqueryB"].append(dummyqueryB)
            statement_library_dict["template"].append(template)
            statement_library_dict["inter_phrase"].append(inter_phrase)
            statement_library_dict["inter_tense"].append(inter_tense)
            # statement_library_dict["inter_groups"].append(inter_groups)
            statement_library_dict["tag"].append(tag)
            
    statement_library_df = pd.DataFrame(statement_library_dict)
    print(len(statement_library_df),' statements in the library')
    exclusion_definition = system_definition['exclusions']
    statement_library_df = remove_exclusions(statement_library_df, exclusion_definition=exclusion_definition)
    statement_library_df.to_excel("systems/{}/statement_library_{}.xlsx".format(system_name,system_name), index=False)
    statement_library_df.to_pickle("systems/{}/statement_library_{}.pickle".format(system_name,system_name), protocol=4)
