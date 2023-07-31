import pandas as pd
from pattern.en import conjugate
from config import system_name, scope_name, unknown_phrase, tense_dict, reserved_keywords
import regex as re
import os, sys, pdb
from utils.commonlib import unique, flatten, streval, first_caps


        



def transform_tense(x, tense, relative_person):
    tense_tr = tense_dict[tense]
    number_format = 'singular'
    try:
        transformed = conjugate(x, tense = tense_tr, person = int(relative_person), number = number_format, mood = "indicative", aspect = "imperfective", negated = False)
    except RuntimeError:
        transformed = conjugate(x, tense = tense_tr, person = int(relative_person), number = number_format, mood = "indicative", aspect = "imperfective", negated = False)
    return transformed

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
                    if '_' in verb:
                        verb, relative_person = verb.split('_')
                    elif ',' in verb:
                        verb, relative_person = verb.split(',')
                    else:
                        relative_person = '3'
                    transformed_verb = transform_tense(verb,tense,relative_person)
                    inp_str = inp_str.replace(tense_transform_candidate, transformed_verb)
    return inp_str

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

def get_relative_period(df):
    list_A = []
    list_B = []
    inter = df['intermediate']
    for i_ in inter:
        i = streval(i_)
        list_A_element = ''
        list_B_element = ''
        if "period" in i:
            if i['period'][0].startswith('relative'):
                list_A_element = i['period'][0].split('_')[-1]
            if i['period'][1].startswith('relative'):
                list_B_element = i['period'][1].split('_')[-1]
        list_A.append(list_A_element)
        list_B.append(list_B_element)
    return list_A, list_B


def get_measurement(df):
    inter = df['intermediate']
    return [streval(i)['measurement'] for i in inter]
    


def get_sentence(inter_phrase, inter_tense, template, measurement_phrase, measurement_unit, rw):
    if isinstance(rw['intermediate'],str):
        meanA, meanB, medianA, percentage, meas = rw['meanA'], rw['meanB'], rw['medianA'], rw['percentage'], eval(rw['intermediate'])['measurement']
    else:
        meanA, meanB, medianA, percentage, meas = rw['meanA'], rw['meanB'], rw['medianA'], rw['percentage'], rw['intermediate']['measurement']
    unit = measurement_unit[meas]
    countA,countB = rw['countA'], rw['countB']
    insight_text = template.replace("!","")
    tense_list = []
    insight_text = insight_text.replace('{{mean:1}}', '{:0.2f} {}'.format(meanA, unit))
    insight_text = insight_text.replace('{{median:1}}', '{:0.2f} {}'.format(medianA, unit))
    insight_text = insight_text.replace('{{count:1}}', '{:0.2f}'.format(countA))
    insight_text = insight_text.replace('{{unit}}', unit)
    insight_text = insight_text.replace('{{none:2}}', '')
    if '{{mean:2}}' in insight_text:
        insight_text = insight_text.replace('{{mean:2}}', '{:0.2f} {}'.format(meanB, unit))
    if '{{count:2}}' in insight_text:
        insight_text = insight_text.replace('{{count:2}}', '{:0.2f}'.format(countB))
    insight_text = insight_text.replace('{{percentage}}', '{:0.2f}%'.format(percentage))
    for ((k,v),(tk,tv)) in zip(inter_phrase.items(), inter_tense.items()):
        if len(v) == 2: # Takes care of the comparative context
            find1 = "{{" + "{}:1".format(k) + "}}"
            replace1 = tensify(v[0],tv[0]) 
            tense_list.append(tv[0])
            if v[1] == """{{measurement_benchmark}}""":
                find2 = "{{" + "measurement_benchmark:2" + "}}"
                replace2 = str(meanB)
            else:
                find2 = "{{" + "{}:2".format(k) + "}}"
                replace2 = tensify(v[1],tv[1]) 
            insight_text = insight_text.replace(find1,replace1).replace(find2,replace2)
        else: # Takes care of the common context
            find = "{{" + "{}".format(k) + "}}"
            if k not in reserved_keywords:
                replace = tensify(v,tv)
            elif k == "measurement":
                replace = measurement_phrase[v]
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
    return insight_text, overall_tense

def get_contexts(irow, tn):
    meas = irow.intermediate['measurement']
    unit = measurement_unit[meas]
    meas_text = tensify(measurement_phrase[meas], tn)
    meas_text = f"{meas_text}\n({unit})"
    contA = irow.first_context[0]
    contB = irow.second_context[0]
    contB = contB[1:] if contB.startswith('!') else contB
    keyA_,contA_ = irow.intermediate[contA][0].split('$')

    if contB!="measurement_benchmark":
        keyB_,contB_ = irow.intermediate[contB][0].split('$')
    
    cont_df = context_definition[f'c_{contA}']
    entities = [contA_]
    textA = eval(cont_df[cont_df.name == keyA_]['prepositional phrase'].values[0])[0]
    
    if irow.change_type == "exclusion":
        nottextA = eval(cont_df[cont_df.name == keyA_]['inversion phrase'].values[0])[0]
    subset_count = irow.countA

    if irow.test in ['benchmark','stat']:
        textB = ''
    else:
        subset_count += irow.countB
        contB = irow.second_context[0]
        if irow.change_type == "exclusion":
            textB = f"{nottextA}\n(on an average)" 
        else:
            keyB_,contB_ = irow.intermediate[contA][-1].split('$')
            entities = [contB_]
            textB = eval(cont_df[cont_df.name == keyB_]['prepositional phrase'].values[0])[0]
    common = irow.common_context
    common_text = []
    if len(common)!=0:
        common_ = [[i,irow.intermediate[i].split('$')] for i in common]
        for (contC,(keyC_,contC_)) in common_:
            entities = [contC_]
            common_t = eval(context_definition[f'c_{contC}'][context_definition[f'c_{contC}'].name == keyC_]['prepositional phrase'].values[0])[0] 
            common_t = tensify(common_t,tn)
            common_text.append(common_t)
    common_text = ', '.join(common_text)
    common_text = f"{common_text}\n({subset_count/irow['total_count']*100:0.02f}% of all data)"
    return textA, textB, meas_text, common_text



if __name__ == "__main__":
    system_definition_file = "systems/{}/system_definition_{}.xlsx".format(system_name,system_name)
    sheet_names = ['schemas','measurements','contexts','exclusions']                    
    system_definition = pd.read_excel(system_definition_file, sheet_name=sheet_names,engine='openpyxl')
    for sheet in sheet_names:
        system_definition[sheet] = system_definition[sheet].dropna(axis=0, how='all')
    context_list = system_definition['contexts']['context'].to_list() 
    context_definition = pd.read_excel(system_definition_file, sheet_name=["c_{}".format(i) for i in context_list],engine='openpyxl',)
    measurement_definition = system_definition['measurements']
    measurement_list = measurement_definition['measurement'].to_list()
    measurement_phrase = {k:v for k,v in zip(measurement_list, measurement_definition['phrase'].to_list())}
    measurement_unit = {k:v for k,v in zip(measurement_list, measurement_definition['unit'].to_list())}
    measurement_benchmark = {k:v for k,v in zip(measurement_list, measurement_definition['benchmark'].to_list())}
    measurement_benchmark_text = {k:v for k,v in zip(measurement_list, measurement_definition['benchmark text'].to_list())}
    if scope_name and scope_name != '':
        scored_file = "systems/{}/outputs/scored_insights_{}_{}.pickle".format(system_name,scope_name,system_name)
    else:
        scored_file = "systems/{}/outputs/scored_insights_{}.pickle".format(system_name,system_name)
    if os.path.exists(scored_file):
        scored_library = pd.read_pickle(scored_file)
        scored_library['contextA'] = ''
        scored_library['contextB'] = ''
    else:
        print('file {} generated by score_insight_lib.py needs to be present. Please Check if scoring ran successfully.'.format(scored_file))
        print('aborting how_to_Say.py')
        sys.exit()
    insight_text_final = []
    textAl = []
    textBl = []
    meas_textl = []
    common_textl = []
    list_of_filters = unique(flatten([list(streval(i).keys()) for i in scored_library['intermediate']]))
    list_of_filters_AB = flatten([[i+'_A', i+'_B'] if i!='measurement' else [i] for i in list_of_filters])
    filter_dict = {i:[] for i in list_of_filters_AB}
    filter_dict.update({'name':[]})
    for ind, row in scored_library.iterrows():
        intermediate = streval(row['intermediate'])
        filter_dict['name'].append(row.name)
        for filt in list_of_filters:
            if filt in intermediate.keys():
                if filt == 'measurement':
                    filter_dict[filt].append(intermediate[filt])
                elif type(intermediate[filt]) == tuple:
                    filter_dict[filt+'_A'].append(intermediate[filt][0])
                    filter_dict[filt+'_B'].append(intermediate[filt][1])
                else:
                    filter_dict[filt+'_A'].append(intermediate[filt])
                    filter_dict[filt+'_B'].append('')
            else:
                filter_dict[filt+'_A'].append('')
                filter_dict[filt+'_B'].append('')
        inter_phrase = streval(row['inter_phrase'])
        inter_tense = streval(row['inter_tense'])
        template = row['template']
        sent1, otense = get_sentence(inter_phrase, inter_tense, template, measurement_phrase, measurement_unit, row)
        textA, textB, meas_text, common_text = get_contexts(row, otense)
        textAl.append(textA)
        textBl.append(textB)
        meas_textl.append(meas_text)
        common_textl.append(common_text)
        sent2 = sent1.replace("""{{comparison}}""",row['comparison'])
        if row['comparison_post'] != row['comparison_pre']:
            sent2 = sent2.replace("""{{comparison_pre}}""",row['comparison_pre']).replace("""{{comparison_post}}""",row['comparison_post'])
        if unknown_phrase:
            sent2 = sent2.replace("""<<unknown>>""",unknown_phrase)
        if ind == 0:
            print(row['insight_text'])
            print(sent1)
            print(sent2)
        insight_text_final.append(sent2)
        if '{{measurement}}' in sent2:
            pass
    filter_df = pd.DataFrame(filter_dict)
    filter_df = filter_df.set_index('name')
    for col in filter_df.columns:
        scored_library[col] = filter_df[col]
    scored_library['insight_text_final'] = insight_text_final
    scored_library['textA'] = textAl
    scored_library['textB'] = textBl
    scored_library['meas_text'] = meas_textl
    scored_library['common_text'] = common_textl
    scored_library['relative_period_A'], scored_library['relative_period_B']  = get_relative_period(scored_library)
    # scored_library['measurement']  = get_measurement(scored_library)
    scored_library.sort_values(by='pscore_final', ascending=False, inplace=True)
    if scope_name and scope_name != '': 
        scored_library.to_excel("systems/{}/outputs/scored_insights_final_{}_{}.xlsx".format(system_name,scope_name,system_name), index=False)
        scored_library.to_pickle("systems/{}/outputs/scored_insights_final_{}_{}.pickle".format(system_name,scope_name,system_name), protocol=4)
    else:
        scored_library.to_excel("systems/{}/outputs/scored_insights_final_{}.xlsx".format(system_name,system_name), index=False)
        scored_library.to_pickle("systems/{}/outputs/scored_insights_final_{}.pickle".format(system_name,system_name), protocol=4)
