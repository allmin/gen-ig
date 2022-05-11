import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
# from pattern.en import conjugate
from config import system_name, scope_name, unknown_phrase, tense_dict, reserved_keywords, recommendation_count
from utils.commonlib import unique, flatten, streval
from sklearn.cluster import KMeans
# import regex as re
# import os, sys
# scope_name = 'user02'

def perform_kmeans(K,X):
    if K == 0:
        return []
    kmeans = KMeans(n_clusters=K, random_state=123).fit(X)
    return kmeans.labels_

def save_to_system_folder(df,prefix):
    if scope_name and scope_name != '':
        df.to_excel("systems/{}/{}_{}_{}.xlsx".format(system_name,prefix,scope_name,system_name), index=False)
        df.to_pickle("systems/{}/{}_{}_{}.pickle".format(system_name,prefix,scope_name,system_name), protocol=4)
    else:
        df.to_excel("systems/{}/{}_{}.xlsx".format(system_name,prefix,system_name), index=False)
        df.to_pickle("systems/{}/{}_{}.pickle".format(system_name,prefix,system_name), protocol=4)


if __name__ == '__main__':
    sorting_criteria = 'pscore_final'
    if scope_name and scope_name != '':
        scored_library = pd.read_pickle("systems/{}/scored_insights_final_{}_{}.pickle".format(system_name,scope_name,system_name))
    else:
        scored_library = pd.read_pickle("systems/{}/scored_insights_final_{}.pickle".format(system_name,system_name))

    
    scored_library['complexity'] = [len(s.split(' ')) for s in scored_library['insight_text_final']]
    list_of_filters = unique(flatten([list(streval(i).keys()) for i in scored_library['intermediate']]))
    list_of_filters_AB = flatten([[i+'_A', i+'_B'] if i!='measurement' else [i] for i in list_of_filters])


    insight_identifier = 'schema'+scored_library['schema_num'].astype(str)+'|'+'complexity'+scored_library['complexity'].astype(str)+'|'
    for col in list_of_filters_AB:
        if len(insight_identifier) != 0:
            insight_identifier = insight_identifier + '|' + scored_library[col]

    scored_library['insight_identifier'] = insight_identifier

    vocab_path = 'systems/{}/vocab.xlsx'.format(system_name)
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
                for k in j.split('_'):
                    if (k not in vocabulary) and (k!=''):
                        numv += 1
                        vocabulary[k] = numv
        vocab = pd.DataFrame({'keys':list(vocabulary.keys()),'indices':list(vocabulary.values())})
        vocab.to_excel(vocab_path,engine='openpyxl',index=False)

    else:
        for i in insight_identifier.values:
            for j in i.split('|'):
                for k in j.split('_'):
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
            for k in j.split('_'):
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

    try:
        if system_name == 'iphil' and os.getcwd().split(os.path.sep)[-1] == 'get_insights':
            prefix = 'scored_insights_final_with_clusters'
            scored_library.to_pickle("../../actions/data/{}_{}_{}.pickle".format(prefix,scope_name,system_name), protocol=4)
    except Exception as e:
        print(e)

