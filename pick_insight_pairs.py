#%%
import pandas as pd
from generate_insight_lib import check_if_exist, create_measurement_detail_lookup, exception_print
import os, pdb
import datetime
import numpy as np
from config import system_name, scope_name

def get_difference_weight(diff, metric_parameters):
    x = diff
    a = 1/metric_parameters
    z = 1/(1 + np.exp(-a*x))
    return z

if __name__ == '__main__':
    project_folder = './'
    system_definition_file = "systems/{}/system_definition_{}.xlsx".format(system_name,system_name)
    if scope_name and scope_name != '': 
        insight_file = "systems/{}/scored_insights_final_with_clusters_{}_{}.pickle".format(system_name,scope_name,system_name)
    else:
        insight_file = "systems/{}/scored_insights_final_with_clusters_{}.pickle".format(system_name,system_name)
    all_insights_df = pd.read_pickle(insight_file)
    all_insights_df['mean_difference'] = all_insights_df['meanA'] - all_insights_df['meanB_']
    all_insights_df['pair_score'] = 0
    all_insights_df['pair_num'] = 0
    check_if_exist(system_definition_file)
    sheet_names = ['pairing','contexts']
    try:                    
        system_definition = pd.read_excel(system_definition_file, sheet_name=sheet_names,engine='openpyxl')
    except Exception as e:
        remedy =  'the scope does not have a tab to define the pairing criteria in the system_definition file'
        exception_print(e,remedy)

    system_definition['pairing'] = system_definition['pairing'].dropna()
    common_items = system_definition['pairing']['common'].tolist()
    uncommon_items = system_definition['pairing']['uncommon'].tolist()
    criteria = system_definition['pairing']['criteria'].tolist()
    contexts = system_definition['contexts']['context'].tolist()
    pair_df_rows = []
    for (main_ind,(com,ucom,cr)) in enumerate(zip(common_items, uncommon_items, criteria)):
        common_columns = []
        for i in eval(com):
            if i in contexts:
                common_columns.extend([i+'_A', i+'_B'])
            else:
                common_columns.extend([i])
        ucom = eval(ucom)[0]
        if ucom in contexts:
            ucom = ucom + '_A'
        unique_ucom = [i for i in all_insights_df[ucom].unique() if i!='']
        gp = all_insights_df.groupby(by=common_columns)
        counts_df = gp['percentage'].count().reset_index()
        counts_df = counts_df[counts_df['percentage'] == 2]
        
        for ind,row in counts_df.iterrows():
            filtered_rows = all_insights_df.copy()
            for col in common_columns:
                filtered_rows  = filtered_rows[filtered_rows[col] == row[col]]
            assert(len(filtered_rows[ucom].unique())==2)
            rowA = filtered_rows[filtered_rows[ucom] == unique_ucom[0]]
            rowB = filtered_rows[filtered_rows[ucom] == unique_ucom[1]]
            diff = abs(rowA['mean_difference'].iloc[0] - rowB['mean_difference'].iloc[0])
            score = get_difference_weight(diff,0.2)
            rowA.loc[rowA.index[0], 'pair_score'] = score
            rowB.loc[rowB.index[0], 'pair_score'] = score
            rowA.loc[rowA.index[0], 'pair_num'] = '{:02d}_{:02d}'.format(main_ind,ind)
            rowB.loc[rowB.index[0], 'pair_num'] = '{:02d}_{:02d}'.format(main_ind,ind)
            pair_df_rows.append(rowA.iloc[0])
            pair_df_rows.append(rowB.iloc[0])

    paired_df = []
    for row in pair_df_rows:
        if len(paired_df) == 0:
            paired_df = pd.DataFrame([row])
        else:
            paired_df = pd.concat([paired_df, pd.DataFrame([row])])
    
    paired_df.sort_values(by='pair_score',ascending=False)
    output_file = "systems/{}/paired_insights_{}".format(system_name,system_name)
    paired_df.to_excel(output_file+'.xlsx',engine='openpyxl',index=False)
    paired_df.to_pickle(output_file+'.pickle',protocol=4)
 