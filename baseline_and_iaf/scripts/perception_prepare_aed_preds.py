import os, sys
import pandas as pd
import numpy as np
import pathlib

from config import PREDICTION_FOLDER
from eval import calc_pearsons
from utils import write_to_csv

fname='results/prediction_muse/perception/lf/label_devel.csv'
df_label=pd.read_csv(fname, index_col=0)
label_dims=df_label.columns
#print(df_label.shape, df_label.head(3))
#sys.exit(0)

# prepare label wise predictions
"""
Input format:
subj_id,aggressive,arrogant,dominant,enthusiastic,friendly,leader_like,likeable,assertiv,confident,independent,risk,sincere,collaborative,kind,warm,good_natured
0,0.59383935,0.59933823,0.599389,0.59933823,0.5993886,0.599389,0.599389,0.5993886,0.599389,0.599389,0.599389,0.599389,0.59383935,0.599389,0.599389,0.599389
"""

"""
Output format:
devel partition
meta_col_0,prediction,label
9,0.39005035161972046,0.3528999984264374
10,0.42247816920280457,0.4943000078201294

test partition
meta_col_0,prediction,label
0,0.3390737771987915,
"""

# Late fusion
if 0:
    for _label in label_dims:
        for partition in ['test', 'devel']: #
            in_fname=f'results/prediction_muse/perception/aed_dejan/predictions_{partition}.csv'
            df_pred=pd.read_csv(in_fname, index_col=0)
            #print(df_pred.shape, df_pred.head(3))
            _df=df_pred[[_label]]
            _df=_df.rename(columns={_label:'prediction'})
            #print(_df.shape, _df.head(3))
            #sys.exit(0)
            
            if partition=='devel':
                _df_label=df_label[[ _label]]
                _df_label=_df_label.rename(columns={_label:'label'})
                _df=pd.concat([_df, _df_label], axis=1)
            else:
                _df['label']=0

            _df.index.names=['meta_col_0']
            #print(_df.shape, _df.head(3))
            #sys.exit(0)

            out_fname=f'results/prediction_muse/perception/{_label}/aed_dejan/predictions_{partition}.csv'

            # write output
            write_to_csv(_df, out_fname)

    # sbatch run-late-fusion-perception-gp-dejan.sh

# pandora fusion
criteria=sys.argv[1]#'pseudo_fusion'#'top_1'#'positive_pearsons'# 
fname='./results/csvs/table2_pred_perception.csv' # script/summarize_pred_mpc.py
df=pd.read_csv(fname)
df=df.sort_values(['label_dim', 'mean_pearsons'], ascending=False)
print(df.shape, df.columns, df.head(3))

### add aed model to Table 2. label-wise p
in_fname=f'results/prediction_muse/perception/aed_dejan/predictions_devel.csv'
df_pred=pd.read_csv(in_fname, index_col=0)

appended_result=[]
for _label in label_dims:
    pred_array=df_pred[_label].values
    label_array=df_label[_label].values
    mpc_per_label=calc_pearsons(pred_array, label_array)
    # label-wise performance
    dct = {'log_name':'aed_dejan',
            'model_id':'aed_dejan',
            'label_dim': _label,
            'mean_pearsons': mpc_per_label}
    appended_result.append(pd.DataFrame([dct]))
    
df_aed=pd.concat(appended_result)
df_mpc=pd.concat([df_aed, df])
df_mpc=df_mpc.sort_values(['label_dim', 'mean_pearsons'], ascending=False)
df_mpc=df_mpc.reset_index(drop=True)
print(df_mpc.shape)

csv_path='./results/csvs/table2_pred_perception_with_aed.csv'
write_to_csv(df_mpc, csv_path)

# perception_pandora_fusion.py pseudo_fusion-gp-dejan


