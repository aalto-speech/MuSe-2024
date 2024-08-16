import os, sys
import pandas as pd
import numpy as np

from eval import mean_pearsons, calc_pearsons
from utils import write_to_csv
from config_model_feat import unimodals, multimodals, dct_fusion_models

# devel label 
fname='results/prediction_muse/perception/lf/label_devel.csv'
df_label=pd.read_csv(fname, index_col=0)

# devel uni- and multi-modal systems
fname='./results/csvs/table2_pred_perception.csv' # script/summarize_pred_mpc.py
df=pd.read_csv(fname)

top_models=unimodals+multimodals

# label-wise system performance from top_models
appended_result=[]
for _model_id in top_models:
    _df=df[(df.model_id==_model_id)]
    #print(_df.shape)
    _model='_'.join(_model_id.split('_')[:-2])
    _df=_df.rename(columns={'mean_pearsons': _model})
    _df=_df.drop(columns=['log_name', 'model_id'],  axis=1)
    _df=_df.reset_index(drop=True)
    appended_result.append(_df)
df_mpc=pd.concat(appended_result, axis=1)
df_mpc = df_mpc.loc[:,~df_mpc.columns.duplicated()]
print(df_mpc.shape, df_mpc.head(3))
df_mpc_unimodals=df_mpc.copy()

# fusion models
appended_result=[]
for _model, _path in dct_fusion_models.items():
    # overall performance
    df_pred=pd.read_csv(_path, index_col=0)
    mpc = mean_pearsons(df_pred.values, df_label.values)
    dct_overall = {'model_id':_model,
                    'mean_pearsons':mpc}
    appended_result_model=[]
    for _label in df_label.columns:
        pred_array=df_pred[_label].values
        label_array=df_label[_label].values
        mpc_per_label=calc_pearsons(pred_array, label_array)
        # label-wise performance
        dct = {'label_dim': _label,
                _model: mpc_per_label}
        appended_result_model.append(pd.DataFrame([dct]))  
    _df=pd.concat(appended_result_model)
    _df=_df.reset_index(drop=True)
    appended_result.append(_df) 
appended_result.append(df_mpc_unimodals)
df_mpc=pd.concat(appended_result, axis=1)
df_mpc=df_mpc.reset_index(drop=True)
df_mpc = df_mpc.loc[:,~df_mpc.columns.duplicated()]
df_mpc=df_mpc.set_index('label_dim')
#print(df_mpc.shape, df_mpc, df_mpc.mean())

# write label-wise performance to output
csv_path='./results/csvs/table2_pred_perception_lf.csv'
write_to_csv(df_mpc, csv_path)

# write overall performance to output
_df=df_mpc.mean().to_frame('Mean Pearsons')
_df=_df.sort_values(by=['Mean Pearsons'])
csv_path='./results/csvs/table3_pred_perception_lf.csv'
write_to_csv(_df, csv_path)

# latex table
pd.options.display.float_format = "{:,.4f}".format
_models=['RNN_[vit-fer]', 'CNN_[vit-fer]', 'CRNN_[vit-fer]', 'CNN-ATTN_[vit-fer]', 'CRNN-ATTN_[vit-fer]', 'IAF_[w2v-msp_vit-fer_gpt2]', 'late_fusion_25', 'AED', 'late_fusion',]
_df=df_mpc[_models]
_traits=['aggressive', 'arrogant', 'assertiv', 'confident', 'dominant', 'independent', 'risk', 'leader_like', 
         'collaborative', 'enthusiastic', 'friendly', 'good_natured', 'kind', 'likeable', 'sincere', 'warm']
_df=_df.reindex(_traits)
_df.loc['mean'] = _df.mean()
print(_df.shape, _df.head(3), len(_traits))

# write label-wise performance to output
csv_path='./results/csvs/table2_pred_perception_for_paper.csv'
write_to_csv(_df, csv_path)