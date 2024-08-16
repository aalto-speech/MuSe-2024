# prepare predictions for perception: 
# # collect from label dir and 
# # summarize into submission format

import os, sys
import pandas as pd
import numpy as np

from utils import write_to_csv
from config import PREDICTION_FOLDER
from config_model_feat import unimodals, multimodals, label_dims

# submission format with order
label_dims_sorted=['aggressive', 'arrogant', 'dominant', 'enthusiastic', 'friendly', 
 'leader_like', 'likeable', 'assertiv', 'confident', 'independent', 
 'risk', 'sincere', 'collaborative', 'kind', 'warm', 'good_natured']

# devel uni- and multi-modal systems
fname='./results/csvs/table2_pred_perception.csv' # script/summarize_pred_mpc.py
df=pd.read_csv(fname)

top_models=unimodals+multimodals

# label-wise system performance from top_models
appended_result=[]
for _model_id in top_models:
    try:
        for partition in ['devel', 'test']:
            appended_preds=[]
            for _label in label_dims: 
                _df=df[(df.model_id==_model_id) & (df.label_dim==_label)]
                _log_name=_df.log_name.values[0]
                #print(_df.head(3), _df.shape, _log_name)
                #sys.exit(0)
                pred_fname=os.path.join(PREDICTION_FOLDER, 'perception', _label, _log_name, f'predictions_{partition}.csv')
                df_pred=pd.read_csv(pred_fname, index_col=0)
                df_pred=df_pred.drop(columns=['label'],  axis=1)
                #print(df_pred.shape, df_pred.head(2))
                #sys.exit(0)
                df_pred=df_pred.rename(columns={'prediction': _label})
                appended_preds.append(df_pred)
            df_out=pd.concat(appended_preds, axis=1)
            df_out.index.names=['subj_id']
            #print(df_out.shape, df_out.head(3))

            # write output
            csv_path=os.path.join(PREDICTION_FOLDER, 'submission', 'perception', _model_id, f'predictions_{partition}.csv')
            write_to_csv(df_out, csv_path)
    except:
        print(f'Problem with {_model_id}')