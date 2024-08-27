#prepare predictions for submission on Humor

import os, sys
import pandas as pd
import numpy as np

from utils import write_to_csv
from config import PREDICTION_FOLDER
from config_model_feat import unimodals, multimodals

# devel uni- and multi-modal systems
fname='./results/csvs/table3_pred_humor.csv' # script/summarize_pred_mpc.py
df=pd.read_csv(fname)

top_models=[i for i in unimodals+multimodals if i.startswith('IAF')]

# system performance from top_models
appended_result=[]
for _model_id in top_models:
    try:
    #if 1:
        for partition in ['devel', 'test']:
            _df=df[(df.model_id==_model_id)]
            _log_name=_df.log_name.values[0]
            #print(_df.head(3), _df.shape)
            #sys.exit(0)
            # drop 'label' columns
        
            pred_fname=os.path.join(PREDICTION_FOLDER, 'humor', _log_name, f'predictions_{partition}.csv')
            df_pred=pd.read_csv(pred_fname, index_col=0)
            df_pred=df_pred.drop(columns=['label'],  axis=1)
            #print(df_pred.shape, df_pred.head(2))
            #sys.exit(0)
            
            # write output
            csv_path=os.path.join(PREDICTION_FOLDER, 'submission', 'humor', _model_id, f'predictions_{partition}.csv')
            write_to_csv(df_pred, csv_path)
    except:
        print(f'Problem with {_model_id}')
