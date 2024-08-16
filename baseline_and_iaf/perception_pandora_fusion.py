import os, sys
import pandas as pd
import pathlib

from config import PREDICTION_FOLDER
from config_model_feat import unimodals, multimodals, label_dims
from utils import write_to_csv

fname='./results/csvs/table2_pred_perception.csv' # script/summarize_pred_mpc.py
# fname='./results/csvs/table2_pred_perception_with_aed.csv' # 
df=pd.read_csv(fname)
df=df.sort_values(['label_dim', 'mean_pearsons'], ascending=False)
# print(df.shape, df.columns, df.head(3))

_impression='full'#'5-sec' # 'None' #  
if _impression:
    if _impression.endswith('-sec'):
        top_models=[i for i in unimodals+multimodals if _impression in i]
        criteria='pseudo_fusion-impressions-' + _impression #'lf-impressions' #  default 'lf'
    elif _impression=='full':
        top_models=[i for i in unimodals+multimodals if not '-sec' in i]
        criteria='pseudo_fusion'
else:
    top_models=unimodals+multimodals
    criteria='pseudo_fusion+impressions'

for partition in ['devel', 'test']:
    appended_preds=[]
    for _label in label_dims[:]:
        _df=df[df.label_dim==_label]
        if criteria=='positive_pearsons':
            _df=_df[_df.mean_pearsons>0]
        elif criteria.startswith('top'):
            k=int(criteria.split('_')[-1])
            #_df=_df.loc[_df.mean_pearsons.idxmax()]
            _df=_df.iloc[k-1] # Series result # df.iloc[[1]]  # DataFrame result
        else:
            # 'pseudo_fusion':
            # avt features or the fusion of them. include v-v-v
            _df=_df[_df['model_id'].isin(top_models)]
            _df=_df.iloc[0]
            
        log_name=_df.log_name
        pred_fname=os.path.join(PREDICTION_FOLDER, 'perception', _label, log_name, f'predictions_{partition}.csv')
        df_pred=pd.read_csv(pred_fname, index_col=0)
        df_pred=df_pred.rename(columns={'prediction': _label})
        #print(df_pred.shape, df_pred.head(3))
        # save to label dir, to do late fusion, 'label' is kept for late_fusion.py
        csv_path=os.path.join(PREDICTION_FOLDER, 'perception', _label, criteria, f'predictions_{partition}.csv')
        csv_dir = pathlib.Path(csv_path).parent.resolve()
        os.makedirs(csv_dir, exist_ok=True)
        df_pred.to_csv(csv_path)
        # append to df and save to perception dir
        #if 0 and partition =='devel': # keep label for sanity check
        #    df_pred=df_pred.rename(columns={'label': _label+'_label'})
        #else:
        df_pred=df_pred.drop(columns=['label'])
        appended_preds.append(df_pred)
    df_out=pd.concat(appended_preds, axis=1)
    df_out.index.names=['subject_id']
    #print(df_out.shape, df_out.head(3))
    #sys.exit(0)

    # write output
    csv_path=os.path.join(PREDICTION_FOLDER, 'perception', criteria, f'predictions_{partition}.csv')
    write_to_csv(df_out, csv_path)
    
# cp results/prediction_muse/perception/lf/*.csv results/submission/c1_perception