import os, sys
import pandas as pd
import pathlib

from utils import write_to_csv
from config import PREDICTION_FOLDER
from config_model_feat import unimodals, multimodals, label_dims

fname='./results/csvs/table2_pred_perception.csv' # script/summarize_pred_mpc.py
df=pd.read_csv(fname)

prepare_sbatch_script=0
_impression='full'# '5-sec' # 'None' # 
if _impression:
    if _impression.endswith('-sec'):
        top_models=[i for i in unimodals+multimodals if _impression in i]
        lf_dir='lf-impressions-' + _impression #'lf-impressions' #  default 'lf'
    elif _impression=='full':
        top_models=[i for i in unimodals+multimodals if not '-sec' in i]
        lf_dir='lf'
else:
    top_models=unimodals+multimodals
    lf_dir='lf+impressions'

if prepare_sbatch_script:
    cmds=[]
    for _label in label_dims[:]:
        _df=df[df.label_dim==_label]
        _df=_df[_df['model_id'].isin(top_models)]
        #print(_df.shape, ' '.join(_df.log_name.values))
        _ids= ' '.join(_df.log_name.values)
        _cmd=f"python3 late_fusion.py --task perception --submission_format --lf_dir {lf_dir} --result_csv $csv --label_dim {_label} --model_ids {_ids}"
        cmds.append(_cmd)

    # output
    # python3 late_fusion.py --task perception --submission_format --label_dim aggressive --model_ids top_1 top_2 top_3 top_4 top_5
    header_txt="""#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --partition=gpu-h100-80g 
#SBATCH --gpus=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load mamba

source activate muse

echo "late fusion."

## perception
csv='perception_late_fusion.csv'

"""
    with open('./run-late-fusion-perception.sh', 'w') as f:
        f.write(f"{header_txt}")
        for line in cmds:
            f.write(f"{line}\n")

    sys.exit(0)

# sbatch run-late-fusion-perception.sh

# combine lf labels into 1 csv submission file
for partition in ['devel', 'test']:
    appended_preds=[]
    for _label in label_dims[:]:
        pred_fname=os.path.join(PREDICTION_FOLDER, 'perception', _label, lf_dir, f'predictions_{partition}.csv')
        df_pred=pd.read_csv(pred_fname, index_col=0)
        #print(df_pred.shape, df_pred.head(2))
        df_pred=df_pred.rename(columns={'prediction': _label})
        appended_preds.append(df_pred)
    df_out=pd.concat(appended_preds, axis=1)
    df_out.index.names=['subj_id']
    #print(df_out.shape, df_out.head(3))

    # write output
    csv_path=os.path.join(PREDICTION_FOLDER, 'perception', lf_dir, f'predictions_{partition}.csv')
    write_to_csv(df_out, csv_path)
    
# cp results/prediction_muse/perception/lf/*.csv results/submission/c1_perception
