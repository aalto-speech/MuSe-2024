import os, sys
import pandas as pd
import glob
import pathlib
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

sys.path.append("/scratch/work/huangg5/muse/MuSe-2024")
from eval import calc_auc

dir_results='/scratch/work/huangg5/muse/MuSe-2024/results/prediction_muse/'
task='humor'
metric ='auc'

# devel meta ids
from data_parser import get_data_partition
meta_fname='/scratch/elec/puhe/c/muse_2024/c2_muse_humor/metadata/partition.csv'
_, partition_to_subject = get_data_partition(meta_fname)
num_devel_spkrs=len(partition_to_subject['devel'])
#print(num_devel_spkrs)

# walk in dir and collect latest pred.
subset='*_*_*' #'RNN*' #'RNN*egemaps*0.0005*' # #  RNN_[vit-fer]_[64_2_False_64]_[0.0005_256] 0.8915 
prediction_dir=os.path.join(dir_results, task)
appended_data = []

onlyfiles=glob.glob(f'{prediction_dir}/{subset}/**/*devel.csv',  recursive = True)
onlyfiles=[f for f in onlyfiles if os.path.isfile(f) ]
print(len(onlyfiles))
#sys.exit(0)

for _file in onlyfiles:    
    try:
        _logname=pathlib.Path(_file).parts[9] # humor has a seed folder for RNN models, not for IAF
        _model_type=_logname.split('_')[0]
        _date_id=_logname.split('_')[1]
        _feat_types=_logname.split('_')[2:-2] # multimodal uses multiple features
        _hyper_params=_logname.split('_')[-2:]
        # read predictions
        model_id='_'.join([_model_type] + _feat_types + _hyper_params) # create new model id
        if _date_id.endswith('combine-train-dev'): # predictions for submission
            model_id += '_combine-train-dev' # create new model id
        df=pd.read_csv(_file)
        df['log_name']=_logname
        df['model_id']=model_id
        df['date_id']=_date_id
        appended_data.append(df)
    except:
        print(f'Problem with {_file}.')
# deal with duplicates, keep latest
df = pd.concat(appended_data)
print(df.shape, df.head(3))
df = df.drop_duplicates(subset=['meta_col_0','meta_col_1','meta_col_2','meta_col_3', 'model_id'], keep='last')

# sanity check
uniq_models=df['model_id'].unique()

### Table 3. overall p
appended_result=[]
missing_ids=[]
for _model in uniq_models:
    pred_arrays=[]
    label_arrays=[]
    auc_arrays=[]
    _df=df[(df.model_id==_model)]
    # sanity check to have 16 labels.
    if 0:
        if _model not in missing_ids:
            missing_ids.append(_model)
    else:
        #print(_model, _label, _df.shape)
        log_name=_df.log_name.unique()[0]
        assert len(_df.log_name.unique())==1
        pred_array=_df.prediction
        label_array=_df.label
        _auc=calc_auc(pred_array.to_numpy(), label_array.to_numpy())
        
        # model-wise performance
        dct = {'log_name':log_name,
                'model_id': _model,
                'metric': _auc}
        appended_result.append(pd.DataFrame([dct]))

df_mpc=pd.concat(appended_result)  
df_mpc=df_mpc.sort_values(['metric'], ascending=False)
print(df_mpc.shape, df_mpc)

if 1 & len(missing_ids)>0:
    print(f'missing labels for {missing_ids}')
    
# write output
if 1:
    csv_path='results/csvs/table3_pred_humor.csv'
    csv_dir = pathlib.Path(csv_path).parent.resolve()
    os.makedirs(csv_dir, exist_ok=True)

    df_mpc.to_csv(csv_path, index=False)

sys.exit(0)

### evaluation metrics
f_name='/scratch/work/huangg5/muse/MuSe-2024/results/prediction_muse/humor/lf/predictions_devel.csv'
df=pd.read_csv(f_name, index_col=0)
#print(df.shape, df.head(3))
preds=df['prediction'].to_numpy()
labels=df['label'].to_numpy()
print(preds.shape, labels.shape)
auc = calc_auc(preds, labels)

testy=labels
yhat_probs=preds
yhat_classes=preds>0.5

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testy, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(testy, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(testy, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(testy, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(testy, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(testy, yhat_classes)
print('Confusion Matrix: \n', matrix)