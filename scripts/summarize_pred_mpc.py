import os, sys
import pandas as pd
import numpy as np
import glob
import pathlib
from tqdm import tqdm

sys.path.append("/scratch/work/huangg5/muse/MuSe-2024")
from eval import calc_pearsons, mean_pearsons
from utils import write_to_csv
from config_model_feat import label_dims, unimodals, multimodals

dir_results='/scratch/work/huangg5/muse/MuSe-2024/results/prediction_muse/'
task='perception'
metric = 'best_val_Pearson'
top_models=unimodals+multimodals
print(len(top_models)) #25

# devel meta ids
from data_parser import get_data_partition
meta_fname='/scratch/elec/puhe/c/muse_2024/c1_muse_perception/metadata/partition.csv'
_, partition_to_subject = get_data_partition(meta_fname)
num_devel_spkrs=len(partition_to_subject['devel'])

# walk in dir and collect latest pred.
subset='*_*_*' #'RNN*ds*0.0005*' #'*' #  mpc: 0.143677 # except: fusion folder 'lf' 
prediction_dir=os.path.join(dir_results, task)
appended_data = []
for _label in label_dims:
    onlyfiles=glob.glob(f'{prediction_dir}/{_label}/{subset}/*devel.csv',  recursive = True)
    onlyfiles=[f for f in onlyfiles if os.path.isfile(f) ]
    for _file in onlyfiles:
        try:
            _logname=pathlib.Path(_file).parts[-2]
            _model_type=_logname.split('_')[0]
            _date_id=_logname.split('_')[1]
            _feat_types=_logname.split('_')[2:-2] # multimodal uses multiple features
            _hyper_params=_logname.split('_')[-2:]
            # read predictions
            model_id='_'.join([_model_type] + _feat_types + _hyper_params) # create new model id
            if _date_id.endswith('combine-train-dev'): # predictions for submission
                model_id += '_combine-train-dev' # create new model id
            df=pd.read_csv(_file)
            df['label_dim']=_label
            df['model_id']=model_id
            #df['feat_id']=_feat_types
            df['log_name']=_logname
            appended_data.append(df)
        except:
            print(f'Problem with {_file}.')
# deal with duplicates, keep latest
df = pd.concat(appended_data)
df = df.drop_duplicates(subset=['meta_col_0', 'model_id', 'label_dim'], keep='last')
#print(df.shape, df.columns, df.model_id.unique())

# sanity check
uniq_models=df.model_id.unique() # top_models # 
num_classes=len(label_dims)
#print(len(uniq_models), uniq_models)
   
### Table 2. label-wise p
appended_result=[]
missing_ids=[]
for _model in uniq_models:
    for _label in label_dims:
        _df=df[(df.model_id==_model) & (df.label_dim==_label)]
        #print(_model, _label, _df.shape, _df.head(3))
        # sanity check to have 58 speakers
        if not _df.shape[0] == num_devel_spkrs:
            if _model not in missing_ids:
                missing_ids.append(_model)
        else:
            log_name=_df.log_name.unique()[0]
            assert len(_df.log_name.unique())==1
            pred_array=_df.prediction
            label_array=_df.label
            mpc_per_label=calc_pearsons(pred_array.to_numpy(), label_array.to_numpy())
    
            # label-wise performance
            dct = {'log_name': log_name,
                   'model_id':_model,
                   'label_dim': _label,
                   'mean_pearsons': mpc_per_label}
            appended_result.append(pd.DataFrame([dct]))
    
df_mpc=pd.concat(appended_result)
df_mpc=df_mpc.sort_values(['label_dim', 'mean_pearsons'], ascending=False)
print(df_mpc.shape)

if 1 & len(missing_ids)>0:
    print(f'missing meta cols for {missing_ids}')
    
# write output
if 1:
    csv_path='results/csvs/table2_pred_perception.csv'
    write_to_csv(df_mpc, csv_path)
    
### Table 3. overall p
appended_result=[]
missing_ids=[]
for _model in uniq_models:
    pred_arrays=[]
    label_arrays=[]
    mpc_arrays=[]
    _df=df[(df.model_id==_model)]
    # sanity check to have 16 labels.
    if not len(_df.label_dim.unique()) == len(label_dims):
        if _model not in missing_ids:
            missing_ids.append(_model)
    else:
        for _label in label_dims:
            _df=df[(df.model_id==_model) & (df.label_dim==_label)]
            #print(_model, _label, _df.shape)
            pred_array=_df.prediction
            label_array=_df.label
            mpc_per_label=calc_pearsons(pred_array.to_numpy(), label_array.to_numpy())
            
            pred_arrays.append(pred_array)
            label_arrays.append(label_array)
            mpc_arrays.append(mpc_per_label)
    
        label_arrays=np.array(label_arrays).T # (t, 16)
        pred_arrays=np.array(pred_arrays).T
        #print(pred_arrays.shape, label_arrays.shape, len(mpc_arrays))
        mpc_per_model=mean_pearsons(pred_arrays, label_arrays) # identical results as np.mean
        #print(mpc_per_model, mpc_per_model==np.mean(mpc_arrays))
        
        # model-wise performance
        dct = {'model_id': _model,
                'mean_pearsons': mpc_per_model}
        appended_result.append(pd.DataFrame([dct]))

df_mpc=pd.concat(appended_result)  
df_mpc=df_mpc.sort_values(['mean_pearsons'], ascending=False)
print(df_mpc.shape)

if 1 & len(missing_ids)>0:
    print(f'missing labels for {missing_ids}')
    
# write output
if 1:
    csv_path='results/csvs/table3_pred_perception.csv'
    write_to_csv(df_mpc, csv_path)

sys.exit(0)

### Table 3. Overall mpc across 16 labels. (verify with Nhan's calculation)
uniq_labels=df['label_dim'].unique()
num_classes = len(uniq_labels)
appended_result=[]
missing_ids=[]
for _model in uniq_models:
    result_df=df[df['model_id']==_model]
    #print(result_df.shape, result_df.head(10)) # (928, 5) 928=58*16

    # Initialize empty dictionaries 
    result_prediction = {}
    result_label = {}

    # Group by ID and model since each ID and model combo will give different prediction and label  
    for id, group_df in result_df.groupby(['meta_col_0', 'model_id']):
        # Initialize prediction and label arrays with default value of 0
        prediction_array = np.zeros(num_classes)
        label_array = np.zeros(num_classes)

        if not group_df.shape[0]==len(uniq_labels):
            missing_labels=[item for item in uniq_labels if item not in group_df['label_dim'].unique()]
            if _model not in missing_ids:
                missing_ids.append(_model) 
        else:
            # Match trait position in 'prediction' and 'label' to position in 'unique_traits'
            for index, row in group_df.iterrows():
                trait_position = np.where(uniq_labels == row['label_dim'])[0][0]
                prediction_array[trait_position] = row['prediction']
                label_array[trait_position] = row['label'] # if 'label' in result_df.columns else 0 #?
            
        # Use the gathered prediction and label arrays for this ID and model as value for the dict
        result_prediction[id] = prediction_array
        result_label[id] = label_array

    # Convert the dictionaries back to arrays
    prediction_arrays = np.array(list(result_prediction.values())) 
    label_arrays = np.array(list(result_label.values()))

    # Call the original mean_pearsons function with the new arrays
    assert prediction_arrays.shape[0]==num_devel_spkrs # (t, label_dims)
    result = mean_pearsons(prediction_arrays, label_arrays)
    
    # Store in dict
    dct = {'model_id': _model,
           'mean_pearsons': result}
    appended_result.append(pd.DataFrame([dct]))

# overall mpc
df_mpc=pd.concat(appended_result)
print(df_mpc)
