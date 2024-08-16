import os, sys
import pathlib
from pathlib import Path
import pandas as pd 
from transformers import pipeline

# muse-2024
text_data = '/scratch/elec/puhe/c/muse_2024/c1_muse_perception/raw/transcriptions'
feat_dir = "/scratch/elec/puhe/c/muse_2024/c1_muse_perception/feature_segments"

# muse-2023 bert-4
if 0:
    f_name = '/scratch/elec/puhe/c/muse_2023/c3_muse_personalisation/feature_segments/bert-4/10.csv'
    df = pd.read_csv(f_name) # (seq_len, 770)
    print(df.shape)

# Define the BERT model checkpoint
checkpoints = ['bert-base-uncased', 'bert-base-multilingual-cased', 'roberta-base', 'xlm-roberta-large', 'gpt2']

# Extract features iteration
onlyfiles = [f for f in os.listdir(text_data) if os.path.isfile(os.path.join(text_data, f)) and not f.startswith('.')]
print(f'Number of speakers: {len(onlyfiles)}')
assert len(onlyfiles)==177

# whether to regnerate feature files
overwrite = True

for checkpoint in checkpoints:
    for _file in onlyfiles:
        spkr_id = Path(_file).stem
        inf_name = os.path.join(text_data, str(spkr_id) + '.csv')
        print(f'Spkr: {spkr_id}, feat: {checkpoint}')
        out_dir = os.path.join(feat_dir, checkpoint) # save features
        out_fname = os.path.join(out_dir, str(spkr_id) + '.csv')
        if overwrite or not os.path.exists(out_fname):
            df = pd.read_csv(inf_name)
            a = list(df['sentence']) 
            text = ' '.join(str(e) for e in a)
            feature_extractor = pipeline("feature-extraction", framework="pt", model=checkpoint)
            feature = feature_extractor(text, return_tensors="pt")[0] # torch.Size([89, 768])
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True) 
            t_np = feature.detach().numpy()
            df = pd.DataFrame(t_np) 
            columns = ['feat_' + str(i) for i in range(t_np.shape[1])]
            df = pd.DataFrame(t_np, columns=columns) 
            # add header column. !Perception data_parser did not throw exception!
            df['timestamp'] = [str(i*500) for i in range(t_np.shape[0])]
            df['subj_id'] = spkr_id
            df = df[ ['timestamp' , 'subj_id'] + [ col for col in df.columns if col not in ['timestamp' , 'subj_id'] ] ]
            df.to_csv(out_fname, index=False)
            #df=pd.read_csv(out_fname, index_col=0)
            #print(df.shape, df.head(3), out_fname)
            #sys.exit(0)

