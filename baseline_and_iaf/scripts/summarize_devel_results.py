import os, sys, re
import pandas as pd
import numpy as np 
import itertools
import ast
import pathlib
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) # avoid xlabels to be cut off

model_types = ('rnn', ) 
model_types += ('crnn', 'cnn', 'cnn-attn', 'crnn-attn',)
fusion_models = ('tfn', 'lmf', 'iaf')
a_features = ('egemaps', 'w2v-msp', 'ds', 'hubert-superb') 
v_features = ('faus', 'facenet512', 'vit-fer') 
t_features = ('bert-base-uncased', 'bert-base-multilingual-cased', 'roberta-base', 'xlm-roberta-large', 'gpt2')
av_features = ('avhubert-base-lrs3-iter5', 'avhubert-large-lrs3-iter5', 'avhubert-base-vox-iter5', 'avhubert-large-vox-iter5', 'avhubert-base-noise-pt-noise-ft-30h', 'avhubert-large-noise-pt-noise-ft-30h')
features = set(av_features).union(a_features, v_features, t_features)
fusion_features = set(['+'.join([i,j,k]) for i in a_features for j in v_features for k in t_features])
label_dims = ('aggressive', 'confident', 'good_natured',) 
label_dims += ('arrogant', 'assertiv',  'dominant', 'independent', 'risk', 'leader_like', 'collaborative', 'enthusiastic', 'friendly', 'kind', 'likeable', 'sincere',  'warm') 

unimodal_sets = list(itertools.product(list(model_types), list(features)))
multimodal_sets = list(itertools.product(list(fusion_models), list(fusion_features)))
_full_sets = unimodal_sets + multimodal_sets
assert len(_full_sets) == len(unimodal_sets) + len(multimodal_sets)

task='perception' 
csv_ins=(f'results/csvs/{task}.csv',)
csv_ins+=(f'results/csvs/{task}_fusion.csv',)
csv_path=f'results/csvs/{task}_analysis.csv'
metric = 'best_val_Pearson'

# verify baseline table2 & table3 numbers
# https://docs.google.com/spreadsheets/d/1qR8YCxBqnx5o9RiGvDO76w3EZx4PI2v71fe_nQg1rL4/edit?usp=sharing
df0 = pd.DataFrame()
for _csv in csv_ins:
    _df = pd.read_csv(_csv, index_col=0)
    df0 = pd.concat([df0, _df])
    # print(_df.columns, df0.columns)
#model_types = set(df0.model_type.tolist())
#features = set(df0.feature.tolist())
#print(df0.shape,  model_types, features)

# get Pearson per seed from log file, output new csv
if 0:
    df = df0
else:
    for (_model, _feat) in unimodal_sets: # _full_sets:
        for _label in label_dims:
            _df = df0[(df0.feature==_feat) & (df0.label_dim==_label) & (df0.model_type==_model)]
            if _df.shape[0]==0:
                print('No results on combination:', _model, _feat, _label, _df.shape)
            elif _df.shape[0]>=1:
                # pd.options.display.max_colwidth = 500
                # Deal with runs that produce duplicated or updated results in df. keep latest.
                _dict = ast.literal_eval(_df['paths'].values[-1])
                model_name = os.path.basename(_dict['model'])
                log_fname = os.path.join(_dict['log'], model_name + '.txt')
                if not os.path.exists(log_fname):
                    print(log_fname)
                    sys.exit(0)
                else:
                    # Open and read the file.
                    with open(log_fname, 'rt', encoding='utf-8') as f:
                        text = f.read()
                    # "ID/Seed 101 | Best [Val Pearson]: 0.1720 |  Loss: ..."
                    # "ID/Seed 101 | Best [Val Pearson]:-0.0965 | Loss:..."
                    # Extract all the data tuples with a findall() each tuple is: (seed, Best Val Pearson)
                    tuples = re.findall(r'ID\/Seed\s(\d+)\s\|\sBest\s\[Val\sPearson\]\:\s*(-?\d+\.\d+)\s\|\s', text)
                    dct = {metric: [float(v) for (k,v) in tuples]}
                    dct.update({'seed': [k for (k,v) in tuples]})
                
                    dct.update({'model_type':_model})
                    dct.update({'feature':_feat})
                    dct.update({'label_dim':_label})
                    df = pd.DataFrame(dct)

                    # make sure the directory exists
                    csv_dir = pathlib.Path(csv_path).parent.resolve()
                    os.makedirs(csv_dir, exist_ok=True)

                    # write back
                    if os.path.exists(csv_path):
                        old_df = pd.read_csv(csv_path)
                        df = pd.concat([old_df, df])
                    df.to_csv(csv_path, index=False)

# TBD: which column to aggregate. For all three options, best feat-model combo remains the same.
# https://docs.google.com/spreadsheets/d/1qR8YCxBqnx5o9RiGvDO76w3EZx4PI2v71fe_nQg1rL4/edit?gid=0#gid=0

# df = pd.read_csv(csv_path)
if 1:
    # option I: group by best_seeds      
    df1 = df.groupby(['model_type', 'feature', 'label_dim']).agg(max_p=(metric, 'max'))           
    df2 = df1.groupby(['model_type', 'feature']).agg({'max_p': ['max', 'mean', 'std']})
    #print(df2)

if 1:
    # option II: group by seeds
    df3 = df.groupby(['model_type', 'feature', 'label_dim']).agg(mean_p=(metric, 'mean'))
    df4 = df3.groupby(['model_type', 'feature']).agg({'mean_p': ['max', 'mean', 'std']})
    #print(df4)

if 1:
    # option III: group by labels
    df5 = df.groupby(['model_type', 'feature', 'seed']).agg(mean_p=(metric, 'mean'))
    df6 = df5.groupby(['model_type', 'feature']).agg({'mean_p': ['max', 'mean', 'std']})
    #print(df5)

df1.to_csv(f'results/csvs/table2_{task}.csv')
df6.to_csv(f'results/csvs/table3_{task}.csv')


if 0:
    # png
    plot = df2.plot(kind ='bar', title="Group by seeds")
    ax = df2.plot.bar()
    ax.set_title("Group by seeds",color='black')
    ax.legend(bbox_to_anchor=(1.0, 1.0))
    ax.plot()
    plt.tight_layout()
    plt.savefig('results/pngs/perception_analysis.png')

