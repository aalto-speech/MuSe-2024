# extract emotional features from audio using HuBert
# source: https://huggingface.co/superb/hubert-large-superb-er

import sys, os
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path
import glob
import librosa
from tqdm import tqdm
import torch
import torch.nn as nn
import datasets
from datasets import load_dataset
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

def rolling_window(a, window, step):
    """
    Reshape a numpy array 'a' of shape (n) to form shape((n - window_size) // step + 1, window_size))

    Create a function to reshape a 1d array using a sliding window with a step.
    NOTE: The function uses numpy's internat as_strided function because looping in python is slow in comparison.

    Adopted from http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html and 
    https://gist.github.com/codehacken/708f19ae746784cef6e68b037af65788

    """
    shape = a.shape[:-1] + ((a.shape[-1] - window + 1)//step, window)
    strides = (a.strides[0] * step,) + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# muse  baseline features
if 0:
    """
    perception
    time_step, feat_dim
    win_len=2000ms, hop_len=500ms, audio_len=30s
    30/0.5=60
    0 w2v-msp (58, 1025)
    0 egemaps (58, 89)
    0 vit-fer (62, 769)
    1 w2v-msp (57, 1025)
    1 egemaps (57, 89)
    1 vit-fer (61, 769)

    # percpetion: timestamp,subj_id,feature_1,feature_2... e.g. faus
    # timestamp,subj_id,AU01,AU02,AU04,AU05,AU06,AU07,AU09,AU10,AU11,AU12,AU14,AU15,AU17,AU20,AU23,AU24,AU25,AU26,AU28,AU43
    # 0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0
    # 500,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0
    
    # humor: timestamp,segment_id, 0,1,2,... e.g. faus
    # timestamp,segment_id,AU01,AU02,AU04,AU05,AU06,AU07,AU09,AU10,AU11,AU12,AU14,AU15,AU17,AU20,AU23,AU24,AU25,AU26,AU28,AU43
    # 0,baum_01_1116500_1133500,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    # 500,baum_01_1116500_1133500,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

    # feature file format only throws exceptions during humor task
    """
    for spkr in [0, 1, 10, 100]:
        for feat in ['w2v-msp', 'egemaps', 'vit-fer']:
        
            fname = f'/scratch/elec/puhe/c/muse_2024/c1_muse_perception/feature_segments/{feat}/{spkr}.csv'
            df = pd.read_csv(fname, index_col=0)
            print(spkr, feat, df.shape)
    sys.exit(0)

# configs
task=sys.argv[1] #'c2_muse_humor' 'c1_muse_perception'#
feature_type=sys.argv[2] #'hubert-superb' 'hubert-er'

if task=='c1_muse_perception': 
    audio_data=f'/scratch/elec/puhe/c/muse_2024/{task}/raw/wav'
elif task=='c2_muse_humor':
    audio_data=f'/scratch/elec/puhe/c/muse_2024/{task}/raw_data/audio'
feat_dir=f'/scratch/elec/puhe/c/muse_2024/{task}/feature_segments'
model=HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
feature_extractor=Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")

# hubert-superb-er demo
if 0:
    def map_to_array(example):
        speech, _ = librosa.load(example["file"], sr=16000, mono=True)
        example["speech"] = speech
        return example
    # load a demo dataset and read audio files
    dataset = load_dataset("anton-l/superb_demo", "er", split="session1")
    dataset = dataset.map(map_to_array)

    # compute attention masks and normalize the waveform if needed
    inputs = feature_extractor(dataset[:4]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")
    print(inputs)

    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]

    print(logits.shape, predicted_ids.shape)
    sys.exit(0)

# hubert-superb-er feature extration on muse
if 0:
    """
    logits: [bs, emo_classes]
    model.config.label2id: {
        "ang": 2,
        "hap": 1,
        "neu": 0,
        "sad": 3
    },
    model(**inputs): logits
    """
    # dataset object huggingface
    dataset = load_dataset("audiofolder", data_dir=audio_data, drop_labels=True, drop_metadata=True)
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    audio_arrays = [x["array"] for x in dataset['train']["audio"]]
    audio_lens =[len(_array) for _array in audio_arrays[:10]]
    print(audio_lens)

# whether to regnerate feature files
overwrite = False

if task=='c1_muse_perception':  
    onlyfiles = glob.glob(f'{audio_data}/*.wav',  recursive = True)
else:
    onlyfiles = glob.glob(f'{audio_data}/*/*/*.wav',  recursive = True)
onlyfiles = [f for f in onlyfiles if os.path.isfile(f)]
for _file in tqdm(onlyfiles):
    # tqdm bar 
    if task=='c1_muse_perception':   
        out_dir = os.path.join(feat_dir, feature_type)
        id_header = 'subj_id'
    else:
        out_dir = os.path.join(feat_dir, feature_type, Path(_file).parts[-3])
        id_header = 'segment_id'
    f_id = Path(_file).stem
    out_fname = os.path.join(out_dir, f_id + '.csv')
    if overwrite or not os.path.exists(out_fname):
        print('Extract feature', f_id)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True) 
        input_audio, sample_rate = librosa.load(_file,  sr=16000) # numpy 1.x 2.x incompatibility? 
        #sample_rate, input_audio = wavfile.read(_file)

        # segment audio with win_len=2000ms, hop_len=500ms 
        audio_arrays = rolling_window(input_audio, window=int(2*sample_rate), step=int(0.5*sample_rate))
        audio_lens =[_array.shape for _array in audio_arrays]
        
        # extract features from layers
        assert sample_rate==16000
        inputs = feature_extractor(audio_arrays, sampling_rate=16000, padding=True, return_tensors="pt")
        outputs = model(**inputs) # odict_keys(['logits', 'hidden_states'])
        logits = outputs.logits # torch.Size([57, 4])
        _states = outputs.hidden_states  # tuple, size=25, 
        last_hidden_state = outputs.hidden_states[-1] # torch.Size([56, 99, 1024])
        # averaging rep. in final layer
        activations = torch.mean(last_hidden_state, 1)
        if feature_type=='hubert-er':
            t_np = logits.detach().numpy()    
        else:
            t_np = activations.detach().numpy()   
        # save features (logits_emo4), (h_states1024) for each audio file hubert-superb-er
        df = pd.DataFrame(t_np) 
        columns = ['feat_' + str(i) for i in range(t_np.shape[1])]
        df = pd.DataFrame(t_np, columns=columns) 
        # add header column. !Perception data_parser did not throw exception!
        df['timestamp'] = [str(i*500) for i in range(t_np.shape[0])]
        df[id_header] = f_id
        df = df[['timestamp', id_header] + [ col for col in df.columns if col not in ['timestamp', id_header] ] ]
        df.to_csv(out_fname, index=False)
        #df=pd.read_csv(out_fname, index_col=0)
        #print(df.shape, df.head(3), out_fname)
        #sys.exit(0)
