#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load mamba

source activate muse

model_types=('iaf') #('lmf' 'tfn' 'iaf') # 
labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured') # ('aggressive') # 
audio_features=('w2v-msp' 'hubert-superb' 'avhubert-base-lrs3-iter5')
video_features=('vit-fer')   
text_features=('bert-base-uncased' 'bert-base-multilingual-cased') # 'roberta-base' 'xlm-roberta-large' 'gpt2') #('bert-base-uncased') #5# 
av_features=('avhubert-base-lrs3-iter5' 'avhubert-large-lrs3-iter5' 'avhubert-base-vox-iter5' 'avhubert-large-vox-iter5' 'avhubert-base-noise-pt-noise-ft-30h' 'avhubert-large-noise-pt-noise-ft-30h') #('avhubert-base-lrs3-iter5') # 
feature_lengths=(6 7 8 9 -6 -7 -8 -9) #(10 -10) #(1 -1 2 -2 3 -3 4 -4 5 -5)

# GENERAL
lr=0.0005
patience=10
n_seeds=5
dropout=0.4
batch_size=32

# adapt
csv='results/csvs/perception_fusion.csv'

for model_type in "${model_types[@]}"; do
    for feature_length in "${feature_lengths[@]}"; do
        for a_feature in "${audio_features[@]}"; do
            for v_feature in "${video_features[@]}";do
                for t_feature in "${text_features[@]}";do
                    for label in "${labels[@]}"; do
                        list="$a_feature $v_feature $t_feature"   
                        echo ${list}
                        python3 main.py --task perception --feature "$list" --feature_length "$feature_length" --batch_size $batch_size --model_type "$model_type" --label_dim "$label" --lr "$lr" --n_seeds "$n_seeds" --early_stopping_patience 10 --predict --result_csv "$csv" # --save_ckpt 
                    done
                    done
                done
            done
        done
    done

