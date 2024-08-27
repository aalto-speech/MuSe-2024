#!/bin/bash
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

model_types=('iaf') #('iaf' 'lmf' 'tfn' )
audio_features=('w2v-msp' 'hubert-superb') #('w2v-msp' )# 
video_features=('vit-fer') #('vit-fer' 'faus' 'facenet512' )#
text_features=('bert-multilingual') 
feature_lengths=(6 7 8 9 -6 -7 -8 -9) #(10 -10) #(1 -1 2 -2 3 -3 4 -4 5 -5)

# GENERAL
lr=0.0005 #(0.01 0.005 0.0005) # 
patience=10
n_seeds=5
dropout=0.4
batch_size=32

# adapt
csv='./results/csvs/humor_fusion.csv'

for model_type in "${model_types[@]}"; do
    for a_feature in "${audio_features[@]}"; do
        for v_feature in "${video_features[@]}";do
            for t_feature in "${text_features[@]}";do
                for feature_length in "${feature_lengths[@]}";do
                    list="$a_feature $v_feature $t_feature" #'faus facenet512 vit-fer' #'w2v-msp vit-fer bert-multilingual' #
                    echo ${list}
                    python3 main.py --task humor --feature "$list" --feature_length "$feature_length" --model_type "$model_type" --lr "$lr" --batch_size "$batch_size" --n_seeds "$n_seeds" --early_stopping_patience 10 --predict # --result_csv "$csv" 
                done
                done
            done
        done
    done
