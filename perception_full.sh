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

model_types=('rnn' 'cnn' 'crnn' 'cnn-attn' 'crnn-attn') #('crnn-attn')  #('rnn') #
labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured') #('aggressive') # 
features=('vit-fer') #('ds') #('faus' 'facenet512' 'vit-fer' 'w2v-msp' 'egemaps' 'ds') #
audio_features=('w2v-msp' 'egemaps' 'ds' 'hubert-superb')  
video_features=('faus' 'facenet512' 'vit-fer')  
text_features=('bert-base-uncased' 'bert-base-multilingual-cased' 'roberta-base' 'xlm-roberta-large' 'gpt2') #('bert-base-uncased') # 
av_features=('avhubert-base-lrs3-iter5' 'avhubert-large-lrs3-iter5' 'avhubert-base-vox-iter5' 'avhubert-large-vox-iter5' 'avhubert-base-noise-pt-noise-ft-30h' 'avhubert-large-noise-pt-noise-ft-30h') #('avhubert-base-lrs3-iter5') # 
feature_lengths=(6 7 8 9 -6 -7 -8 -9) #(10 -10)

# RNN
num_rnn_layers=2
model_dims=(512)

# GENERAL
lrs=(0.0005)
patience=10
n_seeds=5
dropouts=(0.4)
batch_size=32

# adapt
csv='results/csvs/perception.csv'

for model_type in "${model_types[@]}"; do
    for feature in "${features[@]}"; do
        for feature_length in "${feature_lengths[@]}"; do
            for model_dim in "${model_dims[@]}"; do
                for lr in "${lrs[@]}";do
                    for dropout in "${dropouts[@]}";do
                        for label in "${labels[@]}"; do
                            python3 main.py --task perception --feature "$feature" --feature_length "$feature_length" --batch_size $batch_size --model_type $model_type --model_dim $model_dim --label_dim "$label" --rnn_bi --rnn_n_layers $num_rnn_layers --lr "$lr" --n_seeds "$n_seeds" --linear_dropout $dropout --rnn_dropout $dropout --early_stopping_patience 10 --predict --result_csv "$csv" # --save_ckpt #
                        done
                        done
                    done
                done
            done
        done
    done

