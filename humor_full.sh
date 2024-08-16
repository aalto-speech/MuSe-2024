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

model_types=('rnn') 
features=('w2v-msp' 'vit-fer' 'bert-multilingual' 'hubert-superb' ) # 'ds' 'egemaps' 'faus' 'facenet512'  'hubert-er'
feature_lengths=(6 7 8 9 -6 -7 -8 -9) #(10 -10) #(1 -1 2 -2 3 -3 4 -4 5 -5)

# RNN
nums_rnn_layers=(2) #(1 2)
model_dims=(256)

# GENERAL
lrs=(0.0005) #(0.01 0.005 0.0005) #
patience=10
n_seeds=5
dropouts=(0.4)
batch_size=32
early_stopping_patience=3

# adapt
csv='results/csvs/humor.csv'

for model_type in "${model_types[@]}"; do
    for num_rnn_layers in "${nums_rnn_layers[@]}"; do
        for feature in "${features[@]}"; do
            for feature_length in "${feature_lengths[@]}"; do
                for model_dim in "${model_dims[@]}"; do
                    for lr in "${lrs[@]}";do
                        for dropout in "${dropouts[@]}";do
                            python3 main.py --task humor --feature "$feature" --feature_length "$feature_length" --batch_size $batch_size --model_type $model_type --model_dim $model_dim --rnn_bi --rnn_n_layers $num_rnn_layers --lr "$lr" --n_seeds "$n_seeds" --linear_dropout $dropout --rnn_dropout $dropout --early_stopping_patience $early_stopping_patience --predict # --result_csv "$csv" 
                        done
                        done
                    done
                done
            done
        done
    done
