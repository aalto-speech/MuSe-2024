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

model_types=('rnn') 
labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured') #('aggressive') # 
features=('faus' 'facenet512' 'vit-fer' 'w2v-msp' 'egemaps' 'ds')

# RNN
nums_rnn_layers=(2)
model_dims=(512)

# GENERAL
lrs=(0.0005)
patience=10
n_seeds=5
dropouts=(0.4)
batch_size=32

# adapt
csv='results/csvs/perception_baseline.csv'
for model_type in "${model_types[@]}"; do
    for feature in "${features[@]}"; do
        for num_rnn_layers in "${nums_rnn_layers[@]}"; do
            for model_dim in "${model_dims[@]}"; do
                for lr in "${lrs[@]}";do
                    for dropout in "${dropouts[@]}";do
                        for label in "${labels[@]}"; do
                            python3 main.py --task perception --feature "$feature" --batch_size $batch_size --model_type $model_type --model_dim $model_dim --label_dim "$label" --rnn_bi --rnn_n_layers $num_rnn_layers --lr "$lr" --n_seeds "$n_seeds" --result_csv "$csv" --linear_dropout $dropout --rnn_dropout $dropout --early_stopping_patience 10
                        done
                        done
                    done
                done
            done
        done
    done

