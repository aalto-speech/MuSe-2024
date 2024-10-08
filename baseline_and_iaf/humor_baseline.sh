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

# adapt
csv='results/csvs/humor_baseline.csv'

python3 main.py --task humor --feature egemaps --normalize  --model_dim 32 --rnn_n_layers 2 --lr 0.005 --seed 101 --result_csv "$csv" --n_seeds 5 --early_stopping_patience 3 --rnn_dropout 0.5   

python3 main.py --task humor --feature ds  --model_dim 256 --rnn_n_layers 1 --lr 0.001 --seed 101 --result_csv "$csv" --n_seeds 5 --early_stopping_patience 3 --rnn_dropout 0  

python3 main.py --task humor --feature w2v-msp  --model_dim 128 --rnn_n_layers 2 --lr 0.005 --seed 101 --result_csv "$csv" --n_seeds 5 --early_stopping_patience 3 --rnn_dropout 0   

python3 main.py --task humor --feature bert-multilingual  --model_dim 128 --rnn_n_layers 4 --lr 0.001 --seed 101 --result_csv "$csv" --n_seeds 5 --early_stopping_patience 3 --rnn_dropout 0   

python3 main.py --task humor --feature faus  --model_dim 32 --rnn_n_layers 4 --rnn_bi --lr 0.005 --seed 101 --result_csv "$csv" --n_seeds 5 --early_stopping_patience 3 --rnn_dropout 0.5  

python3 main.py --task humor --feature facenet512 --model_dim 64 --rnn_n_layers 4 --lr 0.005 --seed 101 --result_csv "$csv" --n_seeds 5 --early_stopping_patience 3 --rnn_dropout 0.5 

python3 main.py --task humor --feature vit-fer  --model_dim 64 --rnn_n_layers 2 --lr 0.0005 --seed 101 --result_csv "$csv" --n_seeds 5 --early_stopping_patience 3 --rnn_dropout 0.5  
