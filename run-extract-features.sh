#!/bin/bash
#SBATCH --time=10:59:59
#SBATCH --partition=gpu-h100-80g 
#SBATCH --gpus=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load mamba

source activate muse

echo "extract feature from pretrained hubert-superb emotion recognitoin model."
/scratch/work/huangg5/.conda_envs/muse/bin/python extract_hubert_emo_features.py c2_muse_humor hubert-er
#/scratch/work/huangg5/.conda_envs/muse/bin/python extract_hubert_emo_features.py c1_muse_perception hubert-er

#echo "extract text feature from bert family."
#/scratch/work/huangg5/.conda_envs/muse/bin/python extract_bert_features.py # perception only


