#!/bin/bash
#SBATCH --time=00:59:59
#SBATCH --partition=gpu-h100-80g 
#SBATCH --gpus=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load mamba

source activate muse

echo "late fusion."

## humor
csv='humor_late_fusion.csv'
# unimodals 0.9174 0.8838
# python3 late_fusion.py --task humor --result_csv $csv --model_ids RNN_2024-07-08-15-48_[w2v-msp]_[0.0005_32] RNN_2024-07-08-15-51_[vit-fer]_[0.0005_32] RNN_2024-07-08-15-54_[bert-multilingual]_[0.0005_32]
# python3 late_fusion.py --task humor --result_csv $csv --model_ids RNN_2024-07-09-10-09_[faus]_[0.0005_32] RNN_2024-07-09-10-12_[facenet512]_[0.0005_32] RNN_2024-07-08-15-51_[vit-fer]_[0.0005_32]
# uni+multi 0.9219 0.9352 0.9261 0.9261
# python3 late_fusion.py --task humor --result_csv $csv --model_ids RNN_2024-07-08-15-48_[w2v-msp]_[0.0005_32] RNN_2024-07-08-15-51_[vit-fer]_[0.0005_32] RNN_2024-07-08-15-54_[bert-multilingual]_[0.0005_32] IAF_2024-07-08-15-49_[w2v-msp_vit-fer_bert-multilingual]_[0.0005_32]
# python3 late_fusion.py --task humor --result_csv $csv --model_ids RNN_2024-07-08-15-48_[w2v-msp]_[0.0005_32] RNN_2024-07-08-15-51_[vit-fer]_[0.0005_32] RNN_2024-07-08-15-54_[bert-multilingual]_[0.0005_32] IAF_2024-07-08-15-48_[faus_facenet512_vit-fer]_[0.0005_32]
# python3 late_fusion.py --task humor --result_csv $csv --model_ids RNN_2024-07-09-10-09_[faus]_[0.0005_32] RNN_2024-07-09-10-12_[facenet512]_[0.0005_32] RNN_2024-07-08-15-51_[vit-fer]_[0.0005_32] IAF_2024-07-08-15-49_[w2v-msp_vit-fer_bert-multilingual]_[0.0005_32]
# python3 late_fusion.py --task humor --result_csv $csv --model_ids RNN_2024-07-09-10-09_[faus]_[0.0005_32] RNN_2024-07-09-10-12_[facenet512]_[0.0005_32] RNN_2024-07-08-15-51_[vit-fer]_[0.0005_32] IAF_2024-07-08-15-48_[faus_facenet512_vit-fer]_[0.0005_32]
# mutlti+multi 0.9353
python3 late_fusion.py --task humor --result_csv $csv --submission_format --model_ids IAF_2024-07-08-15-48_[faus_facenet512_vit-fer]_[0.0005_32] IAF_2024-07-08-15-49_[w2v-msp_vit-fer_bert-multilingual]_[0.0005_32]
# mutlti+multi train on train+devel data, late fusion 0.9999
python3 late_fusion.py --task humor --result_csv $csv --submission_format --lf_dir 'lf_train_deval' --model_ids IAF_2024-07-11-10-51-combine-train-dev_[faus_facenet512_vit-fer]_[0.0005_32] IAF_2024-07-11-10-51-combine-train-dev_[w2v-msp_vit-fer_bert-multilingual]_[0.0005_32]
