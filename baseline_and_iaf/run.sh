#!/bin/bash
#SBATCH --time=00:10:59

#SBATCH --mem=250G
#SBATCH --gres=gpu:1

#SBATCH --partition=gpu-h100-80g 
#SBATCH --gpus=1
#SBATCH --mem=250G

#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load mamba

source activate muse

### C2 Humor ###
# Baseline: humor_full.sh  ~2hrs $csv/humor_baseline.csv
# egemaps torch.Size([256, 4, 88]) 
# w2v-msp torch.Size([256, 4, 1024]) 
# bert torch.Size([256, 4, 768]) 
# faus torch.Size([256, 4, 20])  
# vit-fer torch.Size([256, 4, 768])
python3 main.py --task humor --cache --feature w2v-msp  --model_dim 128 --rnn_n_layers 2 --lr 0.005 --seed 101 --n_seeds 5 --early_stopping_patience 3 --rnn_dropout 0  
python3 main.py --task humor --cache --feature bert-multilingual  --model_dim 128 --rnn_n_layers 4 --lr 0.001 --seed 101 --n_seeds 5 --early_stopping_patience 3 --rnn_dropout 0   
python3 main.py --task humor --cache --feature vit-fer  --model_dim 64 --rnn_n_layers 2 --lr 0.0005 --seed 101 --n_seeds 5 --early_stopping_patience 3 --rnn_dropout 0.5  
# devel: 0.8932 AUC --eval_model /scratch/work/huangg5/muse/MuSe-2024/results/checkpoints/humor/humor_vit_face 
# devel: 0.8915 AUC --eval_model /scratch/elec/puhe/c/muse_2024/results/model_muse/humor/RNN_2024-06-03-11-03_[vit-fer]_[64_2_False_64]_[0.0005_256]
# best a, v, t on devel: w2v-msp (0.8368990086422252), vit-fer (0.8915327287548634), bert-multilingual (0.8123717357278666)
# predict on devel
python3 main.py --task humor --predict --cache --feature vit-fer --eval_seed 102 --eval_model RNN_2024-06-03-11-03_[vit-fer]_[64_2_False_64]_[0.0005_256]
python3 main.py --task humor --predict --cache --feature bert-multilingual --eval_seed 105 --eval_model RNN_2024-06-03-10-50_[bert-multilingual]_[128_4_False_64]_[0.001_256]
python3 main.py --task humor --predict --cache --feature w2v-msp --eval_seed 105 --eval_model RNN_2024-06-03-10-45_[w2v-msp]_[128_2_False_64]_[0.005_256]
# late fusion (June.2024)
python3 late_fusion.py --task humor --result_csv humor_fusion.csv --seeds 102 105 105 --model_ids RNN_2024-06-03-10-45_[w2v-msp]_[128_2_False_64]_[0.005_256] RNN_2024-06-03-11-03_[vit-fer]_[64_2_False_64]_[0.0005_256] RNN_2024-06-03-10-50_[bert-multilingual]_[128_4_False_64]_[0.001_256]
# Weights [0.3915325858420252, 0.33964122017837817, 0.3123721644663806]
# devel: 0.9317 AUC *** 
# rerun feature extraction with correct header DONE hubert 
# test humor on hubert feature with timestamp
# ERR 1210178 hubert-superb uni-modal w. cnn attn mechanisms, no available "hubert-superb" feature files for coach "mourinho", 
# ERR 1235189 hubert-er unimodal
# ERR 1235204 with cnn kernel size, seq_len=4 is short for 4 layered cnn
# add kernel_size argument
python3 main.py --task humor --cache --feature w2v-msp --model_type cnn --kernel_size 1 --model_dim 128 --rnn_bi --rnn_n_layers 2 --lr 0.005 --seed 101 --n_seeds 5 --early_stopping_patience 3 --rnn_dropout 0  
# early fusion 'lmf' 'tfn' 'iaf'
# set batch_size=32 OOM tfn
# devel: 0.8822 AUC missing lr values in csv - iaf egemaps vit-fer bert-multilingual  
# devel: 0.9195 AUC missing lr values in csv - iaf w2v-msp vit-fer bert-multilingual 
# devel: 0.9287 AUC - iaf faus facenet512 vit-fer  *** 
# DONE 1238102 unimodal run pred on humor and save checkpt. log in 'humor/prediction'
# DONE iaf_bst 1238109
# DONE iaf_avt 1238115
# DONE 1258114 humor 'best' baseline, compare results
# 0.8282 AUC 101 RNN_2024-07-09-09-46_[w2v-msp]_[0.005_256]
# 0.8848 AUC 102 RNN_2024-07-09-09-57_[vit-fer]_[0.0005_256] 
# 0.8073 AUC 101 RNN_2024-07-09-09-47_[bert-multilingual]_[0.001_256]
# humor hyper-parameter search
# 0.8314 AUC 101 RNN_2024-07-08-15-48_[w2v-msp]_[0.0005_32] 
# 0.8798 AUC 101 RNN_2024-07-08-15-51_[vit-fer]_[0.0005_32]
# 0.8001 AUC 101 RNN_2024-07-08-15-54_[bert-multilingual]_[0.0005_32] 
# 0.8217 AUC 101 RNN_2024-07-08-15-58_[hubert-superb]_[0.0005_32]
# 0.7670 AUC 101 RNN_2024-07-09-10-09_[faus]_[0.0005_32]
# 0.6531 AUC 102 RNN_2024-07-09-10-12_[facenet512]_[0.0005_32]
# 0.9287 AUC 101 IAF_2024-07-08-15-48_[faus_facenet512_vit-fer]_[0.0005_32]
# 0.9170 AUC 101 IAF_2024-07-08-15-49_[w2v-msp_vit-fer_bert-multilingual]_[0.0005_32]
# DONE 1258916 humor_late_fusion of pseudo-best uni- and multi-modals  
# 0.9353 AUC IAF+IAF  *** 
# train 28168
# devel 11320
# test 23716
# rm data cache
# IAF 1293376 IAF_1 1293379 train best systems on train+devel and pred
python3 main.py --task humor --feature ds --model_type rnn --combine_train_dev
# 0.9999 AUC IAF+IAF humor_late_fusion combine_train_dev overwirte predictions in 'lf'
# DONE 1355744 run pred and save result_csv onf IAF fusion, view log file of combined train_dev, missing log file!
# DONE 1358187 save csv in submission format
# submission: iaf
# late fusion, mehdi
# first last impression: 1426730 full, 1426608 fusion
# DONE 1623270 1623271 +/-10 sec

# TBD format: drop 'label' in prediction_test.csv
# TBD precision, recall, add to csv
# TBD model_dim hyper-params 
# TBD multi-modal fusion with hubert   
# TBD lr hyper-params 

#===============================================================

### C1 Perception ###
## Baseline: perception_full.sh
# update and differenciate BASE_PATH and DATA_PATH
# egemaps torch.Size([59, 80, 88])
# w2v-msp torch.Size([59, 80, 1024])
# ds torch.Size([59, 83, 4096]) 
# faus torch.Size([59, 81, 20])  
# facenet512 torch.Size([59, 81, 512])
# vit-fer torch.Size([59, 81, 768]) 
# avhubert  
# hubert  
# bert
## CLASSIFIER
# DONE 482253 crnn mean pooling, underfitted? 
# DONE 490798 hp search model_dim & lr, no major improvements.
# 24HR TIMELIMIT 514565 crnn-bi-directional 
# DONE 520597 cnn w. self-attention 
# DONE cnn 535999, crnn 536002, crnn_attn 536003, cnn_attn 536004
# 547667 update out layer for cnns to be the same as baseline
python3 main.py --task perception --cache --feature egemaps --batch_size 64 --model_type crnn --model_dim 256 --label_dim aggressive --rnn_n_layers 2 --lr 0.0005 --n_seeds 5 --linear_dropout 0.4 --rnn_dropout 0.4 --early_stopping_patience 10
## FEAT 
# extract hubert-superb-er embeddings/features, wav to dataloader to encodings to embeddings: time_step, win_length
# DONE verify w2v-msp size, label size
# DONE 577464 extract & apply avhubert embeddings/features
python3 main.py --task perception --cache --feature hubert-er --batch_size 64 --model_dim 256 --label_dim aggressive --rnn_n_layers 2 --lr 0.0005 --n_seeds 5 --linear_dropout 0.4 --rnn_dropout 0.4 --early_stopping_patience 10
# DONE extract all val_results in csv e.g. Table 3. average across all targets, mean Pearson's correlation overl 16 target dimensions
# DONE compare hubert with w2v-msp 1024 emotional embeddings.
# DONE df w.r.t. Google_sheet, verify Table 3 calculation: seeds, max>>average, average>>mean,max
# DONE plot df figures: show 16 emotions, show mean performance, show improvements
# window_size, fixed-size sliding window vs. varying-len sliding window
# TEXT bert/roberta/gpt2
# DONE 838825 extract bert/roberta/gpt2 features
# AV-HUBERT 
# avhubert-base/large, pretrained on LRS3 + VoxCeleb2 (1.2G/3.6G) 'base_vox_iter5', 'large_vox_iter5', 
# avhubert-base/large, pretrained on LRS3 (1.2G/3.6G) 'base_lrs3_iter5', 'large_lrs3_iter5',  
# avhubert-base/large, fine-tuned on LRS3-30h for AV-ASR (1.8G/5.3G) 'base_noise_pt_noise_ft_30h', 'large_noise_pt_noise_ft_30h'
# DONE 730605 extract avhubert features
# Possible bug rnn was used instead of cnn - crnn-attn?
# hyper-params for bert and av-hubert?
## LABEL
# multi-label setup - e.g. concatenate labels - dual perspective: agentive8, commonal8; likert16; bem_sex_role_scale (continuous)
# treat labels as VAD grid/coordinates, to emo embedding, to text embedding, to word2vec 
# treat labels as discrete intervals, to GMMs, to clusters
# treat labels as dynamic emotion-shift to 'perception', additional layer-wise classes? CTC decoding?
## FUSION
# use preds from likert16 as input, 'recurrent' feed back, train & validate & test/predict?
# lt ia it w.r.t. efficientface, A, V, AV? forward and main function expect different number of inputs
python3 main.py --task perception --epochs 10  --feature 'w2v-msp vit-fer bert-base-uncased' --model_typ tfn --label_dim aggressive --lr 0.0005 --n_seeds 5 --early_stopping_patience 10
# 0.4190 initial lmf fusion on agressive
# DONE rm 99G models
# DONE rerun 24h save to perception_avt.csv & perception_avt_fusion.csv
# DONE a 845255 v 845257 t 845258,  cnn-attn gets best overall results > rnn > crnn-attn, cnn has missing values
# DONE 853238 extract av features; 
# DONE 871535 rerun perception av  
# fusion hyper-parameters e.g. dropouts. rnn lstm warn: dropout=0.4 and num_layers=1, h_dim=(300, 300, 300)
# debug avt lsf debug features collate, use_gpu, to_cuda on tuples 
# rm model files 'unimodal', fix arg.feature input
# rm legacy models. disk quota limit reached. 
# results/model_muse large collection during trials, save only when get best in csv? or mannualy remove
# add device configurate, remove --use_gpu 
# debug perception a/v/t self-and-cross fusion on lsf & tfn ?erro best model file not found?
# check pred activations, nan in Pearson metric for fusion method .
# OOM reduce batch_size
# sigmoid(relu(_negative))=0.5, r=nan, best_model_file = ''
# pass "$list" in sh scritp to python
# DONE 930239 935350 tfn lmf agreessive 'w2v-msp vit-fer bert-base-uncased'
# DONE 936020 compare lr for tfn lmf on 'agressive', 0.0005 works better for tfn, not conclusive for lmf or 0.01 is less worse.
# models reach 500G after trials
# implement fusion methods e.g. lt, ia, it, dimension mismatch. do not understand conv+ia+conv theory
# DONE iaf agreessive 943807
# DONE save_ckpt option. atm save only logs, but removed logs before 24 June, due to possible file number limit
# DONE default test_scores to 0.0 from eval run on avt, keep logs, analyze results  
# DONE 1005* run on full, a/v/t & fusion: choose 'best' 1~2 features on 2~3 rep. labels: 2*2*2*2=16, e.g. TFN 14G per seed/model?
# DONE 1010088 iaf hyper-parameter n_attn_head 2~6; bi-2-lstm output 4 feat_dim', not enough values for 8 heads 
# DONE 10101** multi-modal vs. single-modal vs. x**3 self-fusion
# rerun feature extraction with correct header 1186608 bert, DONE hubert, DONE avhubert
# DONE 1187592 rerun on hubert, avhuber
# DONE 1205446 rerun on bert text features, check best unimodal p, new fusion with feat combo if necessary.
# best unimodal p 
# cnn-attn_w2v-msp	0.28013125 
# cnn_vit-fer	0.41874375 
# best multimodal p
# iaf faus vit-fer faus 0.5435 ***
# DONE 1258884 save baseline results to csv, compare results
# DONE 1131206 pred with best unimodal: rnn/cnn/crnn/-attn vit_fer
# DONE 1127025 "egemaps vit-fer bert-base-uncased" pred with best unimodal rnn, 11G, ~700Mx16_labels
# DONE 1262013 "faus vit-fer faus" pred with best multimodal: iaf. 
# calculate on pred wrt label_dims, show p, 
# 0.44225 rnn_vit-fer  
# 0.43225 crnn-attn_vit-fer 
# 0.406448 iaf_faus-vit-fer-faus 
# 0.2988165 iaf_egemaps-vit-fer-bert 
# ?fusion results csv contains only one input for 'faus vit-fer faus'. not reliable calculation
# 0.43326 crnn-attn_vit-fer 1266986 run crnn-attn_vit-fer pred and rm model chekpt, verify p
# calculate p & mpcc on prediction for validation, vs. numpy
# DONE av 1283823, 1283809 tfn, 1283805 lmf, rnn-ds 1281198  24HR 1267920~1267934 run c1_full a/v/t/av c1_fusion iaf/lmf/tfn
# DONE verify p mppc 'RNN*ds*0.0005*' mpc: 0.143677 
# 1293635 self fusion iaf on rep avt , 
# DONE 1355961 self fusion iaf on best systems i.e. 3 best v modals vit faus facenet 
# fusion model_id when summarizing results is not consistent
# table 2 (4276, 3)
# table 3 (266, 2)
# ERR late fusion per label, error with label dim and late_fusion.py required input format not clear for c1_perception
# ERR lf save to label dir, try to do late fusion
# ERR rm cache data, combine train dev, train rep systems, error with collate_fn?
# table 2 (5200, 4)
# table 3 (320, 2) # submission 12.July.2024 12:10 noon
# rename meta_col_0 to 'subject_id'
# overall best system  *iaf not done
# get best top models per label_dim, create pred file format, late fusion with max w. to best model and 0 to others
# late fusion on perception
# table 2 (5880, 4)
# table 3 (363, 2)
# latex table 2, table 3
# plot compare model from cnn to crnn-attn to iaf, feat from w2v to bert to avhubert
# first-last impression, 1s, 2s, 3s on audio, vision, text features (what are the timestamps, pseudo time dimension.)
python3 main.py --task perception --epochs 1 --feature_length 3 --feature 'w2v-msp vit-fer bert-base-uncased' --model_typ lmf --n_attn_head 4 --label_dim aggressive
python3 main.py --task perception --epochs 1 --feature_length -2 --feature 'ds' --model_type rnn --label_dim aggressive --predict --result_csv 'results/csvs/debug.csv'                
# 1428148 full, 1426659 1426663 fusion
# 385 systems
# impression at a-v-t positions
# impression vs. attention values, where do the annotation decide on the values?
# unimodal-t_features
# 1631546 1623254 impression +/-10 sec
# mv 'lf' to 'submission' folder

### TBD post muse2024
# plot activation from bert to avhubert.
# TBD table2 table3 differs when summarizing p from dev logs vs. calculate p from preds
# TBD average preds over 5 preds
# TBD mse + p error rate
# TBD multi-label loss
# TBD multi-label system vs. single-label, loss function, calculate p 
# TBD audio + video fusion
# TBD multi-label vs. uni-label n=16: 1-hot, ordinal, 3-d
# TBD e2e with raw wav/mp4 v_feature with EfficientFace?


