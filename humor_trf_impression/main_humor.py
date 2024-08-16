import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
import random
import argparse
from utils_humor import MuSeDataset, custom_collate_fn, combine_datasets, combine_features
from model_humor import TransformerClassifier
from pipeline import train, validation, test, test_devel
import warnings as wrn

wrn.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument('--random_seed', type=int, default=102, help='Random seed for reproducibility')
    parser.add_argument('--which_part', type=str, choices=['firsthalf', 'lasthalf', 'wholedata'], default='firsthalf', help='Part of the dataset to use')
    parser.add_argument('--batch_size_train', type=int, default=64, help='Batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    args = parser.parse_args()

    random_seed = int(args.random_seed)
    which_part = str(args.which_part)
    batch_size_train = int(args.batch_size_train)
    batch_size_test = int(args.batch_size_test)
    n_epochs = int(args.n_epochs)
    lr = float(args.lr)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    file_path_vit = f'/scratch/elec/puhe/c/muse_2024/results/data_muse/humor_{which_part}/data_humor_vit-fer__.pkl'
    file_path_bert = f'/scratch/elec/puhe/c/muse_2024/results/data_muse/humor_{which_part}/data_humor_bert-multilingual__.pkl'
    file_path_w2v = f'/scratch/elec/puhe/c/muse_2024/results/data_muse/humor_{which_part}/data_humor_w2v-msp__.pkl'
    file_path_ds = f'/scratch/elec/puhe/c/muse_2024/results/data_muse/humor_{which_part}/data_humor_ds__.pkl'
    file_path_faus = f'/scratch/elec/puhe/c/muse_2024/results/data_muse/humor_{which_part}/data_humor_faus__.pkl'
    file_path_facenet = f'/scratch/elec/puhe/c/muse_2024/results/data_muse/humor_{which_part}/data_humor_facenet512__.pkl'
    file_path_egemaps = f'/scratch/elec/puhe/c/muse_2024/results/data_muse/humor_{which_part}/data_humor_egemaps__norm_.pkl'

    with open(file_path_vit, 'rb') as file:
        data_vit = pickle.load(file)

    with open(file_path_bert, 'rb') as file:
        data_bert = pickle.load(file)

    with open(file_path_w2v, 'rb') as file:
        data_w2v = pickle.load(file)

    data = combine_features(data_vit, data_bert, data_w2v)

    train_dataset = MuSeDataset(data, "train")
    devel_dataset = MuSeDataset(data, "devel")
    test_dataset = MuSeDataset(data, "test")

    train_dataset = combine_datasets(train_dataset, devel_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, collate_fn=custom_collate_fn)
    devel_dataloader = DataLoader(devel_dataset, batch_size=batch_size_test, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, collate_fn=custom_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, _ = next(iter(train_dataloader))
    emb_dim = features.shape[-1]
    N_EPOCHS = n_epochs

    if which_part == "wholedata":
        fc1_mat_dim = 4
    else:
        fc1_mat_dim = 2

    model = TransformerClassifier(
        input_dim=emb_dim, 
        nhead=4, 
        num_encoder_layers=2, 
        dim_feedforward=64, 
        mat_dim=fc1_mat_dim, 
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 0.0001
    criterion = nn.BCELoss()

    best_auc = 0
    model_checkpoint_path = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/MUSE2024/MehediBijoy/checkponts/muse_transformer_v1_{which_part}_td_pc_vit.pth"

    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    prediction, label = test(model, test_dataloader, device)
    whichset = "test"
    test_data_w2v = pd.DataFrame(data_w2v[whichset])
    meta_col_0, meta_col_1, meta_col_2, meta_col_3 = [], [], [], []
    for idx in range(len(test_data_w2v)):
        meta_col_0.append(test_data_w2v["meta"].values[idx][0][0])
        meta_col_1.append(test_data_w2v["meta"].values[idx][0][1])
        meta_col_2.append(test_data_w2v["meta"].values[idx][0][2])
        meta_col_3.append(test_data_w2v["meta"].values[idx][0][3])

    test_df = pd.DataFrame({
        "meta_col_0": meta_col_0,
        "meta_col_1": meta_col_1,
        "meta_col_2": meta_col_2,
        "meta_col_3": meta_col_3,
    })
    test_df['prediction'] = prediction
    test_df['predictedlabel'] = label

    test_df.to_csv(f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/MUSE2024/MehediBijoy/results/masked_preds/predictionsOnTestSet_train_devel_mask_vit.csv", index=False)

    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    prediction, label, truelabel = test_devel(model, devel_dataloader, device)
    whichset = "devel"
    test_data_w2v = pd.DataFrame(data_w2v[whichset])
    meta_col_0, meta_col_1, meta_col_2, meta_col_3 = [], [], [], []
    for idx in range(len(test_data_w2v)):
        meta_col_0.append(test_data_w2v["meta"].values[idx][0][0])
        meta_col_1.append(test_data_w2v["meta"].values[idx][0][1])
        meta_col_2.append(test_data_w2v["meta"].values[idx][0][2])
        meta_col_3.append(test_data_w2v["meta"].values[idx][0][3])

    test_df = pd.DataFrame({
        "meta_col_0": meta_col_0,
        "meta_col_1": meta_col_1,
        "meta_col_2": meta_col_2,
        "meta_col_3": meta_col_3,
    })
    test_df['prediction'] = prediction
    test_df['predictedlabel'] = label
    test_df['truelabel'] = truelabel

    test_df.to_csv(f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/MUSE2024/MehediBijoy/results/masked_preds/predictionsOnDevelSet_train_devel_mask_vit.csv", index=False)

    print("Done")
