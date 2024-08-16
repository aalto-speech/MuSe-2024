import torch.nn as nn
import numpy as np


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=88, nhead=4, num_encoder_layers=2, dim_feedforward=256, mat_dim=2, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_encoder_layers
        )
        # self.fc1 = nn.Linear(input_dim * 4, 64) # for wholedata
        # self.fc1 = nn.Linear(input_dim * 2, 64) # for firsthalf or lasthalf
        self.fc1 = nn.Linear(input_dim * mat_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = x.flatten(1)
        x = self.dropout(self.fc1(x))
        x = self.relu(x)
        x = self.fc2(x)
        return x


        