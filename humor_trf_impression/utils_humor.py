import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset
import numpy as np


def smooth_labels(labels, smoothing=0.1):
    confidence = 1.0 - smoothing
    num_classes = labels.size(1)
    smooth_labels = labels * confidence + (smoothing / num_classes)
    return smooth_labels



class MuSeDataset(Dataset):
    def __init__(self, data, partition, prev_frames=0, next_frames=0):
        super(MuSeDataset, self).__init__()
        self.partition = partition
        self.prev_frames = prev_frames
        self.next_frames = next_frames
        features, labels = data[partition]['feature'], data[partition]['label']
        metas = data[partition]['meta']
        self.feature_dim = features[0].shape[-1]
        self.n_samples = len(features)

        feature_lens = [len(feature) for feature in features]
        self.feature_lens = torch.tensor(feature_lens)

        self.features = [torch.tensor(f, dtype=torch.float) for f in features]
        self.labels = torch.tensor(np.array(labels), dtype=torch.float)  # Use float for BCELoss
        self.metas = metas

    def get_feature_dim(self):
        return self.feature_dim

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature_len = self.feature_lens[idx]
        label = self.labels[idx]
        meta = self.metas[idx]

        # Calculate the total number of frames to be included (previous + current + next)
        total_frames = self.prev_frames + 1 + self.next_frames
        padded_feature = torch.zeros((feature_len, total_frames * self.feature_dim))

        for i in range(feature_len):
            frame_list = []

            # Add previous frames
            for p in range(self.prev_frames, 0, -1):
                if i - p < 0:
                    frame_list.append(torch.zeros(self.feature_dim))  # Use zero vector if previous frame doesn't exist
                else:
                    frame_list.append(feature[i - p])

            # Add current frame
            frame_list.append(feature[i])

            # Add next frames
            for n in range(1, self.next_frames + 1):
                if i + n >= feature_len:
                    frame_list.append(torch.zeros(self.feature_dim))  # Use zero vector if next frame doesn't exist
                else:
                    frame_list.append(feature[i + n])

            # Concatenate all the frames to form the padded feature
            padded_feature[i] = torch.cat(frame_list, dim=-1)

        sample = padded_feature, feature_len, label, meta
        return sample



def custom_collate_fn(batch):
    features = [item[0] for item in batch]
    feature_lens = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    metas = [item[3] for item in batch]

    padded_features = pad_sequence(features, batch_first=True)
    feature_lens = torch.tensor(feature_lens)
    labels = torch.stack(labels)
    
    # return padded_features, feature_lens, labels, metas
    return padded_features, labels



def combine_datasets(train_dataset, devel_dataset):
    combined_dataset = ConcatDataset([train_dataset, devel_dataset])
    return combined_dataset



def pad_array(arr, max_first_dim):
    padded_arr = np.zeros((max_first_dim, arr.shape[1]), dtype=arr.dtype)
    padded_arr[:arr.shape[0], :arr.shape[1]] = arr
    return padded_arr



def combine_features(feat1, feat2, feat3=None):
    combined_data = feat1

    def process_partition(partition):
        for idx in range(len(feat1[partition]['feature'])):
            if feat3 is not None:
                max_first_dim = max(feat1[partition]['feature'][idx].shape[0], 
                                    feat2[partition]['feature'][idx].shape[0], 
                                    feat3[partition]['feature'][idx].shape[0])
                feat1[partition]['feature'][idx] = pad_array(feat1[partition]['feature'][idx], max_first_dim)
                feat2[partition]['feature'][idx] = pad_array(feat2[partition]['feature'][idx], max_first_dim)
                feat3[partition]['feature'][idx] = pad_array(feat3[partition]['feature'][idx], max_first_dim)
                
                combined_data[partition]['feature'][idx] = np.concatenate(
                    (
                        np.zeros_like(feat1[partition]['feature'][idx]), 
                        feat2[partition]['feature'][idx], 
                        feat3[partition]['feature'][idx]
                    ), 
                    axis=1
                )
            else:
                max_first_dim = max(feat1[partition]['feature'][idx].shape[0], 
                                    feat2[partition]['feature'][idx].shape[0])
                feat1[partition]['feature'][idx] = pad_array(feat1[partition]['feature'][idx], max_first_dim)
                feat2[partition]['feature'][idx] = pad_array(feat2[partition]['feature'][idx], max_first_dim)

                combined_data[partition]['feature'][idx] = np.concatenate(
                    (
                        feat1[partition]['feature'][idx], 
                        feat2[partition]['feature'][idx]
                    ), 
                    axis=1
                )

    for partition in ['train', 'devel', 'test']:
        process_partition(partition)
    
    return combined_data



