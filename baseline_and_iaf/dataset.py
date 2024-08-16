import sys
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset


class MultiModalMuSeDataset(Dataset):
    def __init__(self, data, partition):
        super(MultiModalMuSeDataset, self).__init__()
        
        """print(data['audio']['train'].keys())
        sys.exit(0)"""
        features_a = data['audio'][partition]['feature']
        features_v = data['video'][partition]['feature']
        features_t = data['text'][partition]['feature']
        labels = data['audio'][partition]['label']
        metas = data['audio'][partition]['meta']
        
        self.feature_dim_a = features_a[0].shape[-1]
        self.feature_dim_v = features_v[0].shape[-1]
        self.feature_dim_t = features_t[0].shape[-1]
        self.n_samples = len(features_a)

        self.feature_lens, self.features_a = self.pad_feature_to_max_len(features_a)
        self.feature_lens_v, self.features_v = self.pad_feature_to_max_len(features_v)
        self.feature_lens_t, self.features_t = self.pad_feature_to_max_len(features_t)
        self.labels = torch.tensor(np.array(labels), dtype=torch.float)
        self.metas = metas
        #print(self.feature_lens_a, self.feature_lens_v, self.feature_lens_t)
        #print(self.features_a[0].shape[0], self.features_v[0].shape[0], self.features_t[0].shape[0])
        pass
        
    def pad_feature_to_max_len(self, features):
        """
        :return:
            a tensor of shape seq_len, feature_dim
        """
        feature_lens = []
        for feature in features:
            feature_lens.append(len(feature))
        max_feature_len = np.max(np.array(feature_lens))
        feature_lens = torch.tensor(feature_lens)
        features = [np.pad(f, pad_width=((0, max_feature_len-f.shape[0]),(0,0))) for f in features]
        features = [torch.tensor(f, dtype=torch.float) for f in features]
        return feature_lens, features

    def get_feature_dim(self):
        return (self.feature_dim_a, self.feature_dim_v, self.feature_dim_t)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        :param idx:
        :return: feature, feature_len, label, meta with
            feature: tuple of (feat_a, feat_v, feat_t), each feat is a tensor of shape seq_len, feature_dim
            feature_len: legacy parameter, not relevant in multimodal muse. int tensor, length of the audio feature tensor before padding
            label: tensor of corresponding label(s) (shape 1 for n-to-1, else (seq_len,1))
            meta: list of lists containing corresponding meta data
        """
        features = [self.features_a[idx], self.features_v[idx], self.features_t[idx]]
        # different sized tensors: list/tuple to tensor, if use_gpu .to_cuda()?
        # features = [torch.from_numpy(item).float() for item in features] 
        # features = [torch.tensor(f, dtype=torch.float) for f in features]
        # features = torch.Tensor(features)
        feature_len = [self.feature_lens[idx], self.feature_lens_v[idx], self.feature_lens_t[idx]]
        label = self.labels[idx]
        meta = self.metas[idx]

        sample = features, feature_len, label, meta
        return sample


class MuSeDataset(Dataset):
    def __init__(self, data, partition):
        super(MuSeDataset, self).__init__()
        self.partition = partition
        features, labels = data[partition]['feature'], data[partition]['label']
        metas = data[partition]['meta']
        self.feature_dim = features[0].shape[-1]
        self.n_samples = len(features)

        feature_lens = []
        label_lens = []
        for feature in features:
            feature_lens.append(len(feature))
        label_lens.append(1)

        max_feature_len = np.max(np.array(feature_lens))
        max_label_len = np.max(np.array(label_lens))
        if max_label_len > 1:
            assert(max_feature_len==max_label_len)

        self.feature_lens = torch.tensor(feature_lens)

        features = [np.pad(f, pad_width=((0, max_feature_len-f.shape[0]),(0,0))) for f in features]
        self.features = [torch.tensor(f, dtype=torch.float) for f in features]
        if max_label_len > 1:
            labels = [np.pad(l, pad_width=((0, max_label_len-l.shape[0]),(0,0))) for l in labels]
        self.labels = torch.tensor(np.array(labels), dtype=torch.float)
        self.metas = metas
        pass

    def get_feature_dim(self):
        return self.feature_dim

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """

        :param idx:
        :return: feature, feature_len, label, meta with
            feature: tensor of shape seq_len, feature_dim
            feature_len: int tensor, length of the feature tensor before padding
            label: tensor of corresponding label(s) (shape 1 for n-to-1, else (seq_len,1))
            meta: list of lists containing corresponding meta data
        """
        feature = self.features[idx]
        feature_len = self.feature_lens[idx]
        label = self.labels[idx]
        meta = self.metas[idx]

        sample = feature, feature_len, label, meta
        return sample

def custom_collate_fn(data):
    """
    Custom collate function to ensure that the meta data are not treated with standard collate, but kept as ndarrays
    :param data: features, feature_lens, labels, metas 
    :return:
    """
    tensors = [d[:3] for d in data]
    np_arrs = [d[3] for d in data]
    coll_features, coll_feature_lens, coll_labels = default_collate(tensors)
    np_arrs_coll = np.row_stack(np_arrs)
    # print(len(data), len(tensors), len(coll_features))
    return coll_features, coll_feature_lens, coll_labels, np_arrs_coll

def custom_mm_collate_fn(data):
    """
    Custom collate function for multi-modal data input
    :param data: feat_a, feat_v, feat_t, feature_lens, labels, metas
    :return:
    """
    tensors = [d[:5] for d in data]
    np_arrs = [d[5] for d in data]
    coll_feat_a, coll_feat_v, coll_feat_t, coll_feature_lens, coll_labels = default_collate(tensors)
    coll_features = (coll_feat_a, coll_feat_v, coll_feat_t)
    np_arrs_coll = np.row_stack(np_arrs)
    return coll_features, coll_feature_lens, coll_labels, np_arrs_coll
