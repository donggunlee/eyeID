import os
import subprocess
from tempfile import NamedTemporaryFile
from torch.utils.data.sampler import Sampler

import pdb

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class FeatDataset(Dataset):
    def __init__(self, manifest_filepath, maxval=400.0):
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.maxval = maxval
        super(FeatDataset, self).__init__()

    def __getitem__(self, index):
        sample = self.ids[index]
        feat_path, id = sample[0], sample[1]

        feat = np.load(feat_path, encoding="latin1")
        feat = feat.item()

        feat1 = feat['pos_xn']
        feat2 = feat['pos_yn']
        feat3 = feat['vel_xn']
        feat4 = feat['vel_yn']
        feat5 = feat['pos_xyn']

        feat1 = torch.FloatTensor(feat1)
        feat2 = torch.FloatTensor(feat2)
        feat3 = torch.FloatTensor(feat3)
        feat4 = torch.FloatTensor(feat4)
        feat5 = torch.FloatTensor(feat5)

        feat_set1 = torch.cat((feat1, feat2), 1)
        feat_set2 = torch.cat((feat1, feat2, feat3, feat4), 1)
        feat_set3 = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        trial_idx = feat['trial']  # added
        trial_idx = torch.FloatTensor(trial_idx)  # added

        feat = feat_set2

        # feat = feat[:, :2]
        # feat = torch.FloatTensor(feat)/self.maxval

        # return feat, id
        return feat, id, trial_idx  # added


    def __len__(self):
        return self.size


def _collate_fn_feat(batch):

    nFrame = batch[0][0].size(0)
    nAxis = batch[0][0].size(1)
    minibatch_size = len(batch)
    input = torch.zeros(minibatch_size, nFrame, nAxis) # NxTx2
    target = torch.LongTensor(minibatch_size)

    trial = torch.zeros(minibatch_size, 110)  # added

    for x in range(minibatch_size):
        sample = batch[x]

        tensor = sample[0]
        id = sample[1]
        trial_idx = sample[2]  # added
        trial_idx = trial_idx.transpose(0, 1)  # added

        input[x] = tensor
        target[x] = int(id)-1  # idx starts from 0
        trial[x] = trial_idx  # added

    return input, target, trial
    # return input, target


class FeatLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(FeatLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_feat


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self):
        np.random.shuffle(self.bins)

