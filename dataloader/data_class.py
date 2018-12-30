# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2018 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

from torch.utils.data import Dataset
import numpy as np
import torch
from torch.distributions import Normal
import os

class ConcatDataset(Dataset):
    '''concatentate input images and target segmentation in tuple and return'''
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return min(len(d) for d in self.datasets)

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

class SegmentationDataset(Dataset):
    '''gym carracing random rollout observation -> segmentation pairs dataset'''

    def __init__(self, data_dir, transform=None):
        super(SegmentationDataset, self).__init__()

        '''
        Args:
            data_dir (string): Path to data directory containing .npy files
        '''
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)
        self.file_count = len(self.file_list)

    def __len__(self):
        return self.file_count

    def __getitem__(self, idx):
        data = np.load(self.data_dir + '/' + str(self.file_list[idx]))

        if self.transform:
            data = self.transform(data)

        return data

class ObservationDataset(Dataset):
    '''gym CarRacing-v0 random rollout observations dataset class'''

    def __init__(self, data_dir, transform=None):
        super(ObservationDataset, self).__init__()
        '''
        Args:
            data_dir (string): Path to data directory containing .npy files
        '''
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)
        self.file_count = len(self.file_list)

    def __len__(self):
        return self.file_count

    def __getitem__(self, idx):
        # data = np.load(self.data_dir + '/' + str(idx).zfill(7) + '.npy')
        data = np.load(self.data_dir + '/' + str(self.file_list[idx]))

        if self.transform:
            data = self.transform(data)

        return data

class RolloutDataset(Dataset):
    '''class for loading gym CarRacing-v0 random rollout sequences'''

    def __init__(self, act_dir, mean_dir, var_dir, transform=None):
        super(RolloutDataset, self).__init__()
        '''
            a_dir (string): Path to directory containing numpy arrays of actions by rollout
            m_dir (string): Path to directory containing numpy arrays of z-vector means by rollout
            s_dir (string): Path to directory containing numpy arrays of z-vector variances by rollout
        '''
        self.act_dir = act_dir
        self.act_list = sorted(os.listdir(act_dir))
        self.mean_dir = mean_dir
        self.mean_list = sorted(os.listdir(mean_dir))
        self.var_dir = var_dir
        self.var_list = sorted(os.listdir(var_dir))

        self.transform = transform
        self.file_count = len(self.act_list)


    def __len__(self):
        return self.file_count

    def __getitem__(self, idx):
        act = np.load(self.act_dir + '/' + str(self.act_list[idx]))
        mean = np.load(self.mean_dir + '/' + str(self.mean_list[idx]))
        var = np.load(self.var_dir + '/' + str(self.var_list[idx]))
        # actions generated prior to rollout, in some cases they are longer than total rollout length
        act = act[:mean.shape[0]]

        act = torch.from_numpy(act).reshape(-1, 1, 3).float()
        mean = torch.from_numpy(mean).reshape(-1, 1, 32).float()
        std = torch.from_numpy(var).reshape(-1, 1, 32).sqrt()

        m = torch.distributions.normal.Normal(loc=mean, scale=std)
        obs = m.sample()
        x = torch.cat((obs, act), 2)[:-1, :, :]
        y = obs[1:, :, :]

        return x, y
