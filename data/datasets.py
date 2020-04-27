import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import save_data


class SyntheticDataset(Dataset):
    def __init__(self, root, nps, ns, dl, dd, nl, s, p, a, uncentered=False, noisy=False, centers=None, double=False):
        self.root = root
        data = self.load_tcl_data(root, nps, ns, dl, dd, nl, s, p, a, uncentered, noisy, centers)
        self.data = data
        self.s = torch.from_numpy(data['s'])
        self.x = torch.from_numpy(data['x'])
        self.u = torch.from_numpy(data['u'])
        self.l = data['L']
        self.m = data['m']
        self.len = self.x.shape[0]
        self.latent_dim = self.s.shape[1]
        self.aux_dim = self.u.shape[1]
        self.data_dim = self.x.shape[1]
        self.prior = p
        self.activation = a
        self.seed = s
        self.n_layers = nl
        self.uncentered = uncentered
        self.noisy = noisy
        self.double = double

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if not self.double:
            return self.x[index], self.u[index], self.s[index]
        else:
            indices = range(len(self))
            index2 = random.choice(indices)
            return self.x[index], self.x[index2], self.u[index], self.s[index]

    @staticmethod
    def load_tcl_data(root, nps, ns, dl, dd, nl, s, p, a, uncentered, noisy, centers):
        path_to_dataset = root + 'tcl_' + '_'.join(
            [str(nps), str(ns), str(dl), str(dd), str(nl), str(s), p, a])
        if uncentered:
            path_to_dataset += '_u'
        if noisy:
            path_to_dataset += '_noisy'
        path_to_dataset += '.npz'

        if not os.path.exists(path_to_dataset) or s is None:
            kwargs = {"n_per_seg": nps, "n_seg": ns, "d_sources": dl, "d_data": dd, "n_layers": nl, "prior": p,
                      "activation": a, "seed": s, "batch_size": 0, "uncentered": uncentered, "noisy": noisy,
                      "centers": centers, "repeat_linearity": True}
            save_data(path_to_dataset, **kwargs)
        print('loading data from {}'.format(path_to_dataset))
        return np.load(path_to_dataset)

    def get_test_sample(self, batch_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.randint(max(0, self.len - batch_size))
        return self.x[idx:idx + batch_size], self.u[idx:idx + batch_size], self.s[idx:idx + batch_size]
