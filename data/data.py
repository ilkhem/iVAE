"""
Script for generating piece-wise stationary data.

Each component of the independent latents is comprised of `ns` segments, and each segment has different parameters.\
Each segment has `nps` data points (measurements).

The latent components are then mixed by an MLP into observations (not necessarily of the same dimension).
It is possible to add noise to the observations
"""

import os

import numpy as np
import scipy
import torch
from scipy.stats import hypsecant
from torch.utils.data import Dataset


def to_one_hot(x, m=None):
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
    return xoh


def lrelu(x, neg_slope):
    """
    Leaky ReLU activation function
    @param x: input array
    @param neg_slope: slope for negative values
    @return:
        out: output rectified array
    """

    def _lrelu_1d(_x, _neg_slope):
        """
        one dimensional implementation of leaky ReLU
        """
        if _x > 0:
            return _x
        else:
            return _x * _neg_slope

    leaky1d = np.vectorize(_lrelu_1d)
    assert neg_slope > 0  # must be positive
    return leaky1d(x, neg_slope)


def sigmoid(x):
    """
    Sigmoid activation function
    @param x: input array
    @return:
        out: output array
    """
    return 1 / (1 + np.exp(-x))


def generate_mixing_matrix(d_sources: int, d_data=None, lin_type='uniform', cond_threshold=25, n_iter_4_cond=None,
                           dtype=np.float32, staircase=False):
    """
    Generate square linear mixing matrix
    @param d_sources: dimension of the latent sources
    @param d_data: dimension of the mixed data
    @param lin_type: specifies the type of matrix entries; either `uniform` or `orthogonal`.
    @param cond_threshold: higher bound on the condition number of the matrix to ensure well-conditioned problem
    @param n_iter_4_cond: or instead, number of iteration to compute condition threshold of the mixing matrix.
        cond_threshold is ignored in this case/
    @param dtype: data type for data
    @param staircase: if True, generate mixing that preserves staircase form of sources
    @return:
        A: mixing matrix
    @rtype: np.ndarray
    """

    def _gen_matrix(ds, dd, dtype):
        A = (np.random.uniform(0, 2, (ds, dd)) - 1).astype(dtype)
        for i in range(dd):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        return A

    def _gen_matrix_staircase(ds, dd, dtype, sq=None):
        if sq is None:
            sq = dd > 2
        A1 = np.zeros((ds, 1))  # first row of A should be e_1
        A1[0, 0] = 1
        A2 = np.random.uniform(0, 2, (ds, dd - 1)) - 1
        if sq:
            A2[0] = 0
        A = np.concatenate([A1, A2], axis=1).astype(dtype)
        for i in range(dd):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        return A

    if d_data is None:
        d_data = d_sources

    if lin_type == 'orthogonal':
        A = (np.linalg.qr(np.random.uniform(-1, 1, (d_sources, d_data)))[0]).astype(dtype)

    elif lin_type == 'uniform':
        if n_iter_4_cond is None:
            cond_thresh = cond_threshold
        else:
            cond_list = []
            for _ in range(int(n_iter_4_cond)):
                A = np.random.uniform(-1, 1, (d_sources, d_data)).astype(dtype)
                for i in range(d_data):
                    A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
                cond_list.append(np.linalg.cond(A))

            cond_thresh = np.percentile(cond_list, 25)  # only accept those below 25% percentile

        gen_mat = _gen_matrix if not staircase else _gen_matrix_staircase
        A = gen_mat(d_sources, d_data, dtype)
        while np.linalg.cond(A) > cond_thresh:
            A = gen_mat(d_sources, d_data, dtype)

    else:
        raise ValueError('incorrect method')
    return A


def generate_nonstationary_sources(n_per_seg: int, n_seg: int, d: int, prior='gauss', var_bounds=np.array([0.5, 3]),
                                   dtype=np.float32, uncentered=False, centers=None, staircase=False):
    """
    Generate source signal following a TCL distribution. Within each segment, sources are independent.
    The distribution withing each segment is given by the keyword `dist`
    @param n_per_seg: number of points per segment
    @param n_seg: number of segments
    @param d: dimension of the sources same as data
    @param prior: distribution of the sources. can be `lap` for Laplace , `hs` for Hypersecant or `gauss` for Gaussian
    @param var_bounds: optional, upper and lower bounds for the modulation parameter
    @param dtype: data type for data
    @param bool uncentered: True to generate uncentered data
    @param centers: if uncentered, pass the desired centers to this parameter. If None, the centers will be drawn
                    at random
    @param staircase: if True, s_1 will have a staircase form, used to break TCL.
    @return:
        sources: output source array of shape (n, d)
        labels: label for each point; the label is the component
        m: mean of each component
        L: modulation parameter of each component
    @rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    var_lb = var_bounds[0]
    var_ub = var_bounds[1]
    n = n_per_seg * n_seg

    L = np.random.uniform(var_lb, var_ub, (n_seg, d))
    if uncentered:
        if centers is not None:
            assert centers.shape == (n_seg, d)
            m = centers
        else:
            m = np.random.uniform(-5, 5, (n_seg, d))
    else:
        m = np.zeros((n_seg, d))

    if staircase:
        m1 = 3 * np.arange(n_seg).reshape((-1, 1))
        a = np.random.permutation(n_seg)
        m1 = m1[a]
        # L[:, 0] = .2
        if uncentered:
            m2 = np.random.uniform(-1, 1, (n_seg, d - 1))
        else:
            m2 = np.zeros((n_seg, d - 1))
        m = np.concatenate([m1, m2], axis=1)

    labels = np.zeros(n, dtype=dtype)
    if prior == 'lap':
        sources = np.random.laplace(0, 1 / np.sqrt(2), (n, d)).astype(dtype)
    elif prior == 'hs':
        sources = scipy.stats.hypsecant.rvs(0, 1, (n, d)).astype(dtype)
    elif prior == 'gauss':
        sources = np.random.randn(n, d).astype(dtype)
    else:
        raise ValueError('incorrect dist')

    for seg in range(n_seg):
        segID = range(n_per_seg * seg, n_per_seg * (seg + 1))
        sources[segID] *= L[seg]
        sources[segID] += m[seg]
        labels[segID] = seg

    return sources, labels, m, L


def generate_data(n_per_seg, n_seg, d_sources, d_data=None, n_layers=3, prior='gauss', activation='lrelu', batch_size=0,
                  seed=10, slope=.1, var_bounds=np.array([0.5, 3]), lin_type='uniform', n_iter_4_cond=1e4,
                  dtype=np.float32, noisy=0, uncentered=False, centers=None, staircase=False, discrete=False,
                  one_hot_labels=True, repeat_linearity=False, simple_mixing=True, mix_bounds=np.array([-1, 1])):
    """
    Generate artificial data with arbitrary mixing
    @param int n_per_seg: number of observations per segment
    @param int n_seg: number of segments
    @param int d_sources: dimension of the latent sources
    @param int or None d_data: dimension of the data
    @param int n_layers: number of layers in the mixing MLP
    @param str activation: activation function for the mixing MLP; can be `none, `lrelu`, `xtanh` or `sigmoid`
    @param str prior: prior distribution of the sources; can be `lap` for Laplace or `hs` for Hypersecant
    @param int batch_size: batch size if data is to be returned as batches. 0 for a single batch of size n
    @param int seed: random seed
    @param np.ndarray var_bounds: upper and lower bounds for the modulation parameter
    @param float slope: slope parameter for `lrelu` or `xtanh`
    @param str lin_type: specifies the type of matrix entries; can be `uniform` or `orthogonal`
    @param int n_iter_4_cond: number of iteration to compute condition threshold of the mixing matrix
    @param dtype: data type for data
    @param float noisy: if non-zero, controls the level of noise added to observations
    @param bool uncentered: True to generate uncentered data
    @param np.ndarray centers: array of centers if uncentered == True
    @param bool staircase: if True, generate staircase data
    @param bool one_hot_labels: if True, transform labels into one-hot vectors
    @param bool simple_mixing: if True, have elements of mixing matrix from a Uniform distribution \
    and skip all the other mixing code
    @param np.ndarray mix_bounds: upper and lower bounds of the Uniform distribution \
    (only active when simple_mixing == True)

    @return:
        tuple of batches of generated (sources, data, auxiliary variables, mean, variance, mixing matrix)
    @rtype: tuple

    """
    if seed is not None:
        np.random.seed(seed)

    if d_data is None:
        d_data = d_sources

    # sources
    S, U, M, L = generate_nonstationary_sources(n_per_seg, n_seg, d_sources, prior=prior,
                                                var_bounds=var_bounds, dtype=dtype,
                                                uncentered=uncentered, centers=centers, staircase=staircase)
    n = n_per_seg * n_seg
    
    # non linearity
    if activation == 'lrelu':
        act_f = lambda x: lrelu(x, slope).astype(dtype)
    elif activation == 'sigmoid':
        act_f = sigmoid
    elif activation == 'xtanh':
        act_f = lambda x: np.tanh(x) + slope * x
    elif activation == 'none':
        act_f = lambda x: x
    else:
        raise ValueError('incorrect non linearity: {}'.format(activation))

    # Mixing time!

    if simple_mixing:
        A = np.random.uniform(mix_bounds[0], mix_bounds[1], (d_sources, d_sources)).astype(dtype)
        X = np.dot(S, A)
        
    else:
        if not repeat_linearity:
            X = S.copy()
            for nl in range(n_layers):
                A = generate_mixing_matrix(X.shape[1], d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype,
                                           staircase=staircase)
                if nl == n_layers - 1:
                    X = np.dot(X, A)
                else:
                    X = act_f(np.dot(X, A))

        else:
            assert n_layers > 1  # suppose we always have at least 2 layers. The last layer doesn't have a non-linearity
            A = generate_mixing_matrix(d_sources, d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
            X = act_f(np.dot(S, A))
            if d_sources != d_data:
                B = generate_mixing_matrix(d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
            else:
                B = A
            for nl in range(1, n_layers):
                if nl == n_layers - 1:
                    X = np.dot(X, B)
                else:
                    X = act_f(np.dot(X, B))
                    

    # add noise:
    if noisy:
        X += noisy * np.random.randn(*X.shape)

    if discrete:
        X = np.random.binomial(1, sigmoid(X))

    if not batch_size:
        if one_hot_labels:
            U = to_one_hot([U], m=n_seg)[0]
            
        # if U is a vector, transform it in a matrix, so that aux_dim=1
        try:
            U.shape[1]
        except:
            U = np.expand_dims(U, axis=1)
        return S, X, U, M, L, A
    
    else:
        idx = np.random.permutation(n)
        Xb, Sb, Ub, Mb, Lb = [], [], [], [], []
        n_batches = int(n / batch_size)
        for c in range(n_batches):
            Sb += [S[idx][c * batch_size:(c + 1) * batch_size]]
            Xb += [X[idx][c * batch_size:(c + 1) * batch_size]]
            Ub += [U[idx][c * batch_size:(c + 1) * batch_size]]
            Mb += [M[idx][c * batch_size:(c + 1) * batch_size]]
            Lb += [L[idx][c * batch_size:(c + 1) * batch_size]]
        if one_hot_labels:
            Ub = to_one_hot(Ub, m=n_seg)
            
    # if U is a vector, transform it in a matrix, so that aux_dim=1
    try:
        U.shape[1]
    except:
        U = np.expand_dims(U, axis=1)
        
        return Sb, Xb, Ub, Mb, Lb, A


def save_data(path, *args, **kwargs):
    """
    Generate data and save it.
    :param str path: path where to save the data
    """
    kwargs['batch_size'] = 0  # leave batch creation to torch DataLoader
    S, X, U, M, L, A = generate_data(*args, **kwargs)
    print('Creating dataset {} ...'.format(path))
    dir_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs('/'.join(path.split('/')[:-1]))
    np.savez_compressed(path, s=S, x=X, u=U, m=M, L=L, A=A)
    print(' ... done')


class SyntheticDataset(Dataset):
    def __init__(self, root, nps, ns, dl, dd, nl, s, p, a, uncentered=False, noisy=False, centers=None, double=False,
                 one_hot_labels=False, simple_mixing=True):
        self.root = root
        data = self.load_tcl_data(root, nps, ns, dl, dd, nl, s, p, a, uncentered, noisy, centers, one_hot_labels)
        self.data = data
        self.s = torch.from_numpy(data['s'])
        self.x = torch.from_numpy(data['x'])
        self.u = torch.from_numpy(data['u'])
        self.l = data['L']
        self.m = data['m']
        self.A_mix = data['A']
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
        self.one_hot_labels = one_hot_labels

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if not self.double:
            return self.x[index], self.u[index], self.s[index]
        else:
            indices = range(len(self))
            index2 = np.random.choice(indices)
            return self.x[index], self.x[index2], self.u[index], self.s[index]

    @staticmethod
    def load_tcl_data(root, nps, ns, dl, dd, nl, s, p, a, uncentered, noisy, centers, one_hot_labels):
        path_to_dataset = root + 'tcl_' + '_'.join(
            [str(nps), str(ns), str(dl), str(dd), str(nl), str(s), p, a])
        if uncentered:
            path_to_dataset += '_u'
        if noisy:
            path_to_dataset += '_noisy'
        if one_hot_labels:
            path_to_dataset += '_one_hot'
        path_to_dataset += '.npz'

        if not os.path.exists(path_to_dataset) or s is None:
            # if the path is not found or if the seed is not defined,
            # create a new dataset
            kwargs = {"n_per_seg": nps, "n_seg": ns, "d_sources": dl, "d_data": dd, "n_layers": nl, "prior": p,
                      "activation": a, "seed": s, "batch_size": 0, "uncentered": uncentered, "noisy": noisy,
                      "centers": centers, "repeat_linearity": True, "one_hot_labels": one_hot_labels}
            save_data(path_to_dataset, **kwargs)
        print('loading data from {}'.format(path_to_dataset))
        return np.load(path_to_dataset)

    def get_test_sample(self, batch_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.randint(max(0, self.len - batch_size))
        return self.x[idx:idx + batch_size], self.u[idx:idx + batch_size], self.s[idx:idx + batch_size]


class CustomSyntheticDataset(Dataset):
    def __init__(self, X, U, S=None, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(self.device)
        self.u = torch.from_numpy(U).to(self.device)
        if S is not None:
            self.s = torch.from_numpy(S).to(self.device)
        else:
            self.s = self.x
        self.len = self.x.shape[0]
        self.latent_dim = self.s.shape[1]
        self.aux_dim = self.u.shape[1]
        self.data_dim = self.x.shape[1]
        self.nps = int(self.len / self.aux_dim)
        print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.u[index], self.s[index]

    def get_metadata(self):
        return {'nps': self.nps,
                'ns': self.aux_dim,
                'n': self.len,
                'latent_dim': self.latent_dim,
                'data_dim': self.data_dim,
                'aux_dim': self.aux_dim,
                }


def create_if_not_exist_dataset(root='data/', nps=1000, ns=40, dl=2, dd=4, nl=3, s=1, p='gauss', a='xtanh',
                                uncentered=False, noisy=False, arg_str=None):
    """
    Create a dataset if it doesn't exist.
    This is useful as a setup step when running multiple jobs in parallel, to avoid having many scripts attempting
    to create the dataset when non-existent.
    This is called in `cmd_utils.create_dataset_before`
    """
    if arg_str is not None:
        # overwrites all other arg values
        # arg_str should be of this form: nps_ns_dl_dd_nl_s_p_a_u_n
        arg_list = arg_str.split('\n')[0].split('_')
        print(arg_list)
        assert len(arg_list) == 10
        nps, ns, dl, dd, nl = map(int, arg_list[0:5])
        p, a = arg_list[6:8]
        if arg_list[5] == 'n':
            s = None
        else:
            s = int(arg_list[5])
        if arg_list[-2] == 'f':
            uncentered = False
        else:
            uncentered = True
        if arg_list[-1] == 'f':
            noisy = False
        else:
            noisy = True

    path_to_dataset = root + 'tcl_' + '_'.join(
        [str(nps), str(ns), str(dl), str(dd), str(nl), str(s), p, a])
    if uncentered:
        path_to_dataset += '_u'
    if noisy:
        path_to_dataset += '_n'
    path_to_dataset += '.npz'

    if not os.path.exists(path_to_dataset) or s is None:
        kwargs = {"n_per_seg": nps, "n_seg": ns, "d_sources": dl, "d_data": dd, "n_layers": nl, "prior": p,
                  "activation": a, "seed": s, "batch_size": 0, "uncentered": uncentered, "noisy": noisy}
        save_data(path_to_dataset, **kwargs)
    return path_to_dataset
