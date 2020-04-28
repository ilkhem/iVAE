import numpy as np
import torch

from data import SyntheticDataset
from metrics import mean_corr_coef as mcc
from models import TCL_wrapper


def runner(args, config):
    torch.manual_seed(args.seed)
    print('Executing script on: {}\n'.format(config.device))

    dset = SyntheticDataset(args.data_path, config.nps, config.ns, config.dl, config.dd, config.nl, config.s, config.p,
                            config.act, uncentered=config.uncentered, noisy=config.noisy, double=False,
                            one_hot_labels=False)
    X, U, S = dset.x.numpy(), dset.u.numpy(), dset.s.numpy()

    perfs = []
    perfs_no_ica = []
    perfs_ica = []

    for seed in range(args.seed, args.seed + args.n_sims):
        dim = config.dd
        cuda = str(config.device) != 'cpu'
        z_tcl, z_tcl_ica, acc = TCL_wrapper(X.T, U, [2 * dim, 2 * dim, dim], random_seed=seed, max_steps=config.steps,
                                            max_steps_init=config.steps, cuda=cuda, batch_size=config.batch_size)
        # print('acc:', acc)
        perf_ica = mcc(z_tcl_ica.T, S ** 2)
        perf_no_ica = mcc(z_tcl.T, S ** 2)
        perf = np.max(perf_ica, perf_no_ica)
        print('MCC: {}\tlinear ICA: {}'.format(perf, perf_ica > perf_no_ica))
        perfs.append(perf)
        perfs_ica.append(perf_ica)
        perfs_no_ica.append(perf_no_ica)

    return perfs, perfs_no_ica, perfs_ica
