import argparse

import numpy as np

from lib.data import generate_data
from lib.metrics import mean_corr_coef as mcc
from lib.wrappers import IVAE_wrapper, VAE_wrapper

LOG_FOLDER = 'log/vae/'
TORCH_CHECKPOINT_FOLDER = 'ckpt/vae/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--nps', type=int, default=500, help='number of points per segment')
    parser.add_argument('--ns', type=int, default=10, help='number of segments')
    parser.add_argument('-m', '--mlayers', type=int, default=3, help='number of mixing layers')
    parser.add_argument('--dseed', type=int, default=1, help='data_seed ')
    parser.add_argument('-d', '--ldim', type=int, default=2, help='dimension of latent space')
    parser.add_argument('--ddim', type=int, default=None, help='dimension of observation space')

    parser.add_argument('--method', type=str, default='ivae', help='method: vae or ivae')

    parser.add_argument('-i', '--maxsteps', type=int, default=1e3, help='max steps for gradient updates')
    parser.add_argument('-n', '--nlayers', type=int, default=3, help='depth (n_layers) of the networks (default 3)')
    parser.add_argument('-g', '--hdim', type=int, default=20, help='hidden dim of the networks (default 50)')
    parser.add_argument('-b', '--bsize', type=int, default=64, help='batch size (default 64)')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate (default 1e-3)')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed (default 1)')
    parser.add_argument('-z', '--latent-dim', type=int, default=None,
                        help='latent dimension. If None, equals the latent dim of the dataset. (default None)')

    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='train on gpu')
    parser.add_argument('-a', '--anneal', action='store_true', default=False, help='use annealing in learning')
    parser.add_argument('-n', '--nolog', action='store_true', default=False, help='run without logging')
    args = parser.parse_args()

    if args.ddim is None:
        args.ddim = args.ldim
    if args.latent_dim is None:
        args.latent_dim = args.ldim

    data_seed = args.dseed
    ldim = args.ldim
    ddim = args.ddim
    nps = args.nps
    ns = args.ns
    mlayers = args.mlayers

    method = args.method
    steps = int(args.maxsteps)
    nlayers = args.nlayers
    hidden_dim = args.hdim
    batch_size = args.bsize
    lr = args.lr
    seed = args.seed
    idim = args.latent_dim  # inference dimension

    cuda = args.cuda
    anneal = args.anneal

    if args.nolog:
        LOG_FOLDER = None
        CKPT_FOLDER = None

    S, X, U, _, _ = generate_data(nps, ns, ldim, ddim, n_layers=mlayers, seed=data_seed, slope=.2, dtype=np.float32)

    print('seed:', seed, '\tsteps:', steps, '\tmethod:', method, '\tldim:', ldim, '\tddim', ddim)

    if method == 'vae':
        z_vae, vae, params, logger = VAE_wrapper(X, S, lr=lr, n_layers=nlayers, batch_size=batch_size,
                                                 cuda=cuda, max_iter=steps, seed=seed, hidden_dim=hidden_dim,
                                                 log_folder=LOG_FOLDER, ckpt_folder=TORCH_CHECKPOINT_FOLDER,
                                                 inference_dim=idim)
        perf = mcc(z_vae.detach().cpu().numpy(), S)
        print('perf:', perf)

    elif method == 'ivae':
        z_ivae, ivae, params, logger = IVAE_wrapper(X, U, S, lr=lr, n_layers=nlayers, batch_size=batch_size,
                                                    cuda=cuda, max_iter=steps, seed=seed, hidden_dim=hidden_dim,
                                                    log_folder=LOG_FOLDER, ckpt_folder=TORCH_CHECKPOINT_FOLDER,
                                                    inference_dim=idim, anneal=anneal)
        perf = mcc(z_ivae.detach().cpu().numpy(), S)
        print('perf:', perf)

    else:
        raise ValueError('wrong method')

    logger.add_metadata(full_perf=perf)
    logger.add_metadata(method=method, cuda=cuda, max_steps=steps, lr=lr, seed=seed, idim=idim,
                        n_layers=nlayers, batch_size=batch_size, hidden_dim=hidden_dim, anneal=anneal)
    logger.add_metadata(data_seed=data_seed, nps=nps, ns=ns, ld=ldim, dd=ddim, mixing_layers=mlayers)

    logger.save_to_npz(log_dir=LOG_FOLDER)
    logger.save_to_json(log_dir=LOG_FOLDER)
