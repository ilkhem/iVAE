import argparse
import os
import pickle
import sys

import numpy as np
import torch
import yaml

from runners import ivae_runner, tcl_runner

def parse():
    '''
    Config arguments:
    Dataset descriptors
    nps: (int) number of points per segment (n_per_seg)
    ns: (int) number of segments (n_seg)
    dl: (int) dimension of latent sources (d_sources)
    dd: (int) d_data (dimension of the mixed data)
    nl: (int) number of layers for ICA mixing
    s: (int) seed for the data generation only
    -- seed is used to set the model seed.
    
    p: (str) probability distribution (e.g. 'gauss' for Normal, 'lap' for Laplace, 'hs' for Hypersecant)
    act: (str) activation function for the mixing transformation (e.g. 'none', 'lrelu', 'sigmoid')
    uncentered: (bool) if True, different distributions have different means
    noisy: (bool) if True, add noise to the observations
    staircase: (bool)
    
    # model args
    n_layers: (int) number of layers in the MLP
    hidden_dim: (int) number of dimensions in each hidden layer
    activation: (str) activation function of the MLP (e.g. 'lrelu', 'none', 'sigmoid')
    ica: (bool) if True, run the iVAE. If False, run the VAE
    initialize: (bool) weight initialization
    batch_norm: (bool) batch normalization
    tcl: (bool) if True, run TCL. If False, run the iVAE
    
    # learning
    a: (int) weight of the logpx term of the ELBO
    b: (int) weight of the (logqs_cux - logqs) term of the ELBO
    c: (int) weight of the (logqs - logqs_i) term of the ELBO
    d: (int) weight of the (logqs_i - logps_cu) term of the ELBO
    gamma: (int) ? used for TCL?
    lr: (float) learning rate
    batch_size: (int) batch size
    epochs: (int) total number of epochs
    no_scheduler: (bool) if False, use a scheduler for the optimizer
    scheduler_tol: (int) scheduler tolerance
    anneal: (bool) annealing
    anneal_epoch: (int)
    
    # more configs
    shuffle: (bool) if True, shuffle data from the trainig batch
    one_hot: (bool) if True, one-hot encode the segments U
    checkpoint: (bool) if True, save the weights and meta-data in every epoch
    log: (bool) if True, save logs of the experiment. Does not work properly yet, use False!
    simple_mixing: if True, have elements of mixing matrix from a Uniform distribution \
    and skip all the other mixing code
    '''
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--config', type=str, default='ivae.yaml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='', help='A string for documentation purpose')

    parser.add_argument('--n-sims', type=int, default=1, help='Number of simulations to run')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    return parser.parse_args()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def make_dirs(args):
    os.makedirs(args.run, exist_ok=True)
    args.log = os.path.join(args.run, 'logs', args.doc)
    os.makedirs(args.log, exist_ok=True)
    args.checkpoints = os.path.join(args.run, 'checkpoints', args.doc)
    os.makedirs(args.checkpoints, exist_ok=True)
    args.data_path = os.path.join(args.run, 'datasets', args.doc)
    os.makedirs(args.data_path, exist_ok=True)


def main():
    args = parse()
    make_dirs(args)

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    new_config = dict2namespace(config)
    new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(new_config)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if new_config.tcl:
        r = tcl_runner(args, new_config)
    else:
        r = ivae_runner(args, new_config)
    # r = clean_vae_runner(args, new_config)
    fname = os.path.join(args.run,
                         '_'.join([os.path.splitext(args.config)[0], str(args.seed), str(args.n_sims)]) + '.p')
    pickle.dump(r, open(fname, "wb"))


if __name__ == '__main__':
    sys.exit(main())
