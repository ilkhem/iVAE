import numpy as np
import tensorflow as tf

from lib.data import generate_data, to_one_hot
from lib.metrics import mean_corr_coef as mcc
from lib.utils import Logger
from lib.wrappers import IVAE_wrapper, TCL_wrapper

LOG_FOLDER = 'log/btcl/'
TORCH_CHECKPOINT_FOLDER = 'ckpt/btcl/'

if __name__ == '__main__':

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('seed', 1, 'seed')
    flags.DEFINE_float('maxsteps', 1e3, 'max_steps')
    flags.DEFINE_string('method', 'ivae', 'method for ICA')
    flags.DEFINE_integer('dim', 2, 'dimension of data')
    flags.DEFINE_bool('cuda', False, 'train on gpu?')
    flags.DEFINE_integer('nps', 500, 'number of points per segment')
    flags.DEFINE_integer('ns', 10, 'number of segments')
    flags.DEFINE_integer('mlayers', 3, 'number of mixing layers')
    flags.DEFINE_integer('dseed', 1, 'data_seed ')
    flags.DEFINE_float('lr', 1e-3, 'learning rate')
    flags.DEFINE_integer('nlayers', 3, 'number of estimation network layers')
    flags.DEFINE_integer('bsize', 64, 'batch size')
    flags.DEFINE_integer('hdim', 20, 'size of hidden dim')
    flags.DEFINE_bool('staircase', False, 'staircase data?')

    data_seed = FLAGS.dseed
    dim = FLAGS.dim
    nps = FLAGS.nps
    ns = FLAGS.ns
    mlayers = FLAGS.mlayers

    method = FLAGS.method
    seed = FLAGS.seed
    steps = int(FLAGS.maxsteps)
    lr = FLAGS.lr
    nlayers = FLAGS.nlayers
    batch_size = FLAGS.bsize
    hidden_dim = FLAGS.hdim
    cuda = FLAGS.cuda
    staircase = FLAGS.staircase

    if staircase:
        LOG_FOLDER = 'log/btcl/'
        TORCH_CHECKPOINT_FOLDER = 'ckpt/btcl/'
    else:
        LOG_FOLDER = 'log/normal_tcl/'
        TORCH_CHECKPOINT_FOLDER = 'ckpt/normal_tcl/'

    S, X, U, _, _ = generate_data(nps, ns, dim, n_layers=mlayers, seed=data_seed, slope=.2, staircase=staircase,
                                  dtype=np.float32, one_hot_labels=False)
    Uh = to_one_hot([U])[0]

    print('seed:', seed, '\tsteps:', steps, '\tmethod:', method, '\tdim:', dim, '\tstaircase', staircase)

    if method == 'tcl':
        z_tcl, z_tcl_ica, acc = TCL_wrapper(X.T, U, [2 * dim, 2 * dim, dim], random_seed=seed, max_steps=steps,
                                            max_steps_init=steps, cuda=cuda, batch_size=batch_size)
        print('acc:', acc)
        perf = mcc(z_tcl.T, S ** 2)
        print('perf:', perf)
        logger = Logger(logdir=LOG_FOLDER)
        logger.add('elbo')
        logger.update('elbo', acc)
        logger.add('perf')
        logger.log()

    elif method == 'ivae':
        z_ivae, ivae, params, logger = IVAE_wrapper(X, Uh, S, lr=lr, n_layers=nlayers, batch_size=batch_size,
                                                    cuda=cuda, max_iter=steps, seed=seed, hidden_dim=hidden_dim,
                                                    log_folder=LOG_FOLDER, ckpt_folder=TORCH_CHECKPOINT_FOLDER)
        perf = mcc(z_ivae.detach().cpu().numpy(), S)
        print('perf:', perf)

    else:
        raise ValueError('wrong method')

    logger.add_metadata(full_perf=perf)
    logger.add_metadata(method=method, cuda=cuda, max_steps=steps, lr=lr, seed=seed,
                        n_layers=nlayers, batch_size=batch_size, hidden_dim=hidden_dim)
    logger.add_metadata(data_seed=data_seed, nps=nps, ns=ns, d=dim, mixing_layers=mlayers)

    logger.save_to_npz()
    logger.save_to_json()
