import numpy as np
import tensorflow as tf
from lib.data import generate_data, to_one_hot
from lib.metrics import mean_corr_coef as mcc
from lib.wrappers import IVAE_wrapper, TCL_wrapper

if __name__ == '__main__':

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('seed', 1, 'seed')
    flags.DEFINE_float('steps', 5e3, 'max_steps')
    flags.DEFINE_string('method', 'tcl', 'method for ICA')
    flags.DEFINE_integer('dim', 2, 'dimension of data')

    seed = FLAGS.seed
    steps = int(FLAGS.steps)
    method = FLAGS.method
    dim = FLAGS.dim

    data_seed = 1
    d = dim
    nps = 500
    ns = 25
    S, X, U = generate_data(nps, ns, d, n_layers=3, seed=data_seed, slope=.2, staircase=True)
    Uh = to_one_hot(U)[0]

    print('seed:', seed, '\tsteps:', steps, '\tmethod:', method, '\tdim:', dim)

    if method == 'tcl':
        z_tcl, z_tcl_ica, acc = TCL_wrapper(X.T, U, [2 * d, 2 * d, d], random_seed=seed, max_steps=steps,
                                            max_steps_init=steps)
        print('acc:', acc)
        perf = mcc(z_tcl.T, S ** 2)
        print('perf:', perf)

    elif method == 'ivae':
        z_ivae, ivae, params = IVAE_wrapper(X.astype(np.float32), Uh.astype(np.float32), lr=1e-3, n_layers=4,
                                            batch_size=64,
                                            cuda=False, max_iter=steps, seed=seed, hidden_dim=20)
        perf = mcc(z_ivae.detach().cpu().numpy(), S)
        print('perf:', perf)
