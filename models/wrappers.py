import os

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data import CustomSyntheticDataset
from metrics import mean_corr_coef as mcc
from utils import Logger, checkpoint
from .nets import iVAE, DiscreteIVAE, VAE, DiscreteVAE


def IVAE_wrapper(X, U, S=None, batch_size=256, max_iter=7e4, seed=None, n_layers=3, hidden_dim=200, lr=1e-2, cuda=True,
                 activation='lrelu', slope=.1, discrete=False, inference_dim=None,
                 anneal=False, log_folder=None, ckpt_folder=None, scheduler_tol=3):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device('cuda' if cuda else 'cpu')
    print('training on {}'.format(torch.cuda.get_device_name(device) if cuda else 'cpu'))

    # load data
    print('Creating shuffled dataset..')
    dset = CustomSyntheticDataset(X, U, S, 'cpu')
    loader_params = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(dset, shuffle=True, batch_size=batch_size, **loader_params)
    data_dim, latent_dim, aux_dim = dset.get_dims()
    if inference_dim is not None:
        latent_dim = inference_dim
    N = len(dset)
    max_epochs = int(max_iter // len(train_loader) + 1)

    # define model and optimizer
    print('Defining model and optimizer..')
    if not discrete:
        model = iVAE(latent_dim, data_dim, aux_dim, activation=activation, device=device,
                     n_layers=n_layers, hidden_dim=hidden_dim, slope=slope, anneal=anneal)
    else:
        model = DiscreteIVAE(latent_dim, data_dim, aux_dim, activation=activation,
                             n_layers=n_layers, hidden_dim=hidden_dim, device=device, slope=slope)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=scheduler_tol, verbose=True)

    logger = Logger(log_dir=log_folder)
    exp_id = logger.exp_id
    if log_folder is None:
        ckpt_folder = None
    logger.add('elbo')
    logger.add('perf')

    # training loop
    print("Training..")
    it = 0
    model.train()
    while it < max_iter:
        epoch = it // len(train_loader) + 1
        for _, (x, u, z) in enumerate(train_loader):
            it += 1
            if anneal:
                model.anneal(N, max_iter, it)
            optimizer.zero_grad()

            x, u = x.to(device), u.to(device)

            elbo, z_est = model.elbo(x, u)
            elbo.mul(-1).backward()
            optimizer.step()

            logger.update('elbo', -elbo.item())

            if S is not None:
                perf = mcc(z_est.cpu().detach().numpy(), z.cpu().numpy())
                logger.update('perf', perf)

            if it % int(max_iter / 5) == 0 and ckpt_folder is not None:
                checkpoint(ckpt_folder, exp_id, it, model, optimizer,
                           logger.get_last('elbo'), logger.get_last('perf'))

        logger.log()
        scheduler.step(logger.get_last('elbo'))

        if S is not None:
            print('epoch {}/{} \tloss: {}\tperf: {}'.format(epoch, max_epochs, logger.get_last('elbo'),
                                                            logger.get_last('perf')))
        else:
            print('epoch {}/{} \tloss: {}'.format(epoch, max_epochs, logger.get_last('elbo')))

    Xt, Ut = dset.x.to(device), dset.u.to(device)
    decoder_params, encoder_params, z, prior_params = model(Xt, Ut)
    params = {'decoder': decoder_params, 'encoder': encoder_params, 'prior': prior_params}

    return z, model, params, logger


def VAE_wrapper(X, S=None, batch_size=256, max_iter=7e4, seed=None, n_layers=3, hidden_dim=200, lr=1e-2, cuda=True,
                activation='lrelu', slope=.1, discrete=False, inference_dim=None,
                log_folder=None, ckpt_folder=None, scheduler_tol=3):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device('cuda' if cuda else 'cpu')
    print('training on {}'.format(torch.cuda.get_device_name(device) if cuda else 'cpu'))

    # load data
    print('Creating shuffled dataset..')
    dset = CustomSyntheticDataset(X, X, S, 'cpu')
    loader_params = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(dset, shuffle=True, batch_size=batch_size, **loader_params)
    data_dim, latent_dim, aux_dim = dset.get_dims()
    if inference_dim is not None:
        latent_dim = inference_dim
    max_epochs = int(max_iter // len(train_loader) + 1)

    # define model and optimizer
    print('Defining model and optimizer..')
    if not discrete:
        model = VAE(latent_dim, data_dim, activation=activation, device=device,
                    n_layers=n_layers, hidden_dim=hidden_dim, slope=slope)
    else:
        model = DiscreteVAE(latent_dim, data_dim, activation=activation,
                            n_layers=n_layers, hidden_dim=hidden_dim, device=device, slope=slope)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=scheduler_tol, verbose=True)

    logger = Logger(log_dir=log_folder)
    exp_id = logger.exp_id
    if log_folder is None:
        ckpt_folder = None
    logger.add('elbo')
    logger.add('perf')

    # training loop
    print("Training..")
    it = 0
    model.train()
    while it < max_iter:
        epoch = it // len(train_loader) + 1
        for _, (x, _, z) in enumerate(train_loader):
            it += 1
            optimizer.zero_grad()

            x = x.to(device)

            elbo, z_est = model.elbo(x)
            elbo.mul(-1).backward()
            optimizer.step()

            logger.update('elbo', -elbo.item())

            if S is not None:
                perf = mcc(z_est.cpu().detach().numpy(), z.cpu().numpy())
                logger.update('perf', perf)

            if it % int(max_iter / 5) == 0 and ckpt_folder is not None:
                checkpoint(ckpt_folder, exp_id, it, model, optimizer,
                           logger.get_last('elbo'), logger.get_last('perf'))

        logger.log()
        scheduler.step(logger.get_last('elbo'))

        if S is not None:
            print('epoch {}/{} \tloss: {}\tperf: {}'.format(epoch, max_epochs, logger.get_last('elbo'),
                                                            logger.get_last('perf')))
        else:
            print('epoch {}/{} \tloss: {}'.format(epoch, max_epochs, logger.get_last('elbo')))

    Xt = dset.x.to(device)
    decoder_params, encoder_params, z, prior_params = model(Xt)
    params = {'decoder': decoder_params, 'encoder': encoder_params, 'prior': prior_params}

    return z, model, params, logger


def TCL_wrapper(sensor, label, list_hidden_nodes, random_seed=0, max_steps=int(7e4), max_steps_init=int(7e4),
                cuda=False, batch_size=512, initial_learning_rate=0.01):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from .tcl import tcl_core, tcl_eval
    from .tcl.tcl_preprocessing import pca
    from .tcl.tcl_core import train_cpu, train_gpu
    from sklearn.decomposition import FastICA

    # Training ----------------------------------------------------
    momentum = 0.9  # momentum parameter of SGD
    decay_steps = int(5e4)  # decay steps (tf.train.exponential_decay)
    decay_factor = 0.1  # decay factor (tf.train.exponential_decay)
    moving_average_decay = 0.9999  # moving average decay of variables to be saved
    checkpoint_steps = 1e5  # interval to save checkpoint
    num_comp = sensor.shape[0]

    # for MLR initialization
    decay_steps_init = int(5e4)  # decay steps for initializing only MLR

    # Other -------------------------------------------------------
    # # Note: save folder must be under ./storage
    train_dir = '../storage/temp5'  # save directory

    num_segment = len(np.unique(label))

    # Preprocessing -----------------------------------------------
    sensor, pca_parm = pca(sensor, num_comp=num_comp)

    # Train model (only MLR) --------------------------------------
    train = train_gpu if cuda else train_cpu
    train(sensor,
          label,
          num_class=len(np.unique(label)),  # num_segment,
          list_hidden_nodes=list_hidden_nodes,
          initial_learning_rate=initial_learning_rate,
          momentum=momentum,
          max_steps=max_steps_init,  # For init
          decay_steps=decay_steps_init,  # For init
          decay_factor=decay_factor,
          batch_size=batch_size,
          train_dir=train_dir,
          checkpoint_steps=checkpoint_steps,
          moving_average_decay=moving_average_decay,
          MLP_trainable=False,  # For init
          save_file='model_init.ckpt',  # For init
          random_seed=random_seed)

    init_model_path = os.path.join(train_dir, 'model_init.ckpt')

    # Train model -------------------------------------------------
    train(sensor,
          label,
          num_class=len(np.unique(label)),  # num_segment,
          list_hidden_nodes=list_hidden_nodes,
          initial_learning_rate=initial_learning_rate,
          momentum=momentum,
          max_steps=max_steps,
          decay_steps=decay_steps,
          decay_factor=decay_factor,
          batch_size=batch_size,
          train_dir=train_dir,
          checkpoint_steps=checkpoint_steps,
          moving_average_decay=moving_average_decay,
          load_file=init_model_path,
          random_seed=random_seed)

    # now that we have trained everything, we can evaluate results:
    apply_fast_ica = True
    eval_dir = '../storage/temp5'
    ckpt = tf.train.get_checkpoint_state(eval_dir)

    with tf.Graph().as_default():
        data_holder = tf.placeholder(tf.float32, shape=[None, sensor.shape[0]], name='data')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, feats = tcl_core.inference(data_holder, list_hidden_nodes, num_class=num_segment)

        # Calculate predictions.
        top_value, preds = tf.nn.top_k(logits, k=1, name='preds')

        # Restore the moving averaged version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)

            tensor_val = tcl_eval.get_tensor(sensor, [preds, feats], sess, data_holder, batch=256)
            pred_val = tensor_val[0].reshape(-1)
            feat_val = tensor_val[1]

    # Calculate accuracy ------------------------------------------
    accuracy, confmat = tcl_eval.calc_accuracy(pred_val, label)

    # Apply fastICA -----------------------------------------------
    if apply_fast_ica:
        ica = FastICA(random_state=random_seed)
        feat_val_ica = ica.fit_transform(feat_val)
        feateval_ica = feat_val_ica.T  # Estimated feature
    else:
        feateval_ica = None

    featval = feat_val.T

    return featval, feateval_ica, accuracy
