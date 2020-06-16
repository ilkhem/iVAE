import time

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from data import SyntheticDataset
from metrics import mean_corr_coef as mcc
from models import cleanIVAE, cleanVAE, Discriminator, permute_dims

from utils import Logger, checkpoint


def runner(args, config):
    st = time.time()

    print('Executing script on: {}\n'.format(config.device))

    factor = config.gamma > 0

    dset = SyntheticDataset(args.data_path, config.nps, config.ns, config.dl, config.dd, config.nl, config.s, config.p,
                            config.act, uncentered=config.uncentered, noisy=config.noisy, double=factor, one_hot_labels=config.one_hot, simple_mixing=config.simple_mixing)
    d_data, d_latent, d_aux = dset.get_dims()

    loader_params = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_loader = DataLoader(dset, batch_size=config.batch_size, shuffle=config.shuffle, drop_last=True, **loader_params)

#     perfs = []
    loss_hists = []
    perf_hists = []
    all_perf_hists = []
    
    # to do: pass the paths as parameters from main.py
    if config.checkpoint:
        dir_log='run/logs/'
        ckpt_folder='run/checkpoints/'

        logger = Logger(log_dir=dir_log)
        exp_id = logger.exp_id

        if (config.log):
            logger.add('elbo')
            logger.add('mcc')
    

    for seed in range(args.seed, args.seed + args.n_sims):
        if config.ica:
            model = cleanIVAE(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim,
                              n_layers=config.n_layers, activation=config.activation, slope=.1).to(config.device)
        else:
            model = cleanVAE(data_dim=d_data, latent_dim=d_latent, hidden_dim=config.hidden_dim,
                             n_layers=config.n_layers, activation=config.activation, slope=.1).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        
        if not config.no_scheduler:
            # only initialize the scheduler if no_scheduler==False
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0, verbose=True)

        if factor:
            D = Discriminator(d_latent).to(config.device)
            optim_D = optim.Adam(D.parameters(), lr=config.lr,
                                 betas=(.5, .9))

        loss_hist = []
        perf_hist = []
        all_mccs = []
        
        # load all training data for eventual evaluation
        Xt, Ut, St = dset.x.to(config.device), dset.u.to(config.device), dset.s
        
        for epoch in range(1, config.epochs + 1):
            model.train()

            if config.anneal:
                a = config.a
                d = config.d
                b = config.b
                c = 0
                if epoch > config.epochs / 1.6:
                    b = 1
                    c = 1
                    d = 1
                    a = 2 * config.a
            else:
                a = config.a
                b = config.b
                c = config.c
                d = config.d

            train_loss = 0
            train_perf = 0
            for i, data in enumerate(data_loader):
                if not factor:
                    x, u, s_true = data
                else:
                    x, x2, u, s_true = data
                
                x, u = x.to(config.device), u.to(config.device)
                optimizer.zero_grad()
                loss, z = model.elbo(x, u, len(dset), a=a, b=b, c=c, d=d)
                
                if factor:
                    D_z = D(z)
                    vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
                    loss += config.gamma * vae_tc_loss

                loss.backward(retain_graph=factor)

                # batch loss
                train_loss += loss.item()
                
                # batch performance
                try:
                    perf = mcc(s_true.numpy(), z.cpu().detach().numpy())
                except:
                    perf = 0
                train_perf += perf

                optimizer.step()
                
                if config.log:
                    logger.update('elbo', train_loss)

                if factor:
                    ones = torch.ones(config.batch_size, dtype=torch.long, device=config.device)
                    zeros = torch.zeros(config.batch_size, dtype=torch.long, device=config.device)
                    x_true2 = x2.to(config.device)
                    _, _, _, z_prime = model(x_true2)
                    z_pperm = permute_dims(z_prime).detach()
                    D_z_pperm = D(z_pperm)
                    D_tc_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                    optim_D.zero_grad()
                    D_tc_loss.backward()
                    optim_D.step()

            train_perf /= len(data_loader)
            perf_hist.append(train_perf)
            train_loss /= len(data_loader)
            loss_hist.append(train_loss)
            
            if config.ica:
                _, _, _, s, _ = model(Xt, Ut)
            else:
                _, _, _, s = model(Xt)
            
            # MCC score on the whole dataset (as opposed to just the training batch)
            try:
                perf_all = mcc(dset.s.numpy(), s.cpu().detach().numpy())
            except:
                perf_all = 0
                
            all_mccs.append(perf_all)    
            print('==> Epoch {}/{}:\t train loss: {:.6f}\t train perf: {:.6f} \t full perf: {:,.6f}'.format(epoch, config.epochs, train_loss, train_perf, perf_all))
            
            if config.checkpoint:
                # save checkpoints (weights, loss, performance, meta-data) after every epoch
                checkpoint(ckpt_folder, exp_id, seed, epoch, model, optimizer,
                           train_loss, train_perf, perf_all)
            
            if (config.log):
                logger.log()

            if not config.no_scheduler:
                scheduler.step(train_loss)
                
        ttime_s = time.time() - st
        print('\ntotal runtime: {} seconds'.format(ttime_s))
        print('\ntotal runtime: {} minutes'.format(ttime_s/60))

        all_perf_hists.append(all_mccs)
        loss_hists.append(loss_hist)
        perf_hists.append(perf_hist)
        
        if config.log:
            logger.add_metadata(method='ivae', lr=config.lr, seed=config.s, idim=d_latent,
                                    n_layers=config.n_layers, batch_size=config.batch_size, hidden_dim=config.hidden_dim, anneal=config.anneal)
            logger.add_metadata(data_seed=config.s, nps=config.nps, ns=config.ns, ld=d_sources, dd=d_data, mixing_layers=config.nl)
            exp_id = logger.exp_id
            logger.save_to_npz(log_dir=dir_log)

    return all_perf_hists, loss_hists, perf_hists
