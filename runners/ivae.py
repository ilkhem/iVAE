import time

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from data.datasets import SyntheticDataset
from losses.losses import elbo_decomposed, elbo_decomposed_vae, get_performance, permute_dims
from models.models import cleanVAEICA, cleanVAE, Discriminator


def runner(args, config):
    st = time.time()

    torch.manual_seed(args.seed)
    print('Executing script on: {}\n'.format(config.device))

    factor = config.gamma > 0

    dset = SyntheticDataset(args.data_path, config.nps, config.ns, config.dl, config.dd, config.nl, config.s, config.p,
                            config.act, uncentered=config.uncentered, noisy=config.noisy, double=factor)
    d_data, d_latent, d_aux = dset.get_dims()

    loader_params = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_loader = DataLoader(dset, batch_size=config.batch_size, shuffle=True, drop_last=True, **loader_params)

    if config.ica:
        model = cleanVAEICA(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim,
                            n_layers=config.n_layers, activation=config.activation, slope=.1).to(config.device)
    else:
        model = cleanVAE(data_dim=d_data, latent_dim=d_latent, hidden_dim=config.hidden_dim,
                         n_layers=config.n_layers, activation=config.activation, slope=.1).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)

    if factor:
        D = Discriminator(d_latent).to(config.device)
        optim_D = optim.Adam(D.parameters(), lr=config.lr,
                             betas=(.5, .9))

    loss_hist = []
    perf_hist = []
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
            if config.ica:
                f, g, v, s, l = model(x, u)
                loss, t1, t2, t3, t4 = elbo_decomposed(x, f, g, v, s, l, len(dset), a=a, b=b, c=c, d=d, detailed=True)
            else:
                f, g, v, s = model(x)
                loss, t1, t2, t3, t4 = elbo_decomposed_vae(x, f, g, v, s, len(dset), a=a, b=b, c=c, d=d, detailed=True)
                if factor:
                    D_z = D(s)
                    vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
                    loss += config.gamma * vae_tc_loss

            loss.backward(retain_graph=factor)

            train_loss += loss.item()
            try:
                perf = get_performance(s_true.numpy(), s.cpu().detach().numpy())
            except:
                perf = 0
            train_perf += perf

            optimizer.step()

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
        print('==> Epoch {}/{}:\ttrain loss: {:.6f}\ttrain perf: {:.6f}'.format(epoch, config.epochs, train_loss,
                                                                                train_perf))

        if not config.no_scheduler:
            scheduler.step(train_loss)
    print('\ntotal runtime: {}'.format(time.time() - st))

    # evaluate perf on full dataset
    Xt, Ut, St = dset.x.to(config.device), dset.u.to(config.device), dset.s
    if config.ica:
        _, _, _, s, _ = model(Xt, Ut)
    else:
        _, _, _, s = model(Xt)
    full_perf = get_performance(dset.s.numpy(), s.cpu().detach().numpy())
    return loss_hist, perf_hist, full_perf
