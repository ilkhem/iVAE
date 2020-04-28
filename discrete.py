import numpy as np
import torch
from torch import optim

from data import generate_nonstationary_sources, generate_mixing_matrix, to_one_hot
from metrics import mean_corr_coef as mcc
from models import DiscreteIVAE, DiscreteVAE


def sigmoid(x):
    """
    Sigmoid activation function
    @param x: input array
    @return:
        out: output array
    """
    return 1 / (1 + np.exp(-x))


def run_ivae(S, X, U, Sb, Xb, Ub, epochs=20, seed=None, n_layers=2, hidden_dim=20, lr=1e-2, device='cpu'):
    print('starting ica')
    if seed is not None:
        torch.manual_seed(seed)
    dl = Sb[0].shape[1]
    dd = Xb[0].shape[1]
    ns = Ub[0].shape[1]
    model = DiscreteIVAE(dl, dd, ns, activation='none', n_layers=n_layers, hidden_dim=hidden_dim, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, verbose=True)

    epoch_loss_hist = []
    epoch_perf_hist = []

    # training loop
    it = 0
    n_batches = len(Xb)
    n_iter = epochs * n_batches
    model.train()
    while it < n_iter:
        elbo_mean = 0
        perf_mean = 0
        for c in range(n_batches):
            it += 1
            optimizer.zero_grad()

            x = torch.Tensor(Xb[c]).to(device)
            u = torch.Tensor(Ub[c]).to(device)
            z = Sb[c]

            elbo, z_est = model.elbo(x, u)
            elbo.mul(-1).backward()
            optimizer.step()
            elbo_mean += elbo.item()
            perf = mcc(z, z_est.cpu().detach().numpy())
            perf_mean += perf
        elbo_mean /= n_batches
        epoch_loss_hist.append(elbo_mean)
        perf_mean /= n_batches
        epoch_perf_hist.append(perf_mean)
        scheduler.step(elbo_mean)
        print('epoch {}:\tloss: {};\tperf: {}'.format(int(it / n_batches), elbo_mean, perf_mean))

    _, _, Z, _ = model(torch.Tensor(X).to(device), torch.Tensor(U).to(device))
    perf = mcc(Z.detach().cpu().numpy(), S)
    print(perf)

    with open('log/discrete/ica_discrete.txt', 'a') as f:
        f.write(str(perf) + '\n')

    np.savez_compressed('log/discrete/ica_{}.npz'.format(args.seed),
                        l=np.array(epoch_loss_hist), p=np.array(epoch_perf_hist))


def run_vae(S, X, Sb, Xb, epochs=20, seed=None, n_layers=2, hidden_dim=20, lr=1e-2, device='cpu'):
    print('starting vae')
    if seed is not None:
        torch.manual_seed(seed)
    dl = Sb[0].shape[1]
    dd = Xb[0].shape[1]
    model = DiscreteVAE(dl, dd, activation='none', n_layers=n_layers, hidden_dim=hidden_dim, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, verbose=True)

    epoch_loss_hist = []
    epoch_perf_hist = []

    # training loop
    it = 0
    n_batches = len(Xb)
    n_iter = epochs * n_batches
    model.train()
    while it < n_iter:
        elbo_mean = 0
        perf_mean = 0
        for c in range(n_batches):
            it += 1
            optimizer.zero_grad()

            x = torch.Tensor(Xb[c]).to(device)
            z = Sb[c]

            elbo, z_est = model.elbo(x)
            elbo.mul(-1).backward()
            optimizer.step()
            elbo_mean += elbo.item()
            perf = mcc(z, z_est.cpu().detach().numpy())
            perf_mean += perf
        elbo_mean /= n_batches
        epoch_loss_hist.append(elbo_mean)
        perf_mean /= n_batches
        epoch_perf_hist.append(perf_mean)
        scheduler.step(elbo_mean)
        print('epoch {}:\tloss: {};\tperf: {}'.format(int(it / n_batches), elbo_mean, perf_mean))

    _, _, Z = model(torch.Tensor(X).to(device))
    perf = mcc(Z.detach().cpu().numpy(), S)
    print(perf)

    with open('log/discrete/vae_discrete.txt', 'a') as f:
        f.write(str(perf) + '\n')

    np.savez_compressed('log/discrete/vae_{}.npz'.format(args.seed),
                        l=np.array(epoch_loss_hist), p=np.array(epoch_perf_hist))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='tcl vs vaeica on simulated data')
    parser.add_argument('-s', '--seed', type=int, default=1, dest='seed', help='random seed - default: 1')
    parser.add_argument('-m', '--method', type=str, default='ica', dest='method', help='method')
    args = parser.parse_args()

    print('start {} {}'.format(args.seed, args.method))

    np.random.seed(1)
    nps = 3000
    ns = 20
    batch_size = 200
    n = nps * ns
    dl, dd = 10, 100

    S, U, _, _ = generate_nonstationary_sources(nps, ns, dl)
    A = generate_mixing_matrix(dl, dd)
    PX = sigmoid(np.dot(S, A))
    # X =  (np.sign(PX-0.5) + 1)/2
    X = np.random.binomial(1, PX)

    idx = np.random.permutation(n)
    Xb, Sb, Ub = [], [], []
    n_batches = int(n / batch_size)
    for c in range(n_batches):
        Sb += [S[idx][c * batch_size:(c + 1) * batch_size]]
        Xb += [X[idx][c * batch_size:(c + 1) * batch_size]]
        Ub += [U[idx][c * batch_size:(c + 1) * batch_size]]
    Ub = to_one_hot(Ub, ns)
    U = to_one_hot(U)[0]

    if args.method == 'ica':
        run_ivae(S, X, U, Sb, Xb, Ub, epochs=50, seed=args.seed, hidden_dim=50, n_layers=2, lr=1e-2, device='cuda')

    elif args.method == 'vae':
        run_vae(S, X, Sb, Xb, epochs=50, seed=args.seed, hidden_dim=50, n_layers=2, lr=1e-2, device='cuda')

    else:
        raise ValueError('wrong method {}'.format(args.method))