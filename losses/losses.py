from numbers import Number

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


# losses

def _check_inputs(size, mu, v):
    """helper function to ensure inputs are compatible"""
    if size is None and mu is None and v is None:
        raise ValueError("inputs can't all be None")
    elif size is not None:
        if mu is None:
            mu = torch.Tensor([0])
        if v is None:
            v = torch.Tensor([1])
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        v = v.expand(size)
        mu = mu.expand(size)
        return mu, v
    elif mu is not None and v is not None:
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        if v.size() != mu.size():
            v = v.expand(mu.size())
        return mu, v
    elif mu is not None:
        v = torch.Tensor([1]).type_as(mu).expand(mu.size())
        return mu, v
    elif v is not None:
        mu = torch.Tensor([0]).type_as(v).expand(v.size())
        return mu, v
    else:
        raise ValueError('Given invalid inputs: size={}, mu_logsigma={})'.format(size, (mu, v)))


def _batch_slogdet(cov_batch):
    """
    compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
    this implementation is just a for loop, but that is what's suggested in torch forums
    """
    batch_size = cov_batch.size(0)
    signs = torch.empty(batch_size, requires_grad=False)
    logabsdets = torch.empty(batch_size, requires_grad=False)
    for i, cov in enumerate(cov_batch):
        signs[i], logabsdets[i] = torch.slogdet(cov)
    return signs, logabsdets


def log_normal(x, mu=None, v=None, broadcast_size=False):
    """compute the log-pdf of a normal distribution with diagonal covariance"""
    if not broadcast_size:
        mu, v = _check_inputs(None, mu, v)
    else:
        mu, v = _check_inputs(x.size(), mu, v)
    assert mu.shape == v.shape
    return -0.5 * (np.log(2 * np.pi) + v.log() + (x - mu).pow(2).div(v))


def log_normal_full(x, mu, v):
    """
    compute the log-pdf of a normal distribution with full covariance
    v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
    mu is batch of means of shape (batch_size, d_latent)
    """
    batch_size, d_latent = mu.size()
    cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
    assert cov.size() == (batch_size, d_latent, d_latent)
    inv_cov = torch.inverse(cov)  # works on batches
    c = d_latent * np.log(2 * np.pi)
    # matrix log det doesn't work on batches!
    _, logabsdets = _batch_slogdet(cov)
    xmu = x - mu
    return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', xmu, inv_cov, xmu))


def log_laplace(x, mu, b, broadcast_size=False):
    """compute the log-pdf of a laplace distribution with diagonal covariance"""
    # b might not have batch_dimension. This case is handled by _check_inputs
    if broadcast_size:
        mu, b = _check_inputs(x.size(), mu, b)
    else:
        mu, b = _check_inputs(None, mu, b)
    return -torch.log(2 * b) - (x - mu).abs().div(b)


def elbo_decomposed(x, f, g, v, s, l, N, a=1., b=1., c=1., d=1., detailed=False):
    """sampled decomposed version of the elbo. The KL term is decomposed into three parts (section 3.3)"""
    # use only for factorial posterior since it's not possible to compute all terms without this assumption
    M, d_latent = s.size()
    decoder_var = .1
    logpx = log_normal(x, f, decoder_var).sum(dim=-1)
    logqs_cux = log_normal(s, g, v).sum(dim=-1)
    logps_cu = log_normal(s, None, l).sum(dim=-1)

    # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
    logqs_tmp = log_normal(s.view(M, 1, d_latent), g.view(1, M, d_latent), v)
    logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
    logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

    elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)).mean()
    if not detailed:
        return elbo
    else:
        return elbo, -logpx.mean(), (logqs_cux - logqs).mean(), (logqs - logqs_i).mean(), (logqs_i - logps_cu).mean()


def elbo_decomposed_vae(x, f, g, v, s, N, a=1., b=1., c=1., d=1., detailed=False):
    """sampled decomposed version of the elbo. The KL term is decomposed into three parts (section 3.3)"""
    # use only for factorial posterior since it's not possible to compute all terms without this assumption
    M, d_latent = s.size()
    decoder_var = .1
    logpx = log_normal(x, f, decoder_var).sum(dim=-1)
    logqs_cux = log_normal(s, g, v).sum(dim=-1)
    logps = log_normal(s, None, None, broadcast_size=True).sum(dim=-1)

    # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
    logqs_tmp = log_normal(s.view(M, 1, d_latent), g.view(1, M, d_latent), v)
    logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
    logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

    elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps)).mean()
    if not detailed:
        return elbo
    else:
        return elbo, -logpx.mean(), (logqs_cux - logqs).mean(), (logqs - logqs_i).mean(), (logqs_i - logps).mean()


def elbo_sampled(x, f, g, v, s, l, a=1., b=1., gaussian=True, full=False):
    """sampled version of the elbo. Even the KL term is approximated by samples here"""
    decoder_var = .1
    logpx_s = log_normal(x, f, decoder_var).sum(dim=-1)
    if full:
        logqs_x = log_normal_full(s, g, v)
    else:
        logqs_x = log_normal(s, g, v).sum(dim=-1)
    if gaussian:
        logps = log_normal(s, None, l, broadcast_size=True).sum(dim=-1)
    else:
        logps = log_laplace(s, None, l, broadcast_size=True).sum(dim=-1)
    return -(a * logpx_s - b * (logqs_x - logps)).mean()


def elbo_closed(x, f, g, v, l, a=1., b=1., gaussian=True, full=False):
    """closed form version of the elbo. The KL term is computed in closed form"""
    decoder_var = .1
    g, v = _check_inputs(None, g, v)
    exp = -0.5 * (np.log(2 * np.pi) + np.log(decoder_var) + (x - f).pow(2) / decoder_var).sum(dim=-1)
    if gaussian:
        if full:
            cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo square"
            _, logabsdet = _batch_slogdet(cov)
            tr = torch.diagonal(cov, dim1=-2, dim2=-1).div(l)
            kld_gauss_full = 0.5 * ((torch.log(l) - 1 + g.pow(2).div(l) + tr).sum(dim=-1) - logabsdet)
            kld = kld_gauss_full
        else:
            kld_gauss = 0.5 * (l.log() - v.log() - 1 + g.pow(2).div(l) + v / l).sum(dim=-1)
            kld = kld_gauss
    else:
        if full:
            raise NotImplementedError(
                "no closed form solution for KL between full variance Gaussian and factorial Laplace")
        else:
            logv = v.log()
            std = v.sqrt()
            kld_lap = -(-2 / np.sqrt(2 * np.pi) * l.mul(std).mul((-g.pow(2).div(2 * v)).exp())
                        + torch.log(l / 2) + 0.5 * logv + 0.5 + np.log(2 * np.pi)
                        - g.mul(l).mul(torch.erf(g.div(std * np.sqrt(2))))).sum(dim=-1)
            kld = kld_lap
    return -(a * exp - b * kld).mean()


def elbo_discrete_vae(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (BCE + KLD) / x.size(0)  # average per batch


def elbo_discrete(recon_x, x, mu, logvar, mup, logl):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    l = logl.exp()
    v = logvar.exp()
    KLD = 0.5 * torch.sum(logl - logvar - 1 + (mu - mup).pow(2).div(l) + v / l)
    return (BCE + KLD) / x.size(0)  # average per batch


# performace metrics

def get_performance(true_s, est_s):
    """
    compare correlation coefficients between true and recovered sources
    this uses correlation coefficient, might be better to take a non-parametric measure of correlation in future...
    """
    p = true_s.shape[1]
    cor_mat = np.corrcoef(np.hstack((true_s, est_s)).T)[:p, p:]
    cor_mat = np.abs(cor_mat)
    return cor_mat[linear_sum_assignment(-1 * cor_mat)].mean()


# plotting

def plot_tcl(x, s, f, g, plot_path, data_path):
    d_latent = s.shape[-1]
    d_data = x.shape[-1]
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='row', figsize=(18, 12))
    colors = ['b', 'r', 'g', 'y', 'v']
    for i in range(d_latent):
        axes[0].plot(s[:, i] * np.sign(s[0, i]), colors[i % len(colors)] + '-',
                     label=r'$s_' + str(i) + '$')
        axes[0].plot(g[:, i] * np.sign(g[0, i]), colors[i % len(colors)] + '--',
                     label=r'$g_' + str(i) + '$')
        axes[0].set_title('sources')
        axes[0].legend()
    for i in range(d_data):
        axes[1].plot(x[:, i] * np.sign(x[0, i]), colors[i % len(colors)] + '-',
                     label=r'$x_' + str(i) + '$')
        axes[1].plot(f[:, i] * np.sign(f[0, i]), colors[i % len(colors)] + '--',
                     label=r'$f_' + str(i) + '$')
        axes[1].set_title('mixed data')
        axes[1].legend()

    plt.savefig(plot_path)
    np.savez_compressed(data_path, s=s, g=g, x=x, f=f)


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)
