# coding: utf-8

# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2018 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal

class VAE(nn.Module):
    ''' WorldModels V model: a convolutional variational autoencoder. '''
    def __init__(self, training=True):
        ''' Initialize a VAE model.

        Parameters
        ----------
        training : boolean, optional (default=True)
            Whether the model is in training mode (should sample from the distribution).
        '''
        super().__init__()
        self.relu = nn.ReLU()
        self.training = training
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0)

        self.mu = nn.Linear(2 * 2 * 256, 32)
        self.sigma = nn.Linear(2 * 2 * 256, 32)

        self.fc = nn.Linear(32, 1024)

        self.conv5 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2, padding=0, output_padding=0)
        self.conv6 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0, output_padding=0)
        self.conv7 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0, output_padding=0)
        self.conv8 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0, output_padding=0)

    def encode(self, x):
        ''' Perform the encoding step of the VAE.

        Parameters
        ----------
        x : torch.Tensor, shape=(N, K, R, C)
            The input images to encode.

        Returns
        -------
        Tuple[torch.Tensor shape=(N, 32), torch.Tensor shape=(N, 32), torch.Tensor shape=(N, 32)]
            (mu, log_var, z): the mean and variance vectors and a sampled vector from the normal distribution.
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.relu(self.mu(x))
        log_var = self.relu(self.sigma(x))
        return mu, log_var

    def reparameterize(self, mu, log_var):
        ''' Reparameterize the distribution.

        Parameters
        ----------
        mu : torch.Tensor, shape=(N, 32)
            The mean of the distribution.

        log_var : torch.Tensor, shape=(N, 32)
            The log of the variance.

        Returns
        -------
        torch.Tensor, shape=(N, 32)
            Reparameterized mu.
        '''
        if self.training:
            std = log_var.mul(0.5).exp_() # sqrt(exp(logvar))
            rsample = Variable(std.data.new(std.size()).normal_())
            return rsample.mul(std).add_(mu)
        return mu

    def decode(self, z):
        ''' Perform the decoding step of the VAE.

        Parameters
        ----------
        z : torch.Tensor, shape=(N, 32)
            A batch of z vectors sampled from the VAE encoder distribution to decode.

        Returns
        -------
        torch.Tensor, shape=(N, 3, R, C)
            The reproduced image.
        '''
        x = F.relu(self.fc(z))
        x = x.view(-1, 1024, 1, 1)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        return F.sigmoid(self.conv8(x))

    def forward(self, x):
        ''' Perform a forward pass of the VAE.

        Parameters
        ----------
        x ; torch.Tensor, shape=(N, K, R, C)
            An input batch of images.

        Returns
        -------
        Tuple[torch.Tensor shape=(N, 3, R, C), torch.Tensor shape=(N, 32), torch.Tensor shape=(N, 32)]
            (reproduced image, mu, log_var)
        '''
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def vae_loss(recon_x, x, mu, log_var):
    ''' The VAE loss function.

    Parameters
    ----------
    recon_x : torch.Tensor, shape=(N, K, R, C)
        The reconstructed image.

    x : torch.Tensor, shape=(N, K, R, C)
        The input image.

    mu : torch.Tensor, shape=(N, 32)
        The mu vector.

    log_var : torch.Tensor, shape=(N, 32)
        The variance vector.

    Returns
    -------
    Tuple[torch.Tensor shape=(), torch.Tensor shape=()]
        (MSE between the reconstructed and actual image, KL-divergence)
    '''
    mse_loss = F.mse_loss(recon_x, x, size_average=False)
    kl_div_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return mse_loss, kl_div_loss

def vae_seg_loss(output, target, mu, log_var):
    ''' Segmentation cross entropy loss

    Parameters
    ----------
    ouput : torch.Tensor, shape=(N, K, R, C)
        The output segmentation.

    target : torch.Tensor, shape=(N, 1, R, C)
        The target segmentation.

    mu : torch.Tensor, shape=(N, 32)
        The mu vector.

    log_var : torch.Tensor, shape=(N, 32)
        The variance vector.

    Returns
    -------
    Tuple[torch.Tensor shape=(), torch.Tensor shape=()]
        (CE between the reconstructed and target segmentation, KL-divergence)
    '''

    ce_loss = F.cross_entropy(output, target)
    kl_div_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return ce_loss, kl_div_loss


## learns prior parameters for every parameter in z in the gaussian Mixture
class MDNRNN(nn.Module):
    ''' World models m-model. A Mixture Density Network combined with a Recurrent Neural Network '''
    def __init__(self, z_size=32, a_size=3, n_hidden=256, n_gaussian=5, n_layers=1):
        ''' Initialize an MDNRNN.

        Parameters
        ----------
        z_size : int, optional (default=32)
            The dimensionality of the latent vector.

        a_size : int, optional (default=3)
            The number of actions.

        n_hidden : int, optional (default=256)
            The dimensionality of the hidden state.

        n_gaussian : int, optional (default=5)
            The assumed number of Gaussians in the mixture density distribution.

        n_layers : int, optional (default=1)
            The number of layers in the LSTM.
        '''
        super().__init__()
        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_gaussian = n_gaussian
        self.lstm = nn.LSTM((z_size + a_size), n_hidden, n_layers, batch_first=False)
        self.dense = nn.Linear(n_hidden, n_gaussian*z_size*3)

    def get_mixture(self, y):
        mu, log_sigma, log_pi = self.dense(y).reshape(y.size(0), -1, 3).permute(2, 0, 1)

        shape = (-1, 1, self.n_gaussian, self.z_size)
        log_pi =  F.log_softmax(log_pi.view(*shape), 2)
        mu = mu.view(*shape)
        sigma = log_sigma.view(*shape).exp()
        return log_pi, mu, sigma

    def forward(self, x, h):
        y, (h, c) = self.lstm(x, h)
        log_pi, mu, sigma = self.get_mixture(y)
        return (log_pi, mu, sigma), (h, c)

    def init_hidden(self, seq_len, device):
        return (torch.zeros(1 ,self.n_layers, self.n_hidden).to(device),
                torch.zeros(1 ,self.n_layers, self.n_hidden).to(device))

def mdnrnn_loss(log_pi, mu, sigma, z):
    ''' Compute the loss for the MDNRNN.

    Parameters
    ----------
    log_pi : torch.Tensor, shape=(N, n_gaussians)
        The weighting over the distributions.

    mu : torch.Tensor, shape=(seq_length, N, n_gaussians, z_size)
        The mean vector.

    sigma : torch.Tensor, shape=(seq_length, N, n_gaussians, z_size)
        The std.

    z : torch.Tensor, shape=(N, 32)
        The state vector.

    Returns
    -------
    torch.Tensor, shape=()
        The loss.
    '''
    z = z.view(-1, 1, 1, 32).expand(-1, 1, 5, 32)
    m = torch.distributions.Normal(loc=mu, scale=sigma)

    log_prob = m.log_prob(z)
    loss = torch.sum(log_prob + log_pi, dim=2)
    return -1*loss.mean()

def gumbel_sample(pi, mu, sigma):
    ''' Sample a z-vector from a mixture of gausian distribution

    Parameters
    ----------
    pi : torch.Tensor, shape=(N, n_gaussians)
        The weighting over the distributions.

    mu : torch.Tensor, shape=(seq_length, N, n_gaussians, z_size)
        The mean vector.

    sigma : torch.Tensor, shape=(seq_length, N, n_gaussians, z_size)
        The std.
    '''
    print(pi.shape)

    pi = pi.view(1, 1, 5, 1).expand(mu.shape).detach().numpy()
    mu = mu.detach().numpy()
    sigma = sigma.detach().numpy()
    rollout_length = mu.shape[0]

    z = np.random.gumbel(loc=0, scale=1, size=pi.shape)
    idx = (pi + z).argmax(axis=2)

    I, J, K = np.ix_(np.arange(rollout_length), np.arange(pi.shape[1]), np.arange(pi.shape[3]))
    rn = np.random.randn(rollout_length, 1, 1, 32)
    s = np.multiply(rn, sigma[I, J, idx, K].reshape(rollout_length, 1, 1, 32))
    s += mu[I, J, idx, K].reshape(rollout_length, 1, 1, 32)
    return torch.tensor(s)

class CONTROLLER(nn.Module):
    ''' World models c-model. A fully connected layer '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256+32, 3, bias=True)
    ''' Feed forward z-vector and hidden state. Scale returned action to valid action space.

    Parameters
    ----------
    z : torch.Tensor, shape=(1, 1, 32)
        The z-vector latent space represenation of the current observation

    h : torch.Tensor, shape=(1, 1, 256)
        The hidden state of the LSTM in the m-model at the current time step
    '''
    def get_action(self, z, h):
        i = torch.cat((z, h.view(1, 256)), 1)
        y = self.fc1(i).view(3)

        y = y.tanh()
        y[1] = (y[1] + 1) / 2
        y[2] = (y[2] + 1) / 2
        return y.detach().cpu().numpy()
