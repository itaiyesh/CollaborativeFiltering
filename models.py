import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import warnings


class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                     d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.weights) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, half_precision=False, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        self.half_precision = half_precision
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        if half_precision:
            self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out).half() for
                                           d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
            self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out).half() for
                                           d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
            self.drop = nn.Dropout(dropout).half()
        else:
            self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                           d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
            self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                           d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
            self.drop = nn.Dropout(dropout)

        # Put all layers in GPU but the first/last (big ones)
        # l = [self.q_layers[0]]
        # for layer in self.q_layers[1:]:
        #     l.append(layer.to(torch.device("cuda")))
        #
        # p = []
        # for layer in self.p_layers[:-1]:
        #     p.append(layer.to(torch.device("cuda")))
        # p.append(self.p_layers[-1])
        #
        # self.q_layers = nn.ModuleList(l)
        # self.p_layers = nn.ModuleList(p)
        # .................#

        if half_precision:
            warnings.warn("Not initializing weights (half precision)")
        else:
            self.init_weights()

        print("Model VAE: {}".format(p_dims))

    def forward_regular(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def forward_sparse(self, input):
        mu, logvar = self.encode_sparse(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def forward(self, input):
        return self.forward_regular(input)
        # if self.half_precision:
        #     return self.forward_sparse(input)
        # else:
        #     return self.forward_regular(input)

    def encode(self, input):
        h = F.normalize(input)

        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def encode_sparse(self, input):
        # h = F.normalize(input)
        #
        # h = self.drop(h)
        h = input

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class SparseMultiVAE(nn.Module):
    def __init__(self, in_dim):
        super(SparseMultiVAE, self).__init__()

        self.in_dim = in_dim
        self.title_emb_dim = 512
        self.small_dim = 50
        hidden_dim = 300

        # Encode
        self.emb1 = nn.EmbeddingBag(in_dim, hidden_dim, mode='sum')
        self.l1_title = nn.Linear(self.title_emb_dim, hidden_dim)

        # Input from concat of emb1 and l1_title
        self.l1 = nn.Linear(2*hidden_dim, 2 * self.small_dim)

        # Decode
        self.l2 = nn.Linear(self.small_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, in_dim)
        self.drop = nn.Dropout(0.5)

        # self.init_weights()

    def forward(self, array, offsets, weights, embs):
        mu, logvar = self.encode(array, offsets, weights, embs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, array, offsets, weights, embs):
        # TODO: Put back?
        # h = F.normalize(input)
        # h = self.drop(h)

        print(array)
        print(offsets)
        print(embs)
        exit()

        x = self.emb1(input = array, offsets = offsets, per_sample_weights=weights)
        e = self.l1_title(embs)
        x = torch.cat((x,e), dim=1)
        x = self.drop(x)
        # print(x.shape)

        x = self.l1(x)
        x = torch.tanh(x)
        mu = x[:, :self.small_dim]
        logvar = x[:, self.small_dim:]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = self.l2(z)
        z = torch.tanh(z)
        z = self.l3(z)
        return z


    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    # BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    # BCE2 = -torch.mean(torch.sum(torch.mm(F.log_softmax(recon_x, 1),x.to_dense().t()), -1))
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x.to_dense(), -1))

    # print("{} - {}".format(BCE, BCE2))

    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD
