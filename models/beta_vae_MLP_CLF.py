import torch
from models.base import BaseVAE
from torch import nn
from torch import bernoulli
from torch.nn import functional as F
from .types_ import *


class BetaVAE_MLP_CLF(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE_MLP_CLF, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(169, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_var = nn.Linear(16, latent_dim)

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 169),
            nn.Sigmoid()
        )

        # Build Classifier
        self.clf = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 9),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        input = input.view(-1, 169)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        #print('DECODER', result.shape)
        #result = bernoulli(result)
        result = result.view(-1, 1, 13, 13)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def classify(self, z: Tensor) -> Tensor:
        preds = self.clf(z)
        return preds 

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        labels = kwargs['labels']
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        preds = self.classify(z)
        return  [recons, input, mu, log_var, preds, labels]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons, input, mu, log_var, preds, labels = args

        # Compute beta-VAE loss
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            betavae_loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            betavae_loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        # Compute classification loss
        #print(preds.shape, labels.shape)
        clf_loss = F.cross_entropy(preds, labels.squeeze())

        # Compute total loss
        loss = betavae_loss + clf_loss

        return {'loss':loss, 'betavae_loss': betavae_loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss, 'clf_loss':clf_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]