import torch
from models.base import BaseVAE
from torch import nn
from torch import bernoulli
from torch.nn import functional as F
from .types_ import *


class BetaVAE_CLF(BaseVAE):

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
        super(BetaVAE_CLF, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            # nn.ConvTranspose2d(
                            #     in_channels=hidden_dims[-1],
                            #     out_channels=hidden_dims[-1],
                            #     kernel_size=3,
                            #     stride=2,
                            #     padding=1,
                            #     output_padding=1
                            #     ),
                            # nn.BatchNorm2d(hidden_dims[-1]),
                            nn.ConvTranspose2d(
                                in_channels=hidden_dims[-1],
                                out_channels=1,
                                kernel_size=4,
                                stride=1,
                                padding=3,
                                output_padding=0
                                ),
                            nn.BatchNorm2d(1),
                            nn.Sigmoid(),
                            # nn.Conv2d(
                            #     hidden_dims[-1], 
                            #     out_channels= 3,
                            #     kernel_size= 3, 
                            #     padding= 1
                            #     ),
                            # nn.Tanh()
                            )

        # Build Classifier
        self.clf = nn.Sequential(
            nn.Linear(latent_dim, 169),
            nn.Softmax(dim=1)
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        #print('DECODER', result.shape)
        result = self.final_layer(result)
        #print('FINAL', result.shape)
        #result = bernoulli(result)
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
    
    def map_label2idx(self, labels: Tensor) -> Tensor:
        if len(labels.shape) == 1:
            return labels[0]*13 + labels[1]
        else:
            return labels[:,0]*13 + labels[:,1]

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
        # the weight of the classification loss is controlled by the parameter clf_w, that depend on self.C_stop_iter
        clf_w = torch.clamp(
            torch.Tensor([1/self.C_stop_iter * self.num_iter,]).to(input.device), 0,  
            1
        )
        clf_loss = F.cross_entropy(preds, self.map_label2idx(labels))
        #clf_loss = F.cross_entropy(preds, labels.squeeze())
        #clf_loss = clf_w*clf_loss

        # Compute total loss
        loss = betavae_loss + clf_w*clf_loss

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