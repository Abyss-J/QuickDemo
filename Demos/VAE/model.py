import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, hiddens=[32,64,128,256], latent_dim=128, KL_weight=0.0003) -> None:
        super(VAE, self).__init__()
        self.weight = KL_weight

        # encoder
        prev_channel= 3 
        image_length = 32
        modules = []
        for cur_channel in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channel, cur_channel, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(cur_channel),
                    nn.ReLU()
                )
            )
            prev_channel = cur_channel
            image_length //= 2
        self.encoder = nn.Sequential(*modules)

        # mean and var
        self.mean = nn.Linear(prev_channel*image_length*image_length, latent_dim)
        self.var = nn.Linear(prev_channel*image_length*image_length, latent_dim)

        # decoder 
        modules = []
        self.decoder_projector = nn.Linear(latent_dim, prev_channel*image_length*image_length)
        self.decoder_input_szie = (prev_channel, image_length, image_length)

        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i - 1]),
                    nn.ReLU()
                )
            )
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(hiddens[0], 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
            )
        )
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        x = torch.flatten(self.encoder(x), start_dim=1)
        mean = self.mean(x)
        log_var = self.var(x)
        return mean, log_var
    
    def reparameterization(self, mean, log_var):
        eps = torch.rand_like(log_var)
        std = torch.exp(log_var / 2)
        z = eps * std + mean
        return z
    
    def decode(self, z):
        x = self.decoder_projector(z)
        x = torch.reshape(x, (-1, *self.decoder_input_szie))
        out = self.decoder(x)
        return out

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        out = self.decode(z)
        loss = self.loss_function(x, out, mean, log_var)

        return out, mean, log_var, loss

    def loss_function(self, x, out, mean, log_var):
        reconstruct_loss = F.mse_loss(out, x)
        KL_loss = torch.mean(
            -0.5 * torch.sum(1+log_var-mean**2 - torch.exp(log_var), 1), 0
        )
        loss = reconstruct_loss + self.weight * KL_loss
        return loss
