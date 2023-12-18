import io
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchvision.utils as vutils
import torch
from torch import TensorType as Tensor
from torch import optim

from src.models import BaseVAE

import warnings
warnings.filterwarnings("ignore")


class VAEXperiment(pl.LightningModule):

    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(
            *results,
            M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
            batch_idx = batch_idx
            )
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(
            *results,
            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
            batch_idx = batch_idx
            )
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        self.sample_images()
        self.plot_latent_traversals()
    
    @staticmethod
    def modified_savefig(fig, dpi):
        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw', dpi=dpi)
            buff.seek(0)
            #data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
            img_arr = np.reshape(
                np.frombuffer(buff.getvalue(), 
                dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), 
                int(fig.bbox.bounds[2]), -1)
                )
        return img_arr

    def plot_latent_traversals(self, delta=4, n_points=50):

        dim = self.model.latent_dim
        fig, axs = plt.subplots(1, dim, figsize=(3*dim,3))
        fig.tight_layout()

        # Update the frames for the movie
        rows = []

        for i in np.linspace(-delta,delta,n_points):
            for latent_channel in range(dim):
                z_ = torch.zeros(1, self.model.latent_dim).to(self.curr_device)
                z_[:,latent_channel] = i
                #z_[:,0] = 1
                #z_[:,3] = j
                rec = self.model.decode(z_)
                _ = axs[latent_channel].imshow(rec.squeeze().detach().cpu().numpy(), cmap='binary')
            plt.close()
            rows.append(self.modified_savefig(fig, 100))

        out_file_path = os.path.join(self.logger.log_dir, "LatentSpace", f"latentspace_epoch_{self.current_epoch}.mp4")
        imageio.mimsave(out_file_path, rows, fps=5, macro_block_size=None)
        

    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(
            recons.data,
            os.path.join(self.logger.log_dir , 
                        "Reconstructions", 
                        f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"
                        ),
            normalize=True,
            nrow=12
            )

        try:
            samples = self.model.sample(
                144,
                self.curr_device,
                labels = test_label
                )
            vutils.save_image(
                samples.cpu().data,
                os.path.join(
                    self.logger.log_dir , 
                    "Samples",      
                    f"{self.logger.name}_Epoch_{self.current_epoch}.png"
                    ),
                normalize=True,
                nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
            )
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(
                    getattr(self.model,self.params['submodel']).parameters(),
                    lr=self.params['LR_2']
                    )
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optims[0],
                    gamma = self.params['scheduler_gamma']
                    )
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(
                            optims[1],
                            gamma = self.params['scheduler_gamma_2']
                            )
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
