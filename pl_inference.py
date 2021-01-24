## 7.3 [INFERENCE]
## Page-40

import torch
from torch import mode, nn
import pytorch_lightning as pl

"""
For research Lightning Modules are best structured as systems.
"""

class Autoencoder(pl.LightningModule):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=28*28)
        )

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # ! step-1: encode
        x = x.view(x.size(0), -1)
        z = self.encoder(x)

        # ! step-2: decode
        recons = self.decoder(z)

        # ! step-3: reconstruction
        recons_loss = nn.functional.mse_loss(recons, x)
        return recons_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        recons = self.decoder(z)
        val_recons_loss = nn.functional.mse_loss(recons, x)
        self.log("val_recons_loss", val_recons_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=0.0002)

###########################################################################################
## ! Training schedule
##
model_autoencoder = Autoencoder()
trainer = pl.Trainer(gpus=1)
trainer.fit(model=model_autoencoder,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader
            )

###########################################################################################
# !NOTE: In the above "Autoencoder" Module;
# !NOTE: the code inside "training_step()" and "validation_step()" are same.
# !NOTE: So, we can club these common parts as a separate method. (as shown below)
###########################################################################################

class Autoencoder(pl.LightningModule):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=28*28)
        )

    def shared_step(self, batch):
        x, _ = batch

        # ! step-1: encode
        x = x.view(x.size(0), -1)
        z = self.encoder(x)

        # ! step-2: decode
        recons = self.decoder(z)

        # ! step-3: reconstruction
        recons_loss = nn.functional.mse_loss(recons, x)
        return recons_loss

    def training_step(self, batch, batch_idx):
        recons_loss = self.shared_step(batch)
        return recons_loss

    def validation_step(self, batch, batch_idx):
        val_recons_loss = self.shared_step(batch)
        self.log("val_recons_loss", val_recons_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=0.0002)

###########################################################################################
###########################################################################################
################# INFERENCE IN PRODUCTION #################################################
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

class ClassificationTask(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        ## ! loss is a tensor
        ## ! the Checkpoint Callback is monitoring the "checkpoint_on"
        metrics = {"val_acc": acc,
                   "val_loss": loss}
        self.log_dict(metrics)
        return metrics
    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {"test_acc": metrics["val_acc"],
                   "test_loss": metrics["val_loss"]}
        self.log_dict(metrics)
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.02)

#################################################
# Now, we can pass in any model to be fit with the task
for model in [Resnet50(), VGG16(), BidirectionalRNN()]:
    task = ClassificationTask(model)
    trainer = pl.Trainer(gpus=2)
    trainer.fit(task,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloader
                )

#################################################
# The tasks can be arbitrarily complex like GAN-training, RL, self-supervised etc
class GANTask(pl.LightningModule):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator,
        self.discriminator = discriminator
        # TODO: Do something....
        # TODO: ...

#################################################
# ! When used like above, model can be kept separate from the task and used in production
# ! without needing to keep it in a pl.LightningModule()
##################################################################################################