import os

import torch
import torch.nn.Functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl

###########################################################################################


class FlashModel(pl.LightningModule):
    """DOCSTRING"""

    def __init__(self):
        super().__init__()
        self.layer1 = torch.Linear(28*28, 10)

    def forward(self, x):
        out = self.layer1(x.view(x.shape[0], -1))
        out = self.relu(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        """Define optimizers and LR schedulers"""
        return torch.optim.Adam(self.parameters(), lr=0.02)

###########################################################################################


train_loader = DataLoader(datasets.MNIST(
    os.getcwd(), download=True, transform=transforms.ToTensor()))
model = FlashModel()
trainer = pl.Trainer()

trainer.fit(model, train_loader)

###########################################################################################
