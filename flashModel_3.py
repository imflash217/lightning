import torch as pt
import pytorch_lightning as pl

#######################################################################
# A typical PyTorch Lightning Model looks like this =>


class FlashModel(pl.LightningModule):
    """DOCSTRING"""
    def __init__(): pass
    def forward(): pass
    def training_step(): pass
    def training_step_end(): pass
    def training_epoch_end(): pass
    def validation_step(): pass
    def validation_step_end(): pass
    def validation_epoch_end(): pass
    def test_step(): pass
    def test_step_end(): pass
    def test_epoch_end(): pass
    def configure_optimizers(): pass

    def any_other_custom_hooks(): pass

#######################################################################
