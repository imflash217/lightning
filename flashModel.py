import torch as pt
import pytorch_lightning as pl

#######################################################################


class FlashModel(pl.LightningModule):
    """This defines a MODEL"""

    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.layer1 = pt.nn.Linear()
        self.layer2 = pt.nn.Linear()
        self.layer3 = pt.nn.Linear()


class FlashModel(pl.LightningModule):
    """This defines a SYSTEM"""

    def __init__(self,
                 encoder: pt.nn.Module = None,
                 decoder: pt.nn.Module = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
