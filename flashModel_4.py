#### FORWARD & TRAINIG STEP ########################################################################
import torch as pt
import pytorch_lightning as pl


class FlashModel(pl.LightningModule):
    """DOCTSRING"""

    def __init__(self): pass

    def forward(self, x, ...):
        """ use this for inference/predictions"""
        embeddings = self.encoder(x)

    def training_step(self, batch, ...):
        """use this for training only"""
        x, y = batch
        z = self.encoder(x)
        # <-- when using data-parallel DP/DDP call this instead of self.encoder()
        z = self(x)
        pred = self.decoder(z)
        ...

####################################################################################################
