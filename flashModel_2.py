import torch as pt
import pytorch_lightning as pl

##### INIT ##################################################################


class FlashModel(pl.LightningModule):
    """ DON'T DO THIS"""

    def __init__(self, params):
        self.lr = params.lr
        self.coeff_x = params.coeff_x


class FlashModel(pl.LightningModule):
    """Instead DO THIS"""

    def __init__(self,
                 encoder: pt.nn.Module = None,
                 coeff_x: float = 0.2,
                 lr: float = 1e-3):
        pass
