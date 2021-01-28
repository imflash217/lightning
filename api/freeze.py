## 7.4      [LightningModule API]
## 7.4.1    [Methods]
## Page-47
##
## ! pl.LightningModule.freeze()
##
"""
Freezes all parameters for inference.

Return:
    None
"""

###################################################################################################
from lightning.api.forward import FlashModel
import pytorch_lightning as pl
model = FlashModel()
model.freeze()                                  ## ! <<- this freezes all model parameters.

###################################################################################################