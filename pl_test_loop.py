# 7.2.3 [Test Loop]
# Page-40
##########################################################################################
"""
The process of adding a Test-loop is similar to the process sof adding a validation loop.
The only difference is that the test loop is called when ".test()" is used.
"""
##########################################################################################

import pytorch_lightning as pl
import torch

class FlashModel(pl.LightningModule):
    pass

flash_model = FlashModel()
trainer = pl.Trainer()
trainer.fit()

# automatically loads the best weights
trainer.test(model=flash_model)

##########################################################################################
# there are two ways to call ".test()"
# 1. call after training
# 2. call with pretrained model

## 1. call after training
trainer = pl.Trainer()
trainer.fit(flash_model)
trainer.test(test_dataloaders=test_dataloader) # automatically loads the best model weights

## 2. call with pretrained model
model = MyLightningModule.load_from_checkpoint(PATH)
trainer = pl.Trainer()
trainer.test(model, test_dataloaders=test_dataloader)

##########################################################################################
