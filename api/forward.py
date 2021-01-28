## 7.4      [LightningModule API]
## 7.4.1    [Methods]
## Page-46
##
## ! pl.LightningModule.forward(*args, **kwargs)
##
"""
Same as 'torch.nn.Module.forward()'
However in Lightning; we want it to define the operations we want to use it during prediction.
"""

###################################################################################################
###################################################################################################
# example: if we were using this model as a feature extractor.

import torch
import pytorch_lightning as pl

class FlashModel(pl.LightningModule):
    def forward(self, x):
        feature_maps = self.convnet(x)
        return feature_maps
    def training_step(self, batch, batch_idx):
        x, y = batch
        feature_maps = self(x)                  ## ! <-- calls forward() method
        logits = self.classifier(feature_maps)
        ## TODO: DO something ...
        return loss

######################
# splitting it this way allows the model to be used as a feature extractor
model = FlashModel()
inputs = server.get_request()
results = model(inputs)
server.write_results(results)

########################################################################################
########################################################################################
# Using vanilla "torch.nn.Module" only.
# ! So use "pl.LightningModule.forward()" instead of "torch.nn.Module.forward()"

class Model(torch.nn.Module):
    def forward(self, batch):
        x, y = batch
        feature_maps = self.convnet(x)
        logits = self.classifier(feature_maps)
        return logits
########################################################################################
########################################################################################
