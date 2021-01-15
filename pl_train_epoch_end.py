import torch
import torch.nn.Functional as F
import pytorch_lightning as pl

##########################################################################################

class FlashModel(pl.LightningModule):
    """DOCSTRING"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        preds = ...
        return {"loss": loss, "preds": preds}

    def training_epoch_end(self, training_step_outputs):
        for pred in training_step_outputs:
            ## do something
            pass
        pass

##########################################################################################
## Under the hood pseudocode
outs = []
for batch in train_dataloader:
    ## Step-1: FORWARD
    out = training_step(val_batch)

    ## Step-2: BACKWARD
    loss.backward()

    ## Step-3: Optim step and zero-grad
    optimizer.step()
    optimizer.zero_grad()

training_epoch_end(outs)

##########################################################################################