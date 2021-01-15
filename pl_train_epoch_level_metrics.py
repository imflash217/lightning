import torch
import torch.nn.Functional as F
import pytorch_lightning as pl

###########################################################################################

class FlashModel(pl.LightningModule):
    """DOCSTRING"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        ## logs metrics for each training_step
        ## and the average across each epoch, to the logger and progress-bar
        self.log("train_loss", loss,
                 on_step=True,
                 on_epoch=True,
                 logger=True,
                 prog_bar=True
                )
        return loss

###########################################################################################
## Under the hood
outs = []
for batch in train_dataloader:
    ## Step-1: FORWARD
    out = training_step(val_batch)
    ## Step-2: BACKWARD
    loss.backward()
    ## optim step and cread grads
    optimizer.step()
    optimizer.zero_grad()

epoch_metric = torch.mean(torch.stack([x["train_loss"] for x in outs]))

###########################################################################################