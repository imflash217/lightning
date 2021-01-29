## 7.4      [LightningModule API]
## 7.4.1    [Methods]
## Page-55
##
## ! pl.LightningModule.training_step()
##
"""
Here we compute and return the training loss and some additional metrics.
In this step you would normally do the forward pass and compute the loss for the batch.
We can also do fancier things like multiple forward passes or something model specific.

Parameters:
    batch (Tensor | (Tensor,...) | [Tensor, ...]) ->> The output of the dataloader
    batch_idx (int) ->> integer displaying the index of this batch
    optimizer_idx (int) ->> when using multiple optimizers this argument will also be present
    hiddens (Tensor) ->> Passed if `truncated_bptt_steps > 0`

Returns:
    Any of
        Tensor (or) ->> The loss tensor
        dict (or) ->> A dictionary. Can include any keys but MUST include the key "loss"
        None ->> Training will skip to the next batch
"""
## NOTE: The loss value shown in the progress bar is SMOOTHED (i.e. averaged) over its previous values.
## NOTE: So, it differs from the actual lossreturned in training/validation step

import torch
import pytorch_lightning as pl

################################################################################
class FlashModel(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        out = self.encoder(x)
        loss = self.loss(out, x)
        return loss

################################################################################
## ! If we are using MULTIPLE OPTIMIZERS

class FlashModel(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            ## TODO: do training with encoder
            pass
        elif optimizer_idx == 1:
            # TODO: DO training with decoder
            pass

################################################################################
## ! Truncated BPTT

class FlashModel(pl.LightningModule):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm

    def training_step(self, batch, batch_idx, hiddens):
        # ! 'hiddens' are the hidden states from the previous truncated backprop step
        # ...
        out, hiddens = self.lstm(data, hiddens)
        loss = self.loss(out, data)
        # ...
        return {"loss": loss,
                "hiddens": hiddens}

################################################################################