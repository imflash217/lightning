import torch
import torch.nn.Functional as F
import pytorch_lightning as pl

###########################################################################################
# Pytorch_Lightning version
##


class FlashModel(pl.LightningModule):
    """DOCSTRING"""

    def __init__(self, model):
        super().__init__()
        self.model = model

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y)
            return loss


###########################################################################################
# Under the hood PL does the following =>
##
# Step-1: Put the model in train mode
model.train()
torch.set_grad_enabled = True
losses = []

for batch in train_dataloader:
    # Step-2: Forward
    loss = training_step(batch)
    losses.append(loss.detach())

    # Step-3: Backward
    loss.backward()

    # Step-4: apply optimizer step and clear grads
    optimizer.step()
    optimizer.zero_grad()

###########################################################################################
