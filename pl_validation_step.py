##########################################################################################
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn.Functional as F
import pytorch_lightning as pl

##########################################################################################

class FlashModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)


##########################################################################################
# Under the Hood
for batch in train_dataloader:
    loss = model.training_step()
    loss.backward()
    # .....
    if validate_at_some_point:
        # disable grads + batchnorm + dropout
        torch.set_grad_enabled(False)
        model.eval()

        ##------------------- VAL loop -------------------##
        for val_batch in model.val_dataloader:
            val_out = model.validation_step(val_batch)
        ##------------------- VAL loop -------------------##

        # enable grads + batchnorm + dropout
        torch.set_grad_enabled(True)
        model.train()

##########################################################################################
# If we need to do something with the validation outputs implement validation_epoch_end() hook.


class FlashModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)
        pred = ...
        return pred  # <- this is the new line here

    # <- this is the new hook here that needs to be implemented.
    def validation_epoch_end(self, validation_step_outputs):
        for preds in validation_step_outputs:
            # Do something with "pred"

##########################################################################################
##########################################################################################
# VALIDATION WITH DATA PARALLEL:
# When training using an "accelerator" that splits data into multiple GPUs
# Sometimes we need to aggregate them on master GPU for processing.
# so implement "validation_step_end()" method

class FlashModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        pred = ...
        return {"loss": loss,
                "pred": pred}

    def validation_step_end(self, batch_parts):
        gpu_0_pred = batch_parts.pred[0]["pred"]
        gpu_1_pred = batch_parts.pred[1]["pred"]
        # Do something with both outputs
        # ...
        return (batch_parts[0]["loss"] + batch_parts[1]["loss"]) / 2

    def validation_epoch_end(self, validation_step_outputs):
        for out in validation_step_outputs:
            ### Do Something here....
            # ...

##########################################################################################
# Under the hood
outs = []
for batch in dataloader:
    batches = split_batch(batch)
    dp_outs = []
    for sub_batch in batches:
        # Step-1
        dp_out = validation_step(sub_batch)
        dp_outs.append(dp_out)
    # Step-2
    out = validation_step_end(dp_outs)
    outs.append(out)

## here, Do something with the outputs of all batches
# Step-3
validation_epoch_end(outs)
##########################################################################################
