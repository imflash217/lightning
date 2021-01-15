##########################################################################################
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
        self.log("val_loss": val_loss)


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
        self.log("val_loss": val_loss)
        pred = ...
        return pred  # <- this is the new line here

    # <- thi is the new hook here that needs tobe implemented.
    def validation_epoch_end(self, validation_step_outputs):
        for preds in validation_step_outputs:
            # Do something with "pred"

            ##########################################################################################
