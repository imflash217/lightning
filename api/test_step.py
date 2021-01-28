## 7.4      [LightningModule API]
## 7.4.1    [Methods]
## Page-50
##
## ! pl.LightningModule.test_step()
##
"""
Operates on a single batch of data from the test set.
In this step you would normally generate examples or calculate anything of interest such as accuracy.
"""

import torch
import torchvision
import pytorch_lightning as pl

# The pseudocode for these calls
test_outs = []
for test_batch in test_data:
    out = test_step(test_batch)
    test_outs.append(out)
test_epoch_end(test_outs)

## if we have one test dataloader
def test_step(self, batch, batch_idx):
    pass

# if we have multiple test dataloaders
def test_step(self, batch, batch_idx, dataloader_idx):
    pass

################################################################################
# Examples: case-1 [a single test dataset]
def test_set(self, batch, batch_idx):
    x, y = batch

    # implement your own
    out = self(x)
    loss = self.loss(out, y)

    #log 6 example images or generated text or whatever
    sample_images = x[:6]
    grid = torchvision.utils.make_grid(sample_images)
    self.logger.experiment.add_image("example_images", grid, 0)

    # calculate accuracy
    labels_hat = torch.argmax(out, dim=1)
    test_acc = torch.sum(y==labels_hat).item() / (len(y) * 1.0)

    # log the outputs
    self.log_dict({"test_loss": loss,
                   "test_acc": test_acc})

################################################################################
# If we pass in multiple test datasets
def test_set(self, batch, batch_idx, dataloader_idx):
    """
    dataloader_idx ->> tells which dataset to use during test iterations
    """

    # TODO: do whatever you want
    # TODO: ...
    pass

################################################################################
# NOTE: If you don't need to validate then you don't need to implement this method.
# NOTE: When the test_step() is called; the model has been put in EVAL mode and all gradients have been disbaled.
#       At the end of the test epoch, the model goes back to the training mode and the gradients are enabled.
################################################################################