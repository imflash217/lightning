import torch
import pytorch_lightning as pl

# A LightningModule ORGANIZES the PyTorch code into the following modules:
# 1. Computations (init)
# 2. Training loop (training_step)
# 3. Validation loop (validation_step)
# 4. Test loop (test_step)
# 5. Optimizers (configure_optimizers)

##############################################################################
model = FlashModel()
trainer = Trainer()
trainer.fit(model)

### NO .cuda() or .to() calls in PL #######################################
# DO NOT do this with PL
x = torch.Tensor(2, 3)
x = x.cuda()
x.to(device)

# INSTEAD DO THIS
x = x               # leave it alone!

# or to init a new tensor fo this ->
new_x = torch.tensor(2, 3)
new_x = new_x.as_type(x)

############# NO SAMPLERS for distributed
# DON'T DO THIS
data = MNIST(...)
sampler = DistributedSampler(data)
DataLoader(data, sampler=sampler)

# DO THIS
data = MNIST(...)
DataLoader(data)

############# A LightningModule is a torch.nn.Module with added functionality. Use it as such
model = FlashModel.load_from_checkpoint(PATH)
model.freeze()
out = model(x)

###########################################################################################