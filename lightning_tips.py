import torch
from torch.utils.data import DataLoader, dataset
import pytorch_lightning as pl

### DATALOADERS ##################################################################
# When building DataLoaders. Set `num_workers>0` and `pin_memory=True`
DataLoader(dataset, num_workers=8, pin_memory=True)

### num_workers ##################################################################
# num_workers depends on the batch size and the machine
# A general place to start is to set num_workers = number of CPUs in the machine.
# Increasing num_workers all increases the CPU usage
# BEST TIP: Increase num_workers slowly and stop when there is no performance increase.

### spawn ##################################################################
# PyTorch has issues with `num_workers > 0` and using `spawn`

### .item(), .numpy(), .cpu() ##################################################################
# DONOT call .item() anywhere in your script. PL takes care of it.

### emptycache() ##################################################################
# DONOT call it anywhere un-necessarily.

### TENSOR CREATION ##################################################################
# Construct TENSOR directly on the device when using PL module (self)

t = torch.rand(2, 2).cuda()  # BAD
t = torch.rand(2, 2, device=self.device)  # GOOD

# For tensors that need to be MODEL's ATTRIBUTES, its best to register them as buffers
# in the module's __init__() method

t = torch.rand(2, 2, device=self.device)  # BAD
self.register_buffer("t", torch.rand(2, 2))  # GOOD

### DDP v/s DP ##################################################################
# Use DDP instead of DP.
# DDP is much faster compared to DP.

###  ##################################################################
