import math
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# #### DATA Generation ####################################################################
# num_data = 10000
# len_seq = 32  # target sequence length. input sequence will be twice as long

# # number of "classes", including 0 (the start-token) and 1 (the end-token)
# num_classes = 128

# # only generate ints in (2, 99) range
# Y = (torch.rand((num_data * 10, len_seq - 2), dtype=torch.float32) * (num_classes - 2)) + 2

# # Make sure we only have unique rows
# Y = torch.tensor(np.unique(Y, axis=0)[:num_data])
# X = torch.repeat_interleave(Y, 2, dim=1)

# # Add special "0" (start-token) and "1" (end-token)
# Y = torch.cat([torch.zeros((num_data, 1)), Y, torch.ones((num_data, 1))], dim=1).long()
# X = torch.cat([torch.zeros((num_data, 1)), X, torch.ones((num_data, 1))], dim=1).long()

# # Look at the data
# print(X, X.shape)
# print(Y, Y.shape)
# print(Y.min(), Y.max())

# ############################################################################################
# # Wrap data in the simplest possible way to enable PyTorch data fetching
# # https://pytorch.org/docs/stable/data.html
# BATCH_SIZE = 128
# TRAIN_FRACTION = 0.8

# # this fulfils the torch.utils.data.Dataset interface
# dataset = list(zip(X, Y))

# # split into train/val
# num_train = int(num_data * TRAIN_FRACTION)
# num_val = num_data - num_train
# data_train, data_val = torch.utils.data.random_split(dataset, (num_train, num_val))

# ## creating dataloaders
# dataloader_train = DataLoader(data_train, batch_size=BATCH_SIZE)
# dataloader_val = DataLoader(data_val, batch_size=BATCH_SIZE)

# ## Sample batch
# x, y = next(iter(dataloader_train))
# print("--"*10)
# print(x, "\n", y, "\n", x.shape, y.shape)

############################################################################################
## POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    """[Classic Attention-is-all-you-need positional encoding]

    Args:
        nn ([type]): [description]
    """
    def __init__(self,
                 dim_model,
                 dropout_rate=0.1,
                 max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        pos_enc = torch.zeros(max_len, dim_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        x += self.pos_enc[:x.size(0), :]
        return self.dropout(x)

############################################################################################
def generate_square_subsequent_mask(size: int):
    """Generate a triangular (size, size) mask.

    Args:
        size (int): [description]
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask==0, float("-inf")).masked_fill(mask==1, float(0.0))
    return mask

############################################################################################

class Transformer(nn.Module):
    """Classic Transformer that both encodes & decodes.
    # !Prediction-time inference is done greedily.
    # NOTE: start-token is hard-cded to 0. end-token is hard-coded to 1.
    # NOTE: So, if changing, please update predict() method accordingly.

    Args:
        nn ([type]): [description]
    """
    def __init__(self,
                 num_classes: int,
                 max_output_len: int,
                 dim: int = 128) -> None:
        super().__init__()

        # Parameters
        self.dim = dim
        self.max_output_len = max_output_len
        num_heads = 4
        num_layers = 4
        dim_feedforward = dim

        # Encoder part
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=dim)
        self.pos_encoder = PositionalEncoding(dim_model=self.dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.dim, nhead=num_heads, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )

        # Decoder Part
        self.y_mask = generate_square_subsequent_mask(size=self.max_output_len)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.dim, nhead=num_heads, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.fc_layer = nn.Linear(in_features=self.dim, out_features=num_classes)

        # It is emperically important to initialize weights
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc_layer.bias.data.zero_()
        self.fc_layer.weight.data.uniform_(-init_range, init_range)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            x (torch.Tensor): [(B, Sx) with elements in (0, C) where C is num_classes]
            y (torch.Tensor): [(B, Sy) with elements in (0, C) where C is num_classes]

        Returns:
            torch.Tensor: [(B, C, Sy) logits]
        """
        encoded_x = self.encode(x)          ## (Sx, B, C)
        output = self.decode(y, encoded_x)  ## (Sy, B, C)
        return output.permute(1, 2, 0)      ## (B, C, Sy)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            x (torch.Tensor): [(B, Sx) with elements in (0, C) where C is num_classes]

        Returns:
            torch.Tensor: [(Sx, B, E) embedding]
        """
        x = x.permute(1, 0)                                 ## (Sx, B)
        x = self.embedding(x) * math.sqrt(self.dim)         ## (Sx, B, E)
        x = self.pos_encoder(x)                             ## (Sx, B, E)
        x = self.transformer_encoder(x)                     ## (Sx, B, E)
        return x

    def decode(self, y: torch.Tensor, encoded_x: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            y (torch.Tensor): [(B, Sy) with elements in (0, C) where C is num_classes]
            encoded_x (torch.Tensor): [(Sx, B, E)]

        Returns:
            torch.Tensor: [(Sy, B, C) logits]
        """
        y = y.permute(1, 0)                                         ## (Sy, B)
        y = self.embedding(y) * math.sqrt(self.dim)                 ## (Sy, B, E)
        y = self.pos_encoder(y)                                     ## (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)           ## (Sy, Sy)
        output = self.transformer_decoder(y, encoded_x, y_mask)     ## (Sy, B, E)
        output = self.fc_layer(output)                              ## (Sy, B, C)
        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Method to use at inference time.
        Predict y from x (one token at a time).
        This method is greedy decoding.
        # TODO: Beam Search can be used for accuracy boost.

        Args:
            x (torch.Tensor): (B, Sx) with elements in (0, C) where C is num_classes

        Returns:
            torch.Tensor: (B, C, Sy) logits
        """
        encoded_x = self.encode(x)
        output_tokens = (torch.ones((x.shape[0], self.max_output_len))).type_as(x)  ## (B, max_len)
        output_tokens[:, 0] = 0     ## Set start token
        for Sy in range(1, self.max_output_len):
            y = output_tokens[:, :Sy]                   ## (B, Sy)
            output = self.decode(y, encoded_x)          ## (Sy, B, C)
            output = torch.argmax(output, dim=-1)       ## (Sy, B)
            output_tokens[:, Sy] = output[-1:]          ## Set the last output token
        return output_tokens

############################################################################################
# model = Transformer(num_classes=num_classes, max_output_len=y.shape[1])
# logits = model(x, y[:, :-1])
# print("=="*20)
# print(x.shape, y.shape, logits.shape)
# print(x[0:1])
# print(model.predict(x[0:1]))

############################################################################################

class FlashModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.val_acc = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        ## Teacher Forcing: model gets input upto the last character
        ## while the ground truth is from second character onwards
        logits = self.model(x, y[:, :-1])
        loss = self.loss(logits, y[:, 1:])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss(logits, y[:, 1:])
        self.log("val_loss", loss, prog_bar=True)
        pred = self.model.predict(x)
        self.val_acc(pred, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

############################################################################################
# if __name__ == "__main__":
#### DATA Generation ####################################################################
num_data = 10000
len_seq = 32  # target sequence length. input sequence will be twice as long

# number of "classes", including 0 (the start-token) and 1 (the end-token)
num_classes = 128

# only generate ints in (2, 99) range
Y = (torch.rand((num_data * 10, len_seq - 2), dtype=torch.float32) * (num_classes - 2)) + 2

# Make sure we only have unique rows
Y = torch.tensor(np.unique(Y, axis=0)[:num_data])
X = torch.repeat_interleave(Y, 2, dim=1)

# Add special "0" (start-token) and "1" (end-token)
Y = torch.cat([torch.zeros((num_data, 1)), Y, torch.ones((num_data, 1))], dim=1).long()
X = torch.cat([torch.zeros((num_data, 1)), X, torch.ones((num_data, 1))], dim=1).long()

# Look at the data
# print(X, X.shape)
# print(Y, Y.shape)
# print(Y.min(), Y.max())

############################################################################################
# Wrap data in the simplest possible way to enable PyTorch data fetching
# https://pytorch.org/docs/stable/data.html
BATCH_SIZE = 128
TRAIN_FRACTION = 0.8

# this fulfils the torch.utils.data.Dataset interface
dataset = list(zip(X, Y))

# split into train/val
num_train = int(num_data * TRAIN_FRACTION)
num_val = num_data - num_train
data_train, data_val = torch.utils.data.random_split(dataset, (num_train, num_val))

## creating dataloaders
dataloader_train = DataLoader(data_train, batch_size=BATCH_SIZE)
dataloader_val = DataLoader(data_val, batch_size=BATCH_SIZE)

## Sample batch
x, y = next(iter(dataloader_train))
print("--"*10)
print(x, "\n", y, "\n", x.shape, y.shape)

############################################################################################
model = Transformer(num_classes=num_classes, max_output_len=y.shape[1])
flashModel = FlashModel(model)
early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
trainer = pl.Trainer(max_epochs=5, callbacks=[early_stop_callback])
# trainer = pl.Trainer(fast_dev_run=True)
trainer.fit(flashModel, dataloader_train, dataloader_val)

############################################################################################
