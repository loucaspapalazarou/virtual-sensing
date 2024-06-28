from mamba_ssm import Mamba  # type: ignore

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import FrankaDataset
import sys
from lightning.pytorch.loggers import WandbLogger

# TODO
"""
- Organize code
- Comments
- Parameterize everything
"""


class FrankaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = FrankaDataset()  # Replace with your dataset initialization
        self.train_dataset, self.val_dataset = random_split(dataset, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MambaModel(pl.LightningModule):
    def __init__(
        self,
        d_model=35,
        lr=0.001,
        step_size=5,
    ):
        super().__init__()
        # self.model = nn.Transformer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     batch_first=True,
        #     num_encoder_layers=3,
        #     num_decoder_layers=3,
        #     dim_feedforward=256,
        # )
        self.model = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )

        self.lr = lr
        self.step_size = step_size
        self.save_hyperparameters()

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        batch_size, seq_len, input_size = batch.size()
        num_subsequences = (seq_len - 1) // self.step_size

        total_loss = 0.0
        for i in range(num_subsequences):
            start_idx = i * self.step_size
            end_idx = start_idx + seq_len - 1

            src = batch[:, start_idx:end_idx, :]
            tgt = batch[:, start_idx + 1 : end_idx + 1, :]

            output = self(src, tgt)
            loss = nn.functional.mse_loss(output, tgt)

            total_loss += loss

        avg_loss = total_loss / num_subsequences
        self.log("train_loss", avg_loss, sync_dist=True)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        batch_size, seq_len, input_size = batch.size()
        num_subsequences = (seq_len - 1) // self.step_size

        total_loss = 0.0
        for i in range(num_subsequences):
            start_idx = i * self.step_size
            end_idx = start_idx + seq_len - 1

            src = batch[:, start_idx:end_idx, :]
            tgt = batch[:, start_idx + 1 : end_idx + 1, :]

            output = self(src, tgt)
            loss = nn.functional.mse_loss(output, tgt)

            total_loss += loss

        avg_loss = total_loss / num_subsequences
        self.log("val_loss", avg_loss, sync_dist=True)
        return avg_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    run_name = "default_run_name"
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]

    dm = FrankaDataModule(batch_size=32)
    model = MambaModel(d_model=36, nhead=4, lr=0.001, step_size=4)

    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Monitor validation loss
        dirpath="checkpoints/",  # Directory to save checkpoints
        filename="{run_name}-{epoch:02d}-{val_loss:.2f}",  # Format for checkpoint filenames
        save_top_k=-1,  # Save all checkpoints
        mode="min",  # Save the model with the lowest validation loss
    )

    wandb_logger = WandbLogger(project="transformer", name=run_name)

    trainer = pl.Trainer(
        max_epochs=10, callbacks=[checkpoint_callback], logger=wandb_logger
    )
    trainer.fit(model, dm)
