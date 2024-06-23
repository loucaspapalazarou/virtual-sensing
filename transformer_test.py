import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset
from torch import nn
import pytorch_lightning as pl
from dataset import FrankaDataset  # Assuming you have defined FrankaDataset somewhere
from constants import DEVICE


class FrankaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = FrankaDataset()  # Replace with your dataset initialization
        self.train_dataset, self.val_dataset = random_split(dataset, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


class TransformerModel(pl.LightningModule):
    def __init__(self, d_model=35, nhead=5, lr=0.001, step_size=5):
        super().__init__()
        self.model = nn.Transformer(d_model=d_model, nhead=nhead, batch_first=True)
        self.lr = lr
        self.step_size = step_size

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
    dm = FrankaDataModule(batch_size=32)
    model = TransformerModel(d_model=36, nhead=4, lr=0.001, step_size=5)

    trainer = pl.Trainer(max_epochs=5, accelerator="gpu", devices=2)
    trainer.fit(model, dm)
