import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import FrankaDataset
import sys
import argparse
from lightning.pytorch.loggers import WandbLogger


class FrankaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = FrankaDataset(
            data_dir=self.data_dir
        )  # Replace with your dataset initialization
        self.train_dataset, self.val_dataset = random_split(
            dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))]
        )

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


class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        d_model=35,
        nhead=5,
        lr=0.001,
        step_size=5,
    ):
        super().__init__()
        self.model = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=256,
        )
        self.d_model = d_model
        self.nhead = nhead
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


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--d_model", type=int, default=36, help="Dimension of the model"
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of heads in the multiheadattention models",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--step_size", type=int, default=4, help="Step size for subsequence processing"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum number of epochs to train for",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="transformer",
        help="Project name for WandbLogger",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/",
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()

    dm = FrankaDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = TransformerModel(
        d_model=args.d_model, nhead=args.nhead, lr=args.lr, step_size=args.step_size
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Monitor validation loss
        dirpath=args.checkpoint_dir,  # Directory to save checkpoints
        filename=f"dmodel-{args.d_model}_nhead-{args.nhead}_lr-{args.lr}_stepsize-{args.step_size}"
        + "-{epoch:02d}-{val_loss:.2f}",  # Format for checkpoint filenames
        save_top_k=-1,  # Save all checkpoints
        mode="min",  # Save the model with the lowest validation loss
    )

    wandb_logger = WandbLogger(project=args.project_name)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, callbacks=[checkpoint_callback], logger=wandb_logger
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
