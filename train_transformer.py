import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import FrankaDataset
import argparse

DATA_DIM = 36


class FrankaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = FrankaDataset(data_dir=self.data_dir)
        self.data_dim = dataset.get_dim()
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


class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        lr,
        stride,
    ):
        super().__init__()
        self.model = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )
        self.d_model = d_model
        self.nhead = nhead
        self.lr = lr
        self.stride = stride
        self.save_hyperparameters()

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        batch_size, seq_len, input_size = batch.size()
        num_subsequences = (seq_len - 1) // self.stride

        total_loss = 0.0
        for i in range(num_subsequences):
            start_idx = i * self.stride
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
        num_subsequences = (seq_len - 1) // self.stride

        total_loss = 0.0
        for i in range(num_subsequences):
            start_idx = i * self.stride
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
        default="../data",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--num_encoder_layers", type=int, default=3, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=3, help="Number of decoder layers"
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of heads in the multiheadattention models",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--stride", type=int, default=4, help="Step size for subsequence processing"
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
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=False,
        help="Log in wandb",
    )

    args = parser.parse_args()

    dm = FrankaDataModule(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    dm.save_hyperparameters()

    model = TransformerModel(
        d_model=DATA_DIM,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=256,
        lr=args.lr,
        stride=args.stride,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.checkpoint_dir,
        filename="dmodel-{d_model}_nhead-{nhead}_lr-{lr}_stepsize-{stride}-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",  # Adjust as per needed operation
    )

    tensorboard_logger = TensorBoardLogger("tb_logs", name="transformer")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=[tensorboard_logger],
    )
    if args.use_wandb:
        trainer.loggers.append(WandbLogger(project=args.project_name))
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
