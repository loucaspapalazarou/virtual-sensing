import torch
from mamba_ssm import Mamba, Mamba2  # type: ignore
import pytorch_lightning as pl


class MambaModel(pl.LightningModule):

    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        expand,
        lr,
        stride,
        window_size,
        prediction_distance,
        name,
    ):
        super().__init__()
        match name:
            case "mamba":
                self.model = Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=d_model,  # Model dimension d_model
                    d_state=d_state,  # SSM state expansion factor
                    d_conv=d_conv,  # Local convolution width
                    expand=expand,  # Block expansion factor
                )
            case "mamba2":
                self.model = Mamba2(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=d_model,  # Model dimension d_model
                    d_state=d_state,  # SSM state expansion factor
                    d_conv=d_conv,  # Local convolution width
                    expand=expand,  # Block expansion factor
                )
            case _:
                raise ValueError(f"Model '{name}' not recognized")
        self.lr = lr
        self.stride = stride
        self.window_size = window_size
        self.prediction_distance = prediction_distance
        self.save_hyperparameters()

    def forward(self, src):
        return self.model(src)

    def training_step(self, batch, batch_idx):
        batch_size, seq_len, input_size = batch.size()
        total_loss = 0.0

        for i in range(
            0, seq_len - (self.window_size + self.prediction_distance + 1), self.stride
        ):

            src = batch[:, i : i + self.window_size, :]
            tgt = batch[
                :,
                i
                + self.prediction_distance : i
                + self.window_size
                + self.prediction_distance,
                :,
            ]

            # Forward pass
            output = self(src)

            # Compute loss
            loss = torch.nn.functional.mse_loss(output, tgt)
            total_loss += loss

        total_steps = (
            seq_len - (self.window_size + self.prediction_distance + 1) // self.stride
        )
        avg_loss = total_loss / total_steps
        self.log("train_loss", avg_loss, sync_dist=True)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        batch_size, seq_len, input_size = batch.size()
        total_loss = 0.0

        for i in range(
            0, seq_len - (self.window_size + self.prediction_distance + 1), self.stride
        ):
            # Extract source and target sequences

            src = batch[:, i : i + self.window_size, :]
            tgt = batch[
                :,
                i
                + self.prediction_distance : i
                + self.window_size
                + self.prediction_distance,
                :,
            ]

            # Forward pass
            output = self(src)

            # Compute loss
            loss = torch.nn.functional.mse_loss(output, tgt)
            total_loss += loss

        total_steps = (
            seq_len - (self.window_size + self.prediction_distance + 1) // self.stride
        )

        avg_loss = total_loss / total_steps
        self.log("val_loss", avg_loss, sync_dist=True)
        return avg_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
