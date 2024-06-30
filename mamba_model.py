import torch
from mamba_ssm import Mamba  # type: ignore
import pytorch_lightning as pl


class MambaModel(pl.LightningModule):

    def __init__(
        self, d_model, d_state, d_conv, expand, lr, stride, prediction_horizon
    ):
        super().__init__()
        self.model = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.lr = lr
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.save_hyperparameters()

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        batch_size, seq_len, input_size = batch.size()
        num_subsequences = (seq_len - self.prediction_horizon) // self.stride

        total_loss = 0.0
        for i in range(num_subsequences):
            start_idx = i * self.stride
            end_idx = start_idx + seq_len - self.prediction_horizon

            src = batch[:, start_idx:end_idx, :]
            tgt = batch[
                :,
                start_idx + self.prediction_horizon : end_idx + self.prediction_horizon,
                :,
            ]

            output = self(src, tgt)
            loss = torch.nn.functional.mse_loss(output, tgt)

            total_loss += loss

        avg_loss = total_loss / num_subsequences
        self.log("train_loss", avg_loss)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        batch_size, seq_len, input_size = batch.size()
        num_subsequences = (seq_len - self.prediction_horizon) // self.stride

        total_loss = 0.0
        for i in range(num_subsequences):
            start_idx = i * self.stride
            end_idx = start_idx + seq_len - self.prediction_horizon

            src = batch[:, start_idx:end_idx, :]
            tgt = batch[
                :,
                start_idx + self.prediction_horizon : end_idx + self.prediction_horizon,
                :,
            ]

            output = self(src, tgt)
            loss = torch.nn.functional.mse_loss(output, tgt)

            total_loss += loss

        avg_loss = total_loss / num_subsequences
        self.log("val_loss", avg_loss)
        return avg_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
