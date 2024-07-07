import torch
from mamba_ssm import Mamba, Mamba2  # type: ignore
import pytorch_lightning as pl


class MambaModule(pl.LightningModule):

    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        expand,
        headdim,
        lr,
        stride,
        window_size,
        prediction_distance,
        name,
    ):
        super().__init__()

        self.model = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.automatic_optimization = False
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

        # Manually optimizing
        optimizer = self.optimizers()

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

            # Backward pass and optimization step
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

        self.lr_schedulers().step()

        total_steps = (
            seq_len - (self.window_size + self.prediction_distance + 1)
        ) // self.stride
        avg_loss = total_loss / total_steps
        self.log("train_loss", avg_loss, sync_dist=True)
        return avg_loss

    def validation_step(self, batch, batch_idx):
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
            seq_len - (self.window_size + self.prediction_distance + 1)
        ) // self.stride

        avg_loss = total_loss / total_steps
        self.log("val_loss", avg_loss, sync_dist=True)
        return avg_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict(self, input):
        self.model.eval()
        with torch.no_grad():
            return self.model(input)
