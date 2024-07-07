import pytorch_lightning as pl
import torch

# TODO: Add prediction


class TransformerModule(pl.LightningModule):

    def __init__(
        self,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        lr,
        stride,
        window_size,
        prediction_distance,
        name,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )
        self.name = name
        self.d_model = d_model
        self.nhead = nhead
        self.lr = lr
        self.stride = stride
        self.window_size = window_size
        self.prediction_distance = prediction_distance

    def forward(self, src, tgt):
        return self.model(src, tgt)

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
            output = self(src, tgt)

            # Compute loss
            loss = torch.nn.functional.mse_loss(output, tgt)
            total_loss += loss

            # Backward pass and optimization step
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

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
            output = self(src, tgt)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
