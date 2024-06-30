import pytorch_lightning as pl
import torch


class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        name,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        lr,
        stride,
        prediction_horizon,
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
        self.d_model = d_model
        self.nhead = nhead
        self.lr = lr
        self.stride = stride
        self.prediction_horizon = prediction_horizon

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
