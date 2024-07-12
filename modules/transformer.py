import pytorch_lightning as pl
import torch
from torch import nn
from modules.resnet import ResNetBlock


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
        target_feature_indices,
        resnet_features,
        name,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )
        self.resnet = ResNetBlock(out_features_per_image=resnet_features)
        self.name = name
        self.d_model = d_model
        self.nhead = nhead
        self.lr = lr
        self.stride = stride
        self.window_size = window_size
        self.prediction_distance = prediction_distance
        self.target_feature_indices = target_feature_indices

        assert all(
            0 <= idx < d_model for idx in target_feature_indices
        ), "All target feature indices must be valid indices within d_model."

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def process_data(self, sensor_data, camera_data):
        # Pass the entire camera data with timesteps through the ResNet block
        resnet_features = self.resnet(camera_data)
        # Concatenate along the feature dimension
        combined_data = torch.cat(
            (sensor_data, resnet_features), dim=2
        )  # Concatenate along the feature dimension
        return combined_data

    def training_step(self, batch, batch_idx):
        sensor_data = batch["sensor_data"]
        camera_data = batch["camera_data"]

        combined_data = self.process_data(sensor_data, camera_data)
        batch_size, seq_len, input_size = combined_data.size()
        total_loss = 0.0

        for i in range(
            0, seq_len - (self.window_size + self.prediction_distance + 1), self.stride
        ):
            src = combined_data[:, i : i + self.window_size, :]
            tgt = combined_data[
                :,
                i
                + self.prediction_distance : i
                + self.window_size
                + self.prediction_distance,
                :,
            ]

            output = self(src, tgt)

            # Extract the target feature indices from both output and tgt
            output_target = output[:, :, self.target_feature_indices]
            tgt_target = tgt[:, :, self.target_feature_indices]

            loss = nn.functional.mse_loss(output_target, tgt_target)
            total_loss += loss

        total_steps = (
            seq_len - (self.window_size + self.prediction_distance + 1)
        ) // self.stride
        avg_loss = total_loss / total_steps
        self.log("train_loss", avg_loss, sync_dist=True)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        sensor_data = batch["sensor_data"]
        camera_data = batch["camera_data"]

        combined_data = self.process_data(sensor_data, camera_data)
        batch_size, seq_len, input_size = combined_data.size()
        total_loss = 0.0

        for i in range(
            0, seq_len - (self.window_size + self.prediction_distance + 1), self.stride
        ):
            src = combined_data[:, i : i + self.window_size, :]
            tgt = combined_data[
                :,
                i
                + self.prediction_distance : i
                + self.window_size
                + self.prediction_distance,
                :,
            ]

            # Forward pass
            output = self(src, tgt)

            # Extract the target feature indices from both output and tgt
            output_target = output[:, :, self.target_feature_indices]
            tgt_target = tgt[:, :, self.target_feature_indices]

            # Compute loss
            loss = nn.functional.mse_loss(output_target, tgt_target)
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
        raise NotImplementedError()
