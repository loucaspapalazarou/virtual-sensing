import torch
from mamba_ssm import Mamba  # type: ignore
import pytorch_lightning as pl
from modules.resnet import ResNetBlock
from modules.utils import combine_sensor_and_camera_data


class MambaModule(pl.LightningModule):

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
        target_feature_indices,
        resnet_features,
        name,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.resnet = ResNetBlock(out_features_per_image=resnet_features)
        self.name = name
        self.automatic_optimization = False
        self.lr = lr
        self.stride = stride
        self.window_size = window_size
        self.prediction_distance = prediction_distance
        self.target_feature_indices = target_feature_indices

        assert all(
            0 <= idx < d_model for idx in target_feature_indices
        ), "All target feature indices must be valid indices within d_model."

    def forward(self, src):
        return self.model(src)

    def training_step(self, batch, batch_idx):
        sensor_data = batch["sensor_data"]
        camera_data = batch["camera_data"]

        combined_data = combine_sensor_and_camera_data(
            self.resnet, sensor_data, camera_data
        )

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
            output = self(src)

            # Extract the target feature indices from both output and tgt
            output_target = output[:, :, self.target_feature_indices]
            tgt_target = tgt[:, :, self.target_feature_indices]

            # Compute loss
            loss = torch.nn.functional.mse_loss(output_target, tgt_target)
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

        combined_data = combine_sensor_and_camera_data(
            self.resnet, sensor_data, camera_data
        )
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
            output = self(src)

            # Extract the target feature indices from both output and tgt
            output_target = output[:, :, self.target_feature_indices]
            tgt_target = tgt[:, :, self.target_feature_indices]

            # Compute loss
            loss = torch.nn.functional.mse_loss(output_target, tgt_target)
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
