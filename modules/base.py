import pytorch_lightning as pl
import torch
from torch import nn
from modules.resnet import ResNetBlock


class BaseModelModule(pl.LightningModule):
    def __init__(
        self,
        d_model,
        lr,
        window_size,
        target_feature_indices,
        resnet_features,
        resnet_checkpoint,
        name,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.resnet = ResNetBlock(
            out_features_per_image=resnet_features, resnet_checkpoint=resnet_checkpoint
        )
        self.name = name
        self.d_model = d_model
        self.lr = lr
        self.window_size = window_size
        # self.prediction_distance = prediction_distance
        self.target_feature_indices = target_feature_indices

        assert all(
            0 <= idx < self.d_model for idx in target_feature_indices
        ), "All target feature indices must be valid indices within d_model."

    def forward(self, src, tgt=None):
        raise NotImplementedError

    def combine_sensor_and_camera_data(self, sensor_data, camera_data):
        resnet_features = self.resnet(camera_data)
        combined_data = torch.cat((sensor_data, resnet_features), dim=2)
        return combined_data

    def shared_step(self, batch, batch_idx):
        sensor_data = batch["sensor_data"]
        camera_data = batch["camera_data"]

        combined_data = self.combine_sensor_and_camera_data(sensor_data, camera_data)
        batch_size, seq_len, input_size = combined_data.size()
        total_loss = 0.0

        src = combined_data[:, : self.window_size]
        tgt = combined_data[:, -self.window_size :]

        output = self(src, tgt)

        # Extract the target feature indices from both output and tgt
        output_target = output[:, :, self.target_feature_indices]
        tgt_target = tgt[:, :, self.target_feature_indices]

        loss = nn.functional.mse_loss(output_target, tgt_target)
        return loss

    def training_step(self, batch, batch_idx):
        avg_loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", avg_loss, sync_dist=True)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        avg_loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", avg_loss, sync_dist=True)
        return avg_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict(self, batch):
        # TODO: When predicting, zero out targets
        # Set the model to evaluation mode
        self.eval()
        with torch.no_grad():  # Disable gradient calculation
            sensor_data = batch["sensor_data"]
            camera_data = batch["camera_data"]

            combined_data = self.combine_sensor_and_camera_data(
                self.resnet, sensor_data, camera_data
            )

            batch_size, seq_len, input_size = combined_data.size()

            # Determine the starting index for the last window
            start_index = seq_len - self.window_size

            # Ensure the start index is non-negative
            if start_index < 0:
                raise ValueError(
                    "The sequence length is too short for the given window size and prediction distance."
                )

            # Prepare the source tensor for the last window
            src = combined_data[:, start_index : start_index + self.window_size, :]

            output = self.forward(src)

            return output[:, -self.prediction_distance :, self.target_feature_indices]
