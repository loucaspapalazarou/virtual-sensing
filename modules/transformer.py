import pytorch_lightning as pl
import torch
from torch import nn
from modules.resnet import ResNetBlock
from modules.utils import combine_sensor_and_camera_data


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
        resnet_checkpoint,
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
        self.resnet = ResNetBlock(
            out_features_per_image=resnet_features, resnet_checkpoint=resnet_checkpoint
        )
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
        output = self.model(src, tgt)
        return nn.functional.tanh(output)

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

    def predict(self, batch):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            sensor_data = batch["sensor_data"]
            camera_data = batch["camera_data"]

            combined_data = combine_sensor_and_camera_data(
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

            # Create a target tensor of zeros with the expected shape
            tgt = torch.zeros_like(src)

            # Create a mask for the tgt tensor
            tgt_mask = self.model.generate_square_subsequent_mask(self.window_size).to(
                src.device
            )

            # Forward pass
            output = self.model(src, tgt, tgt_mask=tgt_mask)

            # Extract the target feature indices from the output
            output_target = output[
                :, self.prediction_distance :, self.target_feature_indices
            ]

            return output_target
