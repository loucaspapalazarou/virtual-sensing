import pytorch_lightning as pl
import torch
from torch import nn
from modules.utils import combine_sensor_and_camera_data, ResNetBlock
from modules.base import BaseModelModule


class TransformerModule(BaseModelModule):
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
        super().__init__(
            lr,
            stride,
            window_size,
            prediction_distance,
            target_feature_indices,
            resnet_features,
            resnet_checkpoint,
            name,
        )
        self.model = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, src, tgt):
        output = self.model(src, tgt)
        return nn.functional.tanh(output)

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

            return output[:, -self.prediction_distance :, self.target_feature_indices]
