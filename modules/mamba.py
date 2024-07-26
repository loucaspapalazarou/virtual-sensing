from mamba_ssm import Mamba  # type: ignore
from modules.base import BaseModelModule
import torch


class MambaModule(BaseModelModule):
    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        expand,
        start_lr,
        end_lr,
        activation,
        window_size,
        target_feature_indices,
        prediction_distance,
        resnet_features,
        resnet_checkpoint,
        name,
    ):
        super().__init__(
            d_model=d_model,
            name=name,
            start_lr=start_lr,
            end_lr=end_lr,
            window_size=window_size,
            prediction_distance=prediction_distance,
            target_feature_indices=target_feature_indices,
            resnet_features=resnet_features,
            resnet_checkpoint=resnet_checkpoint,
        )
        self.activation = activation
        self.model = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, src, tgt=None):
        output = self.model(src)
        if self.activation == "tanh":
            return torch.tanh(output)
        return output

    def predict(self, batch):
        # TODO: zero-out/mask targets
        self.eval()
        with torch.no_grad():
            sensor_data = batch["sensor_data"]
            camera_data = batch["camera_data"]

            combined_data = self.combine_sensor_and_camera_data(
                sensor_data, camera_data
            )

            batch_size, seq_len, input_size = combined_data.size()

            start_index = seq_len - self.window_size

            if start_index < 0:
                raise ValueError(
                    "The sequence length is too short for the given window size and prediction distance."
                )

            src = combined_data[:, start_index : start_index + self.window_size, :]

            output = self.forward(src)

            return output[:, -self.prediction_distance :, self.target_feature_indices]
