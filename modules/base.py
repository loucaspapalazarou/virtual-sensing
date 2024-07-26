import pytorch_lightning as pl
import torch
from torch import nn
from modules.resnet import ResNetBlock
import torch.nn.functional as F


class BaseModelModule(pl.LightningModule):
    def __init__(
        self,
        d_model,
        start_lr,
        end_lr,
        window_size,
        prediction_distance,
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
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.window_size = window_size
        self.target_feature_indices = target_feature_indices
        self.prediction_distance = prediction_distance

        assert all(
            0 <= idx < self.d_model for idx in target_feature_indices
        ), "All target feature indices must be valid indices within d_model."

    def combine_sensor_and_camera_data(self, sensor_data, camera_data):
        resnet_features = self.resnet(camera_data)
        combined_data = torch.cat((sensor_data, resnet_features), dim=2)
        return combined_data

    def shared_step(self, batch, batch_idx, stage):
        sensor_data = batch["sensor_data"]
        camera_data = batch["camera_data"]

        combined_data = self.combine_sensor_and_camera_data(sensor_data, camera_data)
        batch_size, seq_len, input_size = combined_data.size()

        src = combined_data[:, : self.window_size]
        tgt = combined_data[:, -self.window_size :]

        output = self.forward(src, tgt)

        # Extract the target feature indices from both output and tgt
        output_idxs = output[:, :, self.target_feature_indices]
        target_idxs = tgt[:, :, self.target_feature_indices]

        return {
            stage + "/mse_loss": F.mse_loss(output_idxs, target_idxs),
            stage + "/l1_loss": F.l1_loss(output_idxs, target_idxs),
            stage + "/smooth_l1_loss": F.smooth_l1_loss(output_idxs, target_idxs),
        }

    def training_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch, batch_idx, stage="train")
        self.log_dict(
            loss_dict,
            sync_dist=True,
        )
        return loss_dict["train/mse_loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch, batch_idx, stage="val")
        self.log_dict(
            loss_dict,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr)

        def lr_lambda(current_step):
            total_steps = self.trainer.estimated_stepping_batches
            return (self.end_lr / self.start_lr) ** (current_step / total_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()
