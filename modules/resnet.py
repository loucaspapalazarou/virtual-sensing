import torch
import torchvision
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, out_features_per_image, resnet_checkpoint: str) -> None:
        super().__init__()
        # resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        resnet = torchvision.models.resnet18(weights=None)
        resnet.load_state_dict(torch.load(resnet_checkpoint))
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Linear(resnet.fc.in_features, out_features_per_image)
        self.resnet = resnet

    def forward(self, x):
        # Assuming x is of shape [batch_size, num_timesteps, num_images, channels, height, width]
        batch_size, num_timesteps, num_images, channels, height, width = x.shape

        # Reshape the input to [batch_size * num_timesteps * num_images, channels, height, width]
        x = x.view(batch_size * num_timesteps * num_images, channels, height, width)

        # Pass through ResNet
        outputs = self.resnet(x)

        # Reshape back to [batch_size, num_timesteps, num_images, -1]
        outputs = outputs.view(batch_size, num_timesteps, num_images, -1)

        # Concatenate the outputs of all images for each timestep
        outputs = outputs.permute(
            0, 1, 3, 2
        ).contiguous()  # [batch_size, num_timesteps, -1, num_images]
        outputs = outputs.view(
            batch_size, num_timesteps, -1
        )  # [batch_size, num_timesteps, num_images * -1]

        return outputs
