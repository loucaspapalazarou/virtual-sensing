import torch
import torchvision
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, out_features_per_image) -> None:
        super().__init__()
        resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        for param in resnet.parameters():
            param.requires_grad = False
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, out_features_per_image)
        self.resnet = resnet

    def forward(self, x):
        # Assuming x is of shape [batch_size, 3, 3, 224, 224]
        batch_size, num_images, channels, height, width = x.shape
        outputs = []
        for i in range(num_images):
            image = x[
                :, i, :, :, :
            ]  # Extract each image of shape [batch_size, 3, 224, 224]
            output = self.resnet(image)  # Pass through ResNet
            outputs.append(output)
        # Concatenate the outputs along the feature dimension
        return torch.cat(outputs, dim=1)
