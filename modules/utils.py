import torch


def combine_sensor_and_camera_data(
    image_block: torch.nn.Module, sensor_data, camera_data
):
    # Pass the entire camera data with timesteps through the ResNet block
    resnet_features = image_block(camera_data)
    # Concatenate along the feature dimension
    combined_data = torch.cat(
        (sensor_data, resnet_features), dim=2
    )  # Concatenate along the feature dimension
    return combined_data
