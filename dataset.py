import os
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms


def _preprocess_image_tensor(t: torch.Tensor) -> torch.Tensor:
    t = t.permute(2, 0, 1)
    t = t[:3, :, :].float()
    preprocess = torch.nn.Sequential(
        transforms.Resize(256),
        transforms.CenterCrop(224),
    )
    return preprocess(t)


class FrankaDataset(Dataset):
    def __init__(self, data_dir, episode_length):
        self.file_list = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")
        ]
        self.index_map = []  # maps array index to (file, env)
        self.episode_length = episode_length

        # Populate the index_map
        for file_idx, filename in enumerate(self.file_list):
            _, num_envs, _ = torch.load(filename)["sensor_data"].size()
            for env_idx in range(num_envs):
                self.index_map.append((file_idx, env_idx))

    def __getitem__(self, index) -> dict:
        file_idx, env_idx = self.index_map[index]
        t = torch.load(self.file_list[file_idx])

        sensor_data = t["sensor_data"][:, env_idx]

        # image data shape torch.Size([300, 12, 3, 256, 256, 4])
        # becomes torch.Size([300, 3, 256, 256, 4]) after single-ing out an env
        camera_data = t["camera_data"][:, env_idx]
        processed_camera_data = torch.empty((300, 3, 3, 224, 224))
        for timestep in range(camera_data.shape[0]):  # Loop through each timestep
            for i in range(camera_data.shape[1]):  # Loop through each image
                image = camera_data[timestep, i]  # Extract the image
                processed_image = _preprocess_image_tensor(image)  # Apply preprocessing
                processed_camera_data[timestep, i] = (
                    processed_image  # Store in the output tensor
                )

        # Determine the appropriate episode length
        timesteps = min(self.episode_length, sensor_data.shape[0])

        # Truncate the sensor data and processed camera data to the episode length
        truncated_sensor_data = sensor_data[:timesteps]
        truncated_camera_data = processed_camera_data[:timesteps]

        return {
            "sensor_data": truncated_sensor_data,
            "camera_data": truncated_camera_data,
        }

    def __len__(self):
        return len(self.index_map)


class FrankaDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, num_workers, data_portion, episode_length):
        assert 0 < data_portion <= 1.0
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_portion = data_portion
        self.epidsode_length = episode_length

    def setup(self, stage=None):
        dataset = FrankaDataset(
            data_dir=self.data_dir, episode_length=self.epidsode_length
        )
        data_points = int(len(dataset) * self.data_portion)
        print(
            f"Using {data_points}/{len(dataset)} data episodes. Episode length: {self.epidsode_length}"
        )
        dataset = Subset(dataset, range(data_points))
        self.train_dataset, self.val_dataset = random_split(dataset, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    # def get_dim(self):
    #     with open(os.path.join(self.data_dir, "dim")) as f:
    #         dim = int(f.read())
    #         return dim if dim % 2 == 0 else dim + 1
