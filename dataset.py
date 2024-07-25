import os
import json
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms


def _preprocess_image_tensor(t: torch.Tensor) -> torch.Tensor:
    # bring channels to front
    t = t.permute(2, 0, 1)[:3, :, :]
    preprocess = torch.nn.Sequential(
        transforms.Resize(256),
        transforms.CenterCrop(224),
    )
    return preprocess(t)


class FrankaDataset(Dataset):
    def __init__(
        self,
        data_dir,
        episode_length,
        window_size,
        stride,
        prediction_distance,
        use_cpu,
    ):
        self.file_list = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")
        ]
        self.index_map = []  # maps array index to (file, env, step)
        self.episode_length = episode_length
        self.use_cpu = use_cpu
        self.data_dir = data_dir
        self.map_location = torch.device("cpu") if use_cpu else None
        self.curr_file_idx = -1
        self.curr_file_data = None
        self.window_size = window_size
        self.stride = stride
        self.prediction_distance = prediction_distance
        self.image_preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )

        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        assert 0 < episode_length <= self.metadata["steps"]

        if self.file_list:
            # Populate the index_map
            for file_idx, filename in enumerate(self.file_list):
                for env_idx in range(self.metadata["envs"]):
                    last_step = episode_length - (
                        self.window_size + self.prediction_distance + 1
                    )
                    for step_idx in range(
                        0,
                        last_step,
                        self.stride,
                    ):
                        self.index_map.append((file_idx, env_idx, step_idx))

    def __getitem__(self, index) -> dict:
        file_idx, env_idx, step_idx = self.index_map[index]

        # if we're done with a file, load the next into memory
        if file_idx != self.curr_file_idx:
            self.curr_file_data = torch.load(
                self.file_list[file_idx], map_location=self.map_location
            )

        sensor_data = self.curr_file_data["sensor_data"][
            step_idx : step_idx + self.window_size + self.prediction_distance, env_idx
        ]
        camera_data = self.curr_file_data["camera_data"][
            step_idx : step_idx + self.window_size + +self.prediction_distance, env_idx
        ]

        # remove alpha channel
        camera_data = camera_data[..., :3]
        # bring channel dim to front of image dims
        camera_data = camera_data.permute(0, 1, 4, 2, 3)
        # transforms
        camera_data = self.image_preprocess(camera_data)

        return {
            "sensor_data": sensor_data,
            "camera_data": camera_data,
        }

    def __len__(self):
        return len(self.index_map)


class FrankaDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        data_portion,
        episode_length,
        window_size,
        stride,
        prediction_distance,
        use_cpu=False,
    ):
        assert 0 < data_portion <= 1.0
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_portion = data_portion
        self.epidsode_length = episode_length
        self.window_size = window_size
        self.stride = stride
        self.prediction_distance = prediction_distance
        self.use_cpu = use_cpu

    def setup(self, stage=None):
        dataset = FrankaDataset(
            data_dir=self.data_dir,
            episode_length=self.epidsode_length,
            window_size=self.window_size,
            stride=self.stride,
            prediction_distance=self.prediction_distance,
            use_cpu=self.use_cpu,
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

    def get_num_sensor_features(self):
        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            return metadata["dim"]
