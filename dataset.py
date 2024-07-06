import os
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset


class FrankaDataset(Dataset):
    def __init__(self, data_dir, episode_length):
        self.file_list = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")
        ]
        self.index_map = []  # maps array index to (file, env)
        self.episode_length = episode_length

        # Populate the index_map
        for file_idx, filename in enumerate(self.file_list):
            _, num_envs, _ = torch.load(filename).size()
            for env_idx in range(num_envs):
                self.index_map.append((file_idx, env_idx))

    def __getitem__(self, index):
        file_idx, env_idx = self.index_map[index]
        t = torch.load(self.file_list[file_idx])
        data = t[:, env_idx]

        # Make feature dim even if it's not already
        timesteps, features = data.size()
        if features % 2 != 0:
            data = torch.cat((data, torch.zeros(timesteps, 1).to(data.device)), dim=1)

        return data[0 : min(self.episode_length, timesteps), :]

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

    def get_dim(self):
        with open(os.path.join(self.data_dir, "dim")) as f:
            dim = int(f.read())
            return dim if dim % 2 == 0 else dim + 1
