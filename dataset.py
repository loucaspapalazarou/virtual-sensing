import os
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset

DATA_DIM = 36


class FrankaDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")
        ]
        self.index_map = []  # maps array index to (file, env)

        # Populate the index_map
        for file_idx, filename in enumerate(self.file_list):
            _, num_envs, _ = torch.load(filename).size()
            for env_idx in range(num_envs):
                self.index_map.append((file_idx, env_idx))

        os.environ["DATA_DIM"] = str(self.get_dim())

    def __getitem__(self, index):
        file_idx, env_idx = self.index_map[index]
        t = torch.load(self.file_list[file_idx])
        data = t[:, env_idx]

        # just to make feature dim even
        # helps with transformer nhead param
        timesteps, _ = data.size()
        data = torch.cat((data, torch.zeros(timesteps, 1).to(data.device)), dim=1)

        return data

    def __len__(self):
        return len(self.index_map)

    def get_dim(self):
        return self.__getitem__(0).size(1)


class FrankaDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, num_workers, data_portion):
        assert 0 < data_portion <= 1.0
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_portion = data_portion

    def setup(self, stage=None):
        dataset = FrankaDataset(data_dir=self.data_dir)
        data_points = int(len(dataset) * self.data_portion)
        print(f"Using {data_points}/{len(dataset)} data episodes")
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
