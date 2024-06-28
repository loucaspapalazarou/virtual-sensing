import os
import torch
from torch.utils.data import Dataset


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
