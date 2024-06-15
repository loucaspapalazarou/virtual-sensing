import os
import torch
from torch.utils.data import Dataset, DataLoader

FRANKA_DOF = 9
DATA_DIR = "/mnt/BigHD_1/loucas/force-sensor-data/"
MAX_TIMESTEPS = 300

class FrankaSensorDataset(Dataset):
    """
    Custom dataset for loading force sensor data from .pt files.

    Each .pt file represents an episode and contains a tensor with the shape [episode_steps, franka_dof * num_envs].
    Each row in the tensor corresponds to a timestep in the episode, and the tensor is divided into environments.
    The dataset provides sequences of timesteps (defined by focal_length) for a specific environment.

    Attributes:
        focal_length (int): Number of timesteps to include in each data point.
        data_dir (str): Directory containing the .pt files.
        franka_dof (int): Degrees of freedom for the Franka robot (default is 9).
        file_list (list): List of paths to the .pt files.
        index_map (list): List of tuples mapping file index, environment index, and timestep index.
    """
    
    def __init__(self, data_dir: str, franka_dof: int = FRANKA_DOF, focal_length=24):
        """
        Initialize the dataset with the given directory, degrees of freedom, and focal length.

        Args:
            data_dir (str): Path to the directory containing .pt files.
            franka_dof (int): Degrees of freedom for the Franka robot.
            focal_length (int): Number of timesteps to include in each data point.
        """
        assert focal_length <= MAX_TIMESTEPS, "Focal length must be less than or equal to MAX_TIMESTEPS."
        self.focal_length = focal_length
        self.data_dir = data_dir
        self.franka_dof = franka_dof
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.index_map = []
        
        # Create an index map to manage the dynamic sizes
        for file_idx, file_path in enumerate(self.file_list):
            tensor = torch.load(file_path)
            num_timesteps = tensor.shape[0]
            num_data_points = tensor.shape[1]
            num_environments = num_data_points // self.franka_dof
            for env in range(num_environments):
                for t in range(num_timesteps):
                    self.index_map.append((file_idx, env, t))

    def __len__(self):
        """
        Return the total number of data points in the dataset.

        Returns:
            int: Total number of data points.
        """
        return len(self.index_map)
    
    def __getitem__(self, idx):
        """
        Retrieve a data point from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            torch.Tensor: Tensor of shape [focal_length, franka_dof] representing the data point.
        """
        file_idx, environment, timestep = self.index_map[idx]
        tensor = torch.load(self.file_list[file_idx])
        
        # Ensure that the focal length does not exceed the number of timesteps available
        start_timestep = timestep
        end_timestep = min(timestep + self.focal_length, tensor.shape[0])
        
        start_index = environment * self.franka_dof
        end_index = start_index + self.franka_dof
        data_points_list = []
        for t in range(start_timestep, end_timestep):
            data_points = tensor[t, start_index:end_index]
            data_points_list.append(data_points)
        
        # Stack data points to create a tensor of shape [focal_length, franka_dof]
        data_points_tensor = torch.stack(data_points_list)
        return data_points_tensor
    

# Instantiate the dataset
force_sensor_dataset = FrankaSensorDataset(DATA_DIR, focal_length=2)

# Create a DataLoader
dataloader = DataLoader(dataset=force_sensor_dataset, batch_size=2, shuffle=True)

# Iterate over the DataLoader
for item in dataloader:
    print(item)
    break
