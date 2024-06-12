import os
import torch
from torch.utils.data import Dataset, DataLoader

FRANKA_DOF = 9
DATA_ROOT = "/mnt/BigHD_1/loucas/IsaacGymEnvs/isaacgymenvs/force-sensor-data/"

def preprocess_tensor(t: torch.Tensor, degrees_of_freedom: int) -> torch.Tensor:
    return t.view(-1, t.size(1) // FRANKA_DOF, degrees_of_freedom)

class FrankaSensorDataset(Dataset):
    # field of view means how many points of the sensor can we see?
    def __init__(self, root_dir: str, degrees_of_freedom: int = FRANKA_DOF, field_of_view:int= 24):
        self.root_dir = root_dir
        self.degrees_of_freedom = degrees_of_freedom
        self.file_list = []
        
        for _, _, files in os.walk(self.root_dir):
            for name in files:
                if name.endswith('.pt'):  # Assuming the files are saved with .pt extension
                    self.file_list.append(os.path.join(self.root_dir, name))
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_path = self.file_list[idx]
        tensor = torch.load(file_path)
        processed_tensor = preprocess_tensor(tensor, self.degrees_of_freedom)
        
        return processed_tensor

# Instantiate the dataset
force_sensor_dataset = FrankaSensorDataset(DATA_ROOT)

# Create a DataLoader
batch_size = 4  # You can change this to any number you want
dataloader = DataLoader(force_sensor_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Example of iterating through the DataLoader
for batch in dataloader:
    print(batch.shape)

# TENSORS SHOULD BE LOADED VERTICALLY NOT HORIZONTALLY