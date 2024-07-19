import torch
import os
from tqdm import tqdm

DATA_DIR = "/mnt/BigHD_1/loucas/data-w-camera/"


def half_frames(filename: str, keep_one_frame_per_n=2, min_frames=200):
    t = torch.load(filename)
    sensor_data: torch.Tensor = t["sensor_data"]
    camera_data: torch.Tensor = t["camera_data"]
    if sensor_data.size(0) < min_frames:
        return
    t["sensor_data"] = sensor_data[::keep_one_frame_per_n]
    t["camera_data"] = camera_data[::keep_one_frame_per_n]
    torch.save(t, filename)


file_list = [
    os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pt")
]

for filename in tqdm(file_list, desc="Processing files"):
    half_frames(filename)
