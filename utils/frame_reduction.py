import torch
import os

DATA_DIR = "/work/tc064/tc064/s2567498/data-w-camera/"


def half_frames(filename: str, keep_one_frame_per_n=2, min_frames=200):
    t = torch.load(filename)
    sensor_data: torch.Tensor = t["sensor_data"]
    camera_data: torch.Tensor = t["camera_data"]
    print(sensor_data.shape, camera_data.shape, end="\t")
    if sensor_data.size(0) < min_frames:
        return False
    t["sensor_data"] = sensor_data[::keep_one_frame_per_n]
    t["camera_data"] = camera_data[::keep_one_frame_per_n]
    torch.save(t, filename)
    return True


file_list = [
    os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pt")
]

for filename in file_list:
    try:
        changed = half_frames(filename)
        print(f"{"" if changed else "not"} changed {filename}")
    except Exception as e:
        print(f"Error on {filename}. {e}")
