from dataset import FrankaDataset

ds = FrankaDataset(
    data_dir="/work/tc064/tc064/s2567498/data-w-camera/",
    episode_length=150,
    window_size=10,
    stride=1,
    prediction_distance=2,
    use_cpu=True,
)

for batch in ds:
    print(batch["camera_data"].shape)
    break
