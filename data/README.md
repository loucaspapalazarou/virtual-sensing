# Data

## General

The data was stored in a directory `data/` like this one, and included the `metadata.json` file. However, the files are ~2.7G so I will not upload them.

Each data file contains a Python dictionary with two keys, `sensor_data` and `camera_data` with their respective values being tensors of shape `[300, 12, 35]` and `[300, 12, 3, 256, 256, 4]`.

```python
# data-file.pt
{
    "sensor_data": tensor.Size([300, 12, 35]),
    "camera_data": tensor.Size([300, 12, 3, 256, 256, 4])
}
```

The first dimension (300) in `sensor_data` and `camera_data` refers to the simulation samples through time. Although Isaac Gym operates at 60Hz, our specific task environment is set to sample rate to 6Hz, meaning that each environment yields 6 samples per second. Each episode is run for 50 seconds, resulting in a total of 300 samples.

The second dimension (12) denotes the number of environments per episode. Isaac Gym (the simulation software) allows for multiple isolated environments to be run concurrently inside the same episode in order to reduce the rendering overhead. We leverage this feature for the collection of our data by running 12 environments per episode.

The remaining dimensions refer to the specific observations and are covered in the subsequent sections.

## Metadata

The `metadata.json` file contains key descriptive information about the data files. By storing this metadata separately, it enables quicker model creation, as it eliminates the need to repeatedly load and extract information directly from the data files. This approach significantly improves efficiency and speeds up the process.

- `dim`: The data dimension, how many features each timestep produces
- `envs`: How many environments were simulated for each data file
- `steps`: How many timesteps the simulation was run for

