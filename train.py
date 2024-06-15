import torchaudio
import torch
from torchaudio.models import WaveRNN
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt
from dataset import FrankaSensorDataset
from torch.utils.data import DataLoader
from constants import DATA_DIR

# Instantiate the dataset
force_sensor_dataset = FrankaSensorDataset(DATA_DIR, focal_length=25)

# Create a DataLoader
dataloader = DataLoader(dataset=force_sensor_dataset, batch_size=64, shuffle=True)

# Iterate over the DataLoader
for item in dataloader:
    print(item)
    break


# # Initialize WaveRNN model
# wavernn = WaveRNN(upsample_scales=[5, 5, 8], n_classes=512, hop_length=200)

# # Load the audio file
# file = "ahh.mp3"

# waveform, sample_rate = torchaudio.load(file, normalize=True)

# wavernn = WaveRNN(upsample_scales=[5,5,8], n_classes=512, hop_length=200)
# waveform, sample_rate = torchaudio.load(file)
# # waveform shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length)
# specgram = MelSpectrogram(sample_rate)(waveform)  # shape: (n_batch, n_channel, n_freq, n_time)
# output = wavernn(waveform, specgram)
# # output shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length, n_classes)
