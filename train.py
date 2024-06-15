import torchaudio
import torch
from torchaudio.models import WaveRNN
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt

# Initialize WaveRNN model
wavernn = WaveRNN(upsample_scales=[5, 5, 8], n_classes=512, hop_length=200)

# Load the audio file
file = "ahh.mp3"

waveform, sample_rate = torchaudio.load(file, normalize=True)

wavernn = WaveRNN(upsample_scales=[5,5,8], n_classes=512, hop_length=200)
waveform, sample_rate = torchaudio.load(file)
# waveform shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length)
specgram = MelSpectrogram(sample_rate)(waveform)  # shape: (n_batch, n_channel, n_freq, n_time)
output = wavernn(waveform, specgram)
# output shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length, n_classes)
