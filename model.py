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

# Adjust MelSpectrogram parameters
n_fft = 1024  # Increase FFT size
n_mels = 80   # Decrease number of mel bins

specgram = MelSpectrogram(sample_rate, n_fft=n_fft, n_mels=n_mels)(waveform)

print(specgram)
