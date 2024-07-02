import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
from WaveNet.wavenet.networks import WaveNet

import torch
from torch.utils.data import Dataset, DataLoader

data_loader = DataLoader(torch.rand(1000, 300, 32), batch_size=32, shuffle=True)

# Initialize WaveNet model with adjusted parameters
layer_size = 5
stack_size = 4
in_channels = 32
res_channels = 64
wavenet = WaveNet(layer_size, stack_size, in_channels, res_channels).cuda()

t = next(iter(data_loader))
t = t.cuda()

output = wavenet(t)

print(output.shape)

exit()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, targets in data_loader:
        inputs = inputs.unsqueeze(-1)  # Add channel dimension
        targets = targets.unsqueeze(-1)  # Add channel dimension
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        loss = wavenet.train(inputs, targets)
        epoch_loss += loss
    
    avg_loss = epoch_loss / len(data_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Save model
model_dir = './model'
wavenet.save(model_dir)
