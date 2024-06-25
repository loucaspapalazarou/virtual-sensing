import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from dataset import FrankaDataset
from constants import DEVICE

dataset = FrankaDataset()  # You can replace this with your actual dataset

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Transformer model parameters
d_model = 36
nhead = 4
model = nn.Transformer(d_model=d_model, nhead=nhead, batch_first=True).to(DEVICE)

# Loss function and optimizer (not included in the provided snippet)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step size for skipping timesteps
step_size = 5  # Adjust this as per your requirement

# Training loop
for epoch in range(5):  # Example: Training for 5 epochs
    print(f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(train_loader):
        batch_size, seq_len, input_size = batch.size()

        # Determine the number of subsequences we can extract
        num_subsequences = (seq_len - 1) // step_size

        # Iterate over subsequences
        batch_loss = 0.0
        for i in range(num_subsequences):
            start_idx = i * step_size
            end_idx = start_idx + seq_len - 1

            # Extract subsequences with step size
            src = batch[:, start_idx:end_idx, :]
            tgt = batch[:, start_idx + 1 : end_idx + 1, :]

            # Forward pass
            output = model(src=src, tgt=tgt)

            # Compute loss (example with MSE loss)
            loss = criterion(output, tgt)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        print(
            f"Batch {batch_idx}/{len(train_loader)} average loss: {batch_loss/num_subsequences}"
        )

print("Training finished.")
