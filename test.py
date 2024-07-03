import torch
from mamba_ssm import Mamba

batch, length, dim = 16, 64, 36
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim,  # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,  # Local convolution width
    expand=2,  # Block expansion factor
).to("cuda")
y = model(x)
print(y)
print(model.layer_idx)
