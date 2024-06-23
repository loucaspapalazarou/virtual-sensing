import torch

DATA_DIR = "/mnt/BigHD_1/loucas/data/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
