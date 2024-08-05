import os
from dotenv import load_dotenv
import glob

load_dotenv()
WORK_DIR = os.getenv("WORK_DIR")

# Define paths for MAMBA
MAMBA_SMALL_PATH = os.path.join(
    WORK_DIR, "virtual-sensing/lightning_logs/mamba/config-exp1-200k.json/version_0"
)
MAMBA_LARGE_PATH = os.path.join(
    WORK_DIR, "virtual-sensing/lightning_logs/mamba/config-exp2-600k.json/version_0"
)
MAMBA_HUGE_PATH = os.path.join(
    WORK_DIR, "virtual-sensing/lightning_logs/mamba/config-exp3-10M.json/version_0"
)

# Define paths for RNN
RNN_SMALL_PATH = os.path.join(
    WORK_DIR, "virtual-sensing/lightning_logs/rnn/config-exp1-200k.json/version_0"
)
RNN_LARGE_PATH = os.path.join(
    WORK_DIR, "virtual-sensing/lightning_logs/rnn/config-exp2-600k.json/version_0"
)
RNN_HUGE_PATH = os.path.join(
    WORK_DIR, "virtual-sensing/lightning_logs/rnn/config-exp3-10M.json/version_0"
)

# Define paths for TRANSFORMER
TRANSFORMER_SMALL_PATH = os.path.join(
    WORK_DIR,
    "virtual-sensing/lightning_logs/transformer/config-exp1-200k.json/version_0",
)
TRANSFORMER_LARGE_PATH = os.path.join(
    WORK_DIR,
    "virtual-sensing/lightning_logs/transformer/config-exp2-600k.json/version_0",
)
TRANSFORMER_HUGE_PATH = os.path.join(
    WORK_DIR,
    "virtual-sensing/lightning_logs/transformer/config-exp3-10M.json/version_0",
)

# MAMBA
MAMBA_SMALL_PARAMS = os.path.join(MAMBA_SMALL_PATH, "hparams.yaml")
MAMBA_SMALL_CHECKPOINT = os.path.join(
    MAMBA_SMALL_PATH, "checkpoints/epoch=0-step=2900.ckpt"
)
MAMBA_SMALL_LOGS = glob.glob(os.path.join(MAMBA_SMALL_PATH, "*tfevents*"))[0]

MAMBA_LARGE_PARAMS = os.path.join(MAMBA_LARGE_PATH, "hparams.yaml")
MAMBA_LARGE_CHECKPOINT = os.path.join(
    MAMBA_LARGE_PATH, "checkpoints/epoch=0-step=2900.ckpt"
)
MAMBA_LARGE_LOGS = glob.glob(os.path.join(MAMBA_LARGE_PATH, "*tfevents*"))[0]

MAMBA_HUGE_PARAMS = os.path.join(MAMBA_HUGE_PATH, "hparams.yaml")
MAMBA_HUGE_CHECKPOINT = os.path.join(
    MAMBA_HUGE_PATH, "checkpoints/epoch=0-step=2900.ckpt"
)
MAMBA_HUGE_LOGS = glob.glob(os.path.join(MAMBA_HUGE_PATH, "*tfevents*"))[0]

# RNN
RNN_SMALL_PARAMS = os.path.join(RNN_SMALL_PATH, "hparams.yaml")
RNN_SMALL_CHECKPOINT = os.path.join(
    RNN_SMALL_PATH, "checkpoints/epoch=0-step=2900.ckpt"
)
RNN_SMALL_LOGS = glob.glob(os.path.join(RNN_SMALL_PATH, "*tfevents*"))[0]

RNN_LARGE_PARAMS = os.path.join(RNN_LARGE_PATH, "hparams.yaml")
RNN_LARGE_CHECKPOINT = os.path.join(
    RNN_LARGE_PATH, "checkpoints/epoch=0-step=2900.ckpt"
)
RNN_LARGE_LOGS = glob.glob(os.path.join(RNN_LARGE_PATH, "*tfevents*"))[0]

RNN_HUGE_PARAMS = os.path.join(RNN_HUGE_PATH, "hparams.yaml")
RNN_HUGE_CHECKPOINT = os.path.join(RNN_HUGE_PATH, "checkpoints/epoch=0-step=2900.ckpt")
RNN_HUGE_LOGS = glob.glob(os.path.join(RNN_HUGE_PATH, "*tfevents*"))[0]

# TRANSFORMER
TRANSFORMER_SMALL_PARAMS = os.path.join(TRANSFORMER_SMALL_PATH, "hparams.yaml")
TRANSFORMER_SMALL_CHECKPOINT = os.path.join(
    TRANSFORMER_SMALL_PATH, "checkpoints/epoch=0-step=2900.ckpt"
)
TRANSFORMER_SMALL_LOGS = glob.glob(os.path.join(TRANSFORMER_SMALL_PATH, "*tfevents*"))[
    0
]

TRANSFORMER_LARGE_PARAMS = os.path.join(TRANSFORMER_LARGE_PATH, "hparams.yaml")
TRANSFORMER_LARGE_CHECKPOINT = os.path.join(
    TRANSFORMER_LARGE_PATH, "checkpoints/epoch=0-step=2900.ckpt"
)
TRANSFORMER_LARGE_LOGS = glob.glob(os.path.join(TRANSFORMER_LARGE_PATH, "*tfevents*"))[
    0
]

TRANSFORMER_HUGE_PARAMS = os.path.join(TRANSFORMER_HUGE_PATH, "hparams.yaml")
TRANSFORMER_HUGE_CHECKPOINT = os.path.join(
    TRANSFORMER_HUGE_PATH, "checkpoints/epoch=0-step=2900.ckpt"
)
TRANSFORMER_HUGE_LOGS = glob.glob(os.path.join(TRANSFORMER_HUGE_PATH, "*tfevents*"))[0]
