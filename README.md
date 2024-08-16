# Virtual Sensing through Transformers and Selective State Spaces
***MSc Dissertation | The University of Edinburgh***

[Final Dissertation Document](Virtual_Sensing.pdf)

## Introduction

This is the repository that holds all of the code for my dissertation.

...

### Abstract

This dissertation investigates the application of modern machine learning techniques,
specifically Transformers and Selective State Spaces (Mamba), to replicate and replace
physical sensors in robotic systems through virtual sensing. The primary objective is to
develop models capable of inferring sensor outputs from a subset of sensors, thereby
reducing the number of physical sensors needed and lowering associated costs. The
study involves setting up a simulation environment using the Franka Emika Panda robot
to perform a defined task, generating data on various measurements including positions,
orientations, force sensor outputs, and images. Multiple experiments were conducted to
evaluate the effectiveness of the chosen architectures, with a focus on understanding the
impact of model complexity and context size on performance. Despite facing challenges
related to model complexity, data handling, and computational resources, the research
provides valuable insights into the feasibility of virtual sensing. The results indicate
that the models could only predict the average sensor values and struggled to capture
detailed sensor data nuances, highlighting the need for more advanced models and better
training techniques. This work lays the foundation for future exploration in reducing
robotic sensor costs through virtual sensing, with recommendations for employing more
complex models and optimizing data handling strategies

## Repository Structure

## Installation and Setup

Simply clone the repository and create a virtual environment. 

```bash
git clone https://github.com/loucaspapalazarou/virtual-sensing.git
cd virtual-sensing
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt 
```

## Usage

[Entire Document](Virtual_Sensing.pdf)

### Train

Use the `train.py` file to train a specified model. Use `-h` for the list of available options. For example:

```bash
python train.py --model transformer
```

The Pytorch Lightning logs will appear:

```
...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 1 processes
----------------------------------------------------------------------------------------------------

Missing logger folder: ./lightning_logs/transformer/config.json
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

...

  | Name   | Type        | Params | Mode 
-----------------------------------------------
0 | resnet | ResNetBlock | 11.2 M | train
1 | model  | Transformer | 371 M  | train
-----------------------------------------------
371 M     Trainable params
11.2 M    Non-trainable params
383 M     Total params
1,532.178 Total estimated model params size (MB)

...

Sanity Checking: |          | 0/? [00:00<?, ?it/s]
Sanity Checking:   0%|          | 0/2 [00:00<?, ?it/s]
Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]
Sanity Checking DataLoader 0:  50%|█████     | 1/2 [00:00<00:00,  2.30it/s]
Sanity Checking DataLoader 0: 100%|██████████| 2/2 [00:08<00:00,  0.23it/s]
                                                                           
Training: |          | 0/? [00:00<?, ?it/s]
Training:   0%|          | 0/3666 [00:00<?, ?it/s]
Epoch 0:   0%|          | 0/3666 [00:00<?, ?it/s] 
Epoch 0:   0%|          | 1/3666 [00:08<8:44:15,  0.12it/s]
Epoch 0:   0%|          | 1/3666 [00:08<8:44:16,  0.12it/s, v_num=0]
Epoch 0:   0%|          | 2/3666 [00:16<8:36:40,  0.12it/s, v_num=0]

...
```

View training progress in Tensorboard

```bash
tensorboard --logdir ./lightning_logs
```

If that doesn't work, use

```bash
python -m tensorboard.main --logdir ./lightning_logs
```

### Predict

For predictions, the `analysis/` folder contains useful examples on how to use the models.

```python
# load the a model using its module, a checkpoint and its hyper parameters
from modules.transformer import TransformerModule

checkpoint = "path/to/model_logs/checkpoints/epoch=0-step=2900.ckpt"
hparams = "path/to/model_logs/hparams.yaml"
transformer = TransformerModule.load_from_checkpoint(
    checkpoint_path=checkpoint,
    hparams_file=hparams,
)

# preferably use the dataloader as it handles preprocessing and data concatenation
ds = FrankaDataset(data_dir=os.path.join("path/to/work_dir", "data-folder"),
                      episode_length=200,
                      limited_gpu_memory=True,
                      stride=1,
                      prediction_distance=1,
                      window_size=10
                      )
dl = DataLoader(ds, batch_size=1, shuffle=False)
sample = next(iter(dl))

output = transformer.predict(sample)
```
