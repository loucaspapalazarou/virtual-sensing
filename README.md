# Virtual Sensing

***Work in progress***

```python
# data-file-n.pt
{
    "sensor_data": torch.Size([300, 12, 35]),
    "camera_data": torch.Size([300, 12, 3, 256, 256, 4])
}
```


```bash
# train.py
options:
  -h, --help            show this help message and exit
  --model {transformer,mamba,rnn}
                        The name of the model you want to train
  --config-file CONFIG_FILE
                        Path to the JSON file with parameters
  --checkpoint CHECKPOINT
                        Path to a checkpoint file to resume training
  --fast-dev-run, --no-fast-dev-run
                        Dev run
  --devices DEVICES     List of GPU devices. Example: [0,1,2]
```

Example
```bash
python train.py --model transformer
```