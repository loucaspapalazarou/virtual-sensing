#!/bin/bash

conda run --no-capture-output -n diss tensorboard --logdir ./lightning_logs/ --host 0.0.0.0 --port 6006