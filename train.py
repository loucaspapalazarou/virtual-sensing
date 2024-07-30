from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset import FrankaDataModule
from dotenv import load_dotenv
from services import EmailCallback
import argparse
import json
import os

from modules.transformer import TransformerModule
from modules.mamba import MambaModule
from modules.rnn import RNNModule  # Assuming you save the RNN module in modules/rnn.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the model you want to train",
        choices=["transformer", "mamba", "rnn"],
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="config.json",
        help="Path to the JSON file with parameters",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a checkpoint file to resume training",
    )
    parser.add_argument(
        "--fast-dev-run",
        default=False,
        help="Dev run",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="List of GPU devices. Example: [0,1,2]",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        params = json.load(f)

    data_module = FrankaDataModule(
        data_dir=os.path.join(os.getenv("WORK_DIR"), params["data_dir"]),
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        data_portion=params["data_portion"],
        episode_length=params["episode_length"],
        stride=params["stride"],
        window_size=params["window_size"],
        prediction_distance=params["prediction_distance"],
        limited_gpu_memory=True,
    )

    # [num of sensor features] + [resnet_featues * 3] (3 images)
    data_dim = data_module.get_num_sensor_features() + (3 * params["resnet_features"])

    common_params = {
        "name": args.model,
        "d_model": data_dim,
        "start_lr": params["start_lr"],
        "end_lr": params["end_lr"],
        "activation": params["activation"],
        "window_size": params["window_size"],
        "prediction_distance": params["prediction_distance"],
        "target_feature_indices": params["target_feature_indices"],
        "resnet_features": params["resnet_features"],
        "resnet_checkpoint": os.path.join(
            os.getenv("WORK_DIR"), params["resnet_checkpoint"]
        ),
    }

    model_specific_params = {}
    if args.model == "transformer":
        model_class = TransformerModule
        model_specific_params = {
            "nhead": params[args.model]["nhead"],
            "num_encoder_layers": params[args.model]["num_encoder_layers"],
            "num_decoder_layers": params[args.model]["num_decoder_layers"],
            "dim_feedforward": params[args.model]["dim_feedforward"],
        }
    elif args.model == "mamba":
        model_class = MambaModule
        model_specific_params = {
            "d_state": params[args.model]["d_state"],
            "d_conv": params[args.model]["d_conv"],
            "expand": params[args.model]["expand"],
        }
    elif args.model == "rnn":
        model_class = RNNModule
        model_specific_params = {
            "rnn_hidden_size": params[args.model]["rnn_hidden_size"],
            "num_layers": params[args.model]["num_layers"],
        }
    else:
        raise ValueError("Invalid model")

    model_params = {**common_params, **model_specific_params}

    if args.checkpoint:
        model = model_class.load_from_checkpoint(args.checkpoint, **model_params)
    else:
        model = model_class(**model_params)

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=100,
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="./lightning_logs", name=f"{args.model}"
    )

    load_dotenv()
    email_callback = EmailCallback("loukis500@gmail.com", os.getenv("EMAIL_PASSWORD"))

    trainer = Trainer(
        logger=tb_logger,
        max_epochs=params["max_epochs"],
        log_every_n_steps=params["log_every_n_steps"],
        fast_dev_run=args.fast_dev_run,
        val_check_interval=500,
        check_val_every_n_epoch=None,
        accelerator="gpu",
        devices="auto" if args.devices == "auto" else eval(args.devices),
        num_nodes=1,
        strategy="ddp",
        callbacks=[checkpoint_callback, email_callback],
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
