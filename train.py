import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers
from dataset import FrankaDataModule
import argparse
import json

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
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        params = json.load(f)

    data_module = FrankaDataModule(
        data_dir=params["data_dir"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        data_portion=params["data_portion"],
        episode_length=params["episode_length"],
    )

    # [num of sensor features] + [resnet_featues * 3] (3 images)
    data_dim = data_module.get_num_sensor_features() + (3 * params["resnet_features"])

    common_params = {
        "name": args.model,
        "d_model": data_dim,
        "lr": params["lr"],
        "stride": params["stride"],
        "window_size": params["window_size"],
        "prediction_distance": params["prediction_distance"],
        "target_feature_indices": params["target_feature_indices"],
        "resnet_features": params["resnet_features"],
        "resnet_checkpoint": params["resnet_checkpoint"],
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

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="./lightning_logs", name=f"{args.model}"
    )

    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=params["max_epochs"],
        log_every_n_steps=params["log_every_n_steps"],
        fast_dev_run=args.fast_dev_run,
        val_check_interval=0.2,
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy="ddp",
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
