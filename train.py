import pytorch_lightning as pl
from dataset import FrankaDataModule
import argparse
import json

from modules.transformer import TransformerModule
from modules.mamba import MambaModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the model you want to train",
        choices=["transformer", "mamba", "mamba2"],
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="config.json",
        help="Path to the JSON file with parameters",
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

    match args.model:
        case "transformer":
            model = TransformerModule(
                # model specific params
                name=args.model,
                d_model=data_dim,
                nhead=params[args.model]["nhead"],
                num_encoder_layers=params[args.model]["num_encoder_layers"],
                num_decoder_layers=params[args.model]["num_decoder_layers"],
                dim_feedforward=params[args.model]["dim_feedforward"],
                # general params
                lr=params["lr"],
                stride=params["stride"],
                window_size=params["window_size"],
                prediction_distance=params["prediction_distance"],
                target_feature_indices=params["target_feature_indices"],
                resnet_features=params["resnet_features"],
            )
        case "mamba":
            model = MambaModule(
                # model specific params
                name=args.model,
                d_model=data_dim,
                d_state=params[args.model]["d_state"],
                d_conv=params[args.model]["d_conv"],
                expand=params[args.model]["expand"],
                # general params
                lr=params["lr"],
                stride=params["stride"],
                window_size=params["window_size"],
                prediction_distance=params["prediction_distance"],
                target_feature_indices=params["target_feature_indices"],
                resnet_features=params["resnet_features"],
            )
        case _:
            raise ValueError("Invalid model")

    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        log_every_n_steps=params["log_every_n_steps"],
        fast_dev_run=args.fast_dev_run,
        val_check_interval=0.5,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
