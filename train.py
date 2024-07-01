import pytorch_lightning as pl
from dataset import FrankaDataModule, DATA_DIM
import argparse
import json

from models.transformer_model import TransformerModel
# from models.mamba_model import MambaModel
# from models.wavenet_model import WaveNet

def main():
    parser = argparse.ArgumentParser(description="Read parameters from a JSON file")
    parser.add_argument(
        "--model", type=str, required=True, help="The model you want to train"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the JSON file with parameters",
    )
    parser.add_argument(
        "--fast_dev_run",
        default=False,
        help="Dev run",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        params = json.load(f)

    match args.model:
        case "transformer":
            model = TransformerModel(
                name=args.model,
                d_model=DATA_DIM,
                nhead=params[args.model]["nhead"],
                num_encoder_layers=params[args.model]["num_encoder_layers"],
                num_decoder_layers=params[args.model]["num_decoder_layers"],
                dim_feedforward=params[args.model]["dim_feedforward"],
                lr=params[args.model]["lr"],
                stride=params[args.model]["stride"],
                prediction_horizon=params[args.model]["prediction_horizon"],
            )
        case "mamba":
            raise NotImplementedError()
        case "wavenet":
            raise NotImplementedError()
        case _:
            raise ValueError("Invalid model")
    
    data_module = FrankaDataModule(
        data_dir=params["data_dir"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=20,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
