import pytorch_lightning as pl
from models.mamba_model import MambaModel
from models.transformer_model import TransformerModel
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The checkpoint you want to load the model from",
    )
    parser.add_argument(
        "--hparams",
        type=str,
        required=True,
        help="The hyperparameters file of the model",
    )
    args = parser.parse_args()

    with open(args.hparams, "r") as file:
        hparams = yaml.safe_load(file)
    model_name = hparams.get("name", "Unknown")

    match model_name:
        case "transformer":
            model = TransformerModel.load_from_checkpoint(
                checkpoint_path=args.checkpoint, hparams_file=args.hparams
            )
        # the mamba version is inside hparams, and encapsulated in MambaModel
        # with either version provided, the initialization here is the same
        case "mamba" | "mamba2":
            model = MambaModel.load_from_checkpoint(
                checkpoint_path=args.checkpoint, hparams_file=args.hparams
            )
        case _:
            raise ValueError("Invalid model")

    print(model)


if __name__ == "__main__":
    main()
