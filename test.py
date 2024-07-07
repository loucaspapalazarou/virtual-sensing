import pytorch_lightning as pl
from modules.mamba import MambaModule
from modules.transformer import TransformerModule
import argparse
import yaml
import torch


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
            model = TransformerModule.load_from_checkpoint(
                checkpoint_path=args.checkpoint, hparams_file=args.hparams
            )
        # the mamba version is inside hparams, and encapsulated in MambaModel
        # with either version provided, the initialization here is the same
        case "mamba":
            model = MambaModule.load_from_checkpoint(
                checkpoint_path=args.checkpoint, hparams_file=args.hparams
            )
        case _:
            raise ValueError("Invalid model")

    raise NotImplementedError("what now?")

    # res = model.predict(torch.rand(1, 40, 36))
    # print(res)
    # print(res.shape)


if __name__ == "__main__":
    main()
