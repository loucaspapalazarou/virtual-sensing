import argparse
import torch
import yaml
from modules.transformer import TransformerModule
from modules.mamba import MambaModule


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
    parser.add_argument(
        "--input-tensor",
        type=str,
        required=True,
        help="A .pt file containing a tensor of size B, S, N or S, N",
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
        case "mamba":
            model = MambaModule.load_from_checkpoint(
                checkpoint_path=args.checkpoint, hparams_file=args.hparams
            )
        case _:
            raise ValueError("Invalid model")

    # Load the input tensor
    input_tensor = torch.load(args.input_tensor)

    # Check the dimensions of the tensor and reshape if necessary
    if len(input_tensor.shape) == 2:
        # If the tensor is of shape (S, N), add a batch dimension
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Ensure the tensor is of shape (B, S, N)
    if len(input_tensor.shape) != 3:
        raise ValueError("Input tensor must be of shape (B, S, N) or (S, N)")

    # Perform inference
    model = model.cuda()
    input_tensor = input_tensor.cuda()

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model.predict(input_tensor)

    # Print the result
    print(output)
    print(output.shape)


if __name__ == "__main__":
    main()
