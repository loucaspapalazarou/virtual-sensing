from torch import nn
from modules.base import BaseModelModule


class RNNModule(BaseModelModule):
    def __init__(
        self,
        d_model,
        rnn_hidden_size,
        num_layers,
        lr,
        stride,
        window_size,
        prediction_distance,
        target_feature_indices,
        resnet_features,
        resnet_checkpoint,
        name,
    ):
        super().__init__(
            lr,
            stride,
            window_size,
            prediction_distance,
            target_feature_indices,
            resnet_features,
            resnet_checkpoint,
            name,
        )
        self.model = nn.RNN(
            input_size=d_model,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, src, tgt=None):
        rnn_out, _ = self.model(src)
        return rnn_out
