from torch import nn
from modules.base import BaseModelModule


class RNNModule(BaseModelModule):
    def __init__(
        self,
        d_model,
        rnn_hidden_size,
        num_layers,
        lr,
        window_size,
        target_feature_indices,
        resnet_features,
        resnet_checkpoint,
        name,
    ):
        super().__init__(
            d_model=d_model,
            name=name,
            lr=lr,
            window_size=window_size,
            target_feature_indices=target_feature_indices,
            resnet_features=resnet_features,
            resnet_checkpoint=resnet_checkpoint,
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
