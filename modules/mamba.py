from mamba_ssm import Mamba  # type: ignore
from modules.base import BaseModelModule


class MambaModule(BaseModelModule):
    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        expand,
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
        self.model = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, src, tgt=None):
        return self.model(src)
