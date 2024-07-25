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
        self.model = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, src, tgt=None):
        return self.model(src)
