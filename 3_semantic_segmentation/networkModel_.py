import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        block_config = [3, 4, 6, 3]
        img_size = 475
        img_size_8 = 60

        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(
            n_block=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1
        )
        self.feature_res_2 = ResidualBlockPSP(
            n_block=block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1
        )
        self.feature_dilated_res_1 = ResidualBlockPSP(
            n_block=block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2
        )
        self.feature_dilated_res_2 = ResidualBlockPSP(
            n_block=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4
        )

        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_size=[
            6, 3, 2, 1], height=img_size_8, width=img_size_8)

        self.decode_feature = DecodePSPFeature(
            height=img_size, width=img_size, n_classes=n_classes
        )

        self.aux = AuxiliaryPSPPlayers(
            in_channels=1024, height=img_size, width=img_size, n_classes=n_classes
        )

        def forward(self, x):
            x = self.feature_conv(x)
            x = self.feature_res_1(x)
            x = self.feature_res_2(x)
            x = self.feature_dilated_res_1(x)

            output_aux = self.aux(x)

            x = self.feature_dilated_res_2(x)

            x = self.pyramid_pooling(x)
            output = self.decode_feature(x)

            return (output, output_aux)
