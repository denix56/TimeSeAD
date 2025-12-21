# Implementation from Time Series Library https://github.com/thuml/Time-Series-Library
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class InceptionBlockV1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_kernels=6,
        init_weight=True,
        circular_padding: bool = False,
        use_spectral_norm: bool = False,
    ):
        super(InceptionBlockV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.circular_padding = circular_padding
        self.use_spectral_norm = use_spectral_norm
        kernels = []
        for i in range(self.num_kernels):
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=2 * i + 1,
                padding=0,
                bias=True,
            )
            kernels.append(spectral_norm(conv) if self.use_spectral_norm else conv)
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            pad = i
            x_padded = x
            if pad > 0:
                # Torch circular padding does not allow wrapping more than once.
                # When the temporal dimension ("width") becomes very small (e.g. period=1),
                # the requested padding may exceed the dimension size and trigger a runtime
                # error. Clamp the padding so we never wrap more than the available length.
                max_pad = min(
                    pad,
                    max(x_padded.size(-1) - 1, 0),
                    max(x_padded.size(-2) - 1, 0),
                )
                if max_pad == 0:
                    res_list.append(self.kernels[i](x_padded))
                    continue
                padding = (max_pad, max_pad, max_pad, max_pad)
                if self.circular_padding:
                    x_padded = F.pad(x_padded, padding, mode="circular")
                else:
                    x_padded = F.pad(x_padded, padding)
            res_list.append(self.kernels[i](x_padded))
        res = sum(res_list)
        return res
