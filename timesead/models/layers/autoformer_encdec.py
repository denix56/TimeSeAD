# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

torch.set_printoptions(
    threshold=float('inf'),  # print all elements
    linewidth=200,           # avoid line wrapping
)

def log_debug(tensor: torch.Tensor, debug: bool):
    if debug and not torch.isfinite(tensor).all():
        logger.debug("%s", tensor, stacklevel=2)
        return False
    return True


class CustomLayerNorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(CustomLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_hat = self.layernorm(x.float()).to(x.dtype)
        if not log_debug(x_hat, debug=True):
            st = {'x': x, 'x_hat': x_hat, 'ln': self.layernorm.state_dict()}
            from pathlib import Path
            Path('/raid/work/senkin/noboom/debug').mkdir(parents=True, exist_ok=True)
            torch.save(st, '/raid/work/senkin/noboom/debug/ln.pt')
        bias = torch.mean(x_hat, dim=1, keepdim=True)
        log_debug(bias, debug=True)
        return x_hat - bias


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    Expects input of shape (batch, length, channels).
    """

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        # AvgPool1d expects (N, C, L)
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,
                                stride=stride,
                                padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # Efficient border padding instead of repeat+cat
        # pad = (pad_left, pad_right) for last dim (L)
        x = torch._C._nn.pad(x, (self.padding, self.padding), mode="replicate")

        # Average pooling along time dimension
        x = self.avg(x)  # (B, C, L_out)

        # Back to (B, L_out, C)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size: int):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        log_debug(x, debug=True)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        log_debug(new_x, debug=True)
        x = x + self.dropout(new_x)
        log_debug(x, debug=True)
        x, _ = self.decomp1(x)
        log_debug(x, debug=True)
        y = x
        log_debug(y, debug=True)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        log_debug(y, debug=True)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        log_debug(y, debug=True)
        res, _ = self.decomp2(x + y)
        log_debug(res, debug=True)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                log_debug(x, debug=True)
                x, attn = attn_layer(x, attn_mask=attn_mask)
                log_debug(x, debug=True)
                x = conv_layer(x)
                log_debug(x, debug=True)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            log_debug(x, debug=True)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                log_debug(x, debug=True)
                x, attn = attn_layer(x, attn_mask=attn_mask)
                log_debug(x, debug=True)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
            log_debug(x, debug=True)

        return x, attns
