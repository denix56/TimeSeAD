# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import numpy as np
import torch
import torch.nn as nn
torch._dynamo.config.capture_scalar_outputs = True


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = np.arange(0, seq_len // 2)
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = np.arange(0, modes)
    index.sort()
    return index


# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, num_heads=8, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        # print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain,
        it does FFT, linear transform, and Inverse FFT.
        """
        self.seq_len = seq_len
        # get modes on frequency domain
        index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        self.register_buffer('index', torch.from_numpy(index))
        # print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(len(self.index), num_heads, in_channels // num_heads, out_channels // num_heads, dtype=torch.cfloat))

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.to(torch.float32)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=1)
        freq_len = x_ft.size(-1)
        # index is sorted; searchsorted counts how many entries are < freq_len
        valid = torch.searchsorted(self.index, freq_len).item()
        assert valid > 0
        assert valid <= self.index.shape[0]
        index = self.index[:valid]
        # Perform Fourier neural operations
        out_ft = x_ft.new_zeros((B, freq_len, H, E))
        out_ft[:, index] = torch.einsum("blhe,lhek->blhk", x_ft[:, index], self.weights[:valid])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=1).to(x.dtype)
        return (x, None)


# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        seq_len_q,
        seq_len_kv,
        modes=64,
        mode_select_method='random',
        activation='tanh',
        policy=0,
        num_heads=8,
    ):
        super(FourierCrossAttention, self).__init__()
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform,
        attention mechanism and Inverse FFT.
        """
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim_in = in_channels // num_heads
        self.head_dim_out = out_channels // num_heads

        # get modes and store as buffers (like in FourierBlock)
        index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
        self.register_buffer('index_q', torch.from_numpy(index_q))
        self.register_buffer('index_kv', torch.from_numpy(index_kv))

        self.scale = 1.0 / (in_channels * out_channels)

        # complex weights encoded as (..., 2) real tensor, similar to FourierBlock
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(
                len(index_q),
                num_heads,
                self.head_dim_in,
                self.head_dim_out,
                dtype=torch.cfloat,
            )
        )

    def forward(self, q, k, v, mask):
        # q, k, v: [B, L, H, E]
        B, Lq, H, E = q.shape
        assert H == self.num_heads
        _, Lkv, _, _ = k.shape

        xq = q.to(torch.float32)
        xk = k.to(torch.float32)
        # original code does not actually use v; keep behavior
        # xv = v.permute(0, 2, 3, 1)

        # FFT
        xq_ft = torch.fft.rfft(xq, dim=1)   # [B, H, E, Lq//2+1]
        xk_ft = torch.fft.rfft(xk, dim=1)   # [B, H, E, Lkv//2+1]

        freq_len_q = xq_ft.size(1)
        freq_len_kv = xk_ft.size(1)
        valid_q = torch.searchsorted(self.index_q, freq_len_q).item()
        valid_kv = torch.searchsorted(self.index_kv, freq_len_kv).item()
        index_q = self.index_q[:valid_q]
        index_kv = self.index_kv[:valid_kv]

        # select requested modes in one shot (no Python loop)
        xq_ft_sel = xq_ft[:, index_q]      # [B, H, E, Mq]
        xk_ft_sel = xk_ft[:, index_kv]     # [B, H, E, Mk]

        # attention in frequency domain:
        # xqk_ft: [B, H, Mq, Mk]
        xqk_ft = torch.einsum('blhe,bmhe->bmhl',xq_ft_sel, xk_ft_sel)

        if self.activation == 'tanh':
            # TODO: check - do we perform per chanel tanh or complex tanh
            xqk_ft = torch.view_as_complex(torch.view_as_real(xqk_ft).tanh())
        elif self.activation == 'softmax':
            attn = torch.softmax(xqk_ft.abs(), dim=1)
            xqk_ft = torch.complex(attn, torch.zeros_like(attn))
        else:
            raise Exception(f'{self.activation} activation function is not implemented')

        # linear transform with complex weights
        weights_c = self.weights[:valid_q]    # [H, Ein, Eout, Mq]

        # combine with keys again: [B, H, E, Mq]
        xqkvw = torch.einsum("bmhl,bmhe,lhek->blhk", xqk_ft, xk_ft_sel, weights_c)

        # place selected freqs back into full spectrum
        out_ft = xqk_ft.new_zeros(B, freq_len_q, H, self.head_dim_out)
        out_ft[:, index_q] = xqkvw

        # iFFT back to time domain
        out = torch.fft.irfft(
            out_ft / (self.in_channels * self.out_channels),
            n=xq.size(-1),
            dim=1,
        )  # [B, H, Eout, L]

        return out.to(q.dtype), None
