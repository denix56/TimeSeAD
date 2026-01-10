# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
from typing import Tuple
import logging

import torch
import torch.nn as nn

from .. import BaseModel
from ..layers import DataEmbedding
from ..layers import AutoCorrelationLayer, FourierBlock, MultiWaveletTransform
from ..layers.autoformer_encdec import Encoder, EncoderLayer, CustomLayerNorm, SeriesDecomp


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

torch.set_printoptions(
    threshold=float('inf'),  # print all elements
    linewidth=200,           # avoid line wrapping
)

def log_debug(tensor: torch.Tensor, debug: bool):
    if debug and (not torch.isfinite(tensor).all() or tensor.max().detach().cpu().item() >= 1e+9):
        logger.debug("%s", tensor, stacklevel=2)


class FEDformer(BaseModel):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(
            self,
            window_size: int,
            input_dim: int,
            moving_avg: int=25,
            model_dim: int=128,
            dropout: float=0.1,
            num_heads: int=8,
            fcn_dim: int=128,
            activation: str='gelu',
            encoder_layers: int=3,
            version: str='fourier',
            mode_select: str='random',
            modes: int=32,
            fft_norm: str = "backward",
            w_init: str = "random",
            mode_policy: str = "static",
            topk: int = 0,
            freq_norm_mode: str | None = None,  # None -> no normalization
            lrfop: bool = False,
            shared_attention: bool = False,
            scatter_freq: bool = False,
        ) -> None:
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(FEDformer, self).__init__()
        self.seq_len = window_size

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = SeriesDecomp(moving_avg)
        self.enc_embedding = DataEmbedding(input_dim, model_dim, dropout)

        if self.version == 'Wavelets':
            encoder_self_att = lambda: MultiWaveletTransform(ich=model_dim, L=1, base='legendre')
        else:
            encoder_self_att = lambda: FourierBlock(in_channels=model_dim,
                                            out_channels=model_dim,
                                            seq_len=self.seq_len,
                                            num_heads=num_heads,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select,
                                            fft_norm=fft_norm,
                                            w_init=w_init,
                                            freq_norm_mode=freq_norm_mode,
                                            lrfop=lrfop,
                                            mode_policy=mode_policy,
                                            topk=topk,
                                            scatter_freq=scatter_freq,
            )
        if shared_attention:
            encoder_self_att_inst = encoder_self_att()
            encoder_self_att = lambda: encoder_self_att_inst
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att(),  # instead of multi-head attention in transformer
                        model_dim, num_heads),
                    model_dim,
                    fcn_dim,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(encoder_layers)
            ],
            norm_layer=CustomLayerNorm(model_dim)
        )

        self.projection = nn.Linear(model_dim, input_dim, bias=True)


    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        x_enc = inputs[0].contiguous()
        log_debug(x_enc, debug=True)
        enc_out = self.enc_embedding(x_enc)
        log_debug(enc_out, debug=True)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        log_debug(enc_out, debug=True)
        dec_out = self.projection(enc_out)
        return dec_out

