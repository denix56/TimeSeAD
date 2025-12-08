# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
# TODO(AR): Optimize the code

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _delayed_agg(self, tmp_values: torch.Tensor, tmp_corr: torch.Tensor, index: torch.LongTensor, top_k: int) -> torch.Tensor:
        B, head, channel, length = tmp_values.shape
        device = tmp_values.device
        delays_agg = torch.zeros_like(tmp_values, dtype=torch.float)

        # shifts for each top-k index (negative because you use -index[i])
        shifts = (-index) % length  # (top_k,)

        # Base positions along the last dimension
        base = torch.arange(length, device=device)  # (length,)

        # For each k, compute circularly shifted indices:
        # pos[k, j] = (j + shifts[k]) % length
        pos = (base.unsqueeze(0) + shifts.unsqueeze(1)) % length  # (top_k, length)

        # We want to gather along the last dimension.
        # Expand shapes to match tmp_values
        # tmp_values_expanded: (B, head, channel, top_k, length)
        tmp_values_expanded = tmp_values.unsqueeze(3).expand(B, head, channel, top_k, length)

        # pos_expanded: (B, head, channel, top_k, length)
        pos_expanded = pos.view(1, 1, 1, top_k, length).expand(B, head, channel, top_k, length)

        # All "rolled" patterns at once: (B, head, channel, top_k, length)
        rolled = torch.gather(tmp_values_expanded, dim=-1, index=pos_expanded)

        # Broadcast weights from tmp_corr: (B, top_k) -> (B, 1, 1, top_k, 1)
        weights = tmp_corr.view(B, top_k, 1, 1, 1).permute(0, 2, 3, 1, 4)  # (B, 1, 1, top_k, 1)

        # Multiply and sum over top_k dimension
        # contrib: (B, head, channel, top_k, length)
        contrib = rolled * weights

        # Aggregate over top_k
        delays_agg = contrib.sum(dim=3)  # (B, head, channel, length)

        return delays_agg

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        # Assuming: corr.shape = (B, ..., L) so that mean_value.shape = (B, L)

        top_k = int(self.factor * math.log(length))
        # (B, L)
        mean_value = corr.mean(dim=(1, 2))
        # (L,) – scores across batch
        scores = mean_value.mean(dim=0)
        # top-k indices along L
        _, index = torch.topk(scores, k=top_k, dim=0)  # index: (top_k,)
        # (B, top_k) – gather all columns in one go
        # update corr
        tmp_corr = torch.softmax(mean_value[:, index], dim=-1)
        # aggregation
        delays_agg = self._delayed_agg(values, tmp_corr, index, top_k)
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length, device=values.device).view(1, 1, 1, -1).expand(batch, head, channel, -1)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model: int, n_heads: int, d_keys: Optional[int] = None,
                 d_values: Optional[int] = None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.q_proj_out = d_keys * n_heads
        self.k_proj_out = d_keys * n_heads
        self.v_proj_out = d_values * n_heads
        self.qkv_projection = nn.Linear(d_model, self.q_proj_out + self.k_proj_out + self.v_proj_out)

    def _split_qkv_weights(self):
        q_end = self.q_proj_out
        k_end = q_end + self.k_proj_out
        return (
            self.qkv_projection.weight[:q_end],
            self.qkv_projection.weight[q_end:k_end],
            self.qkv_projection.weight[k_end:],
        )

    def _split_qkv_biases(self):
        if self.qkv_projection.bias is None:
            return None, None, None
        q_end = self.q_proj_out
        k_end = q_end + self.k_proj_out
        return (
            self.qkv_projection.bias[:q_end],
            self.qkv_projection.bias[q_end:k_end],
            self.qkv_projection.bias[k_end:],
        )

    def _project_shared_qkv(self, tensor):
        fused = self.qkv_projection(tensor)
        q_out, k_out, v_out = fused.split(
            [self.q_proj_out, self.k_proj_out, self.v_proj_out],
            dim=-1,
        )
        return q_out, k_out, v_out

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        q_w, k_w, v_w = self._split_qkv_weights()
        q_b, k_b, v_b = self._split_qkv_biases()

        if queries is keys and queries is values:
            q_out, k_out, v_out = self._project_shared_qkv(queries)
        else:
            q_out = F.linear(queries, q_w, q_b)
            k_out = F.linear(keys, k_w, k_b)
            v_out = F.linear(values, v_w, v_b)

        queries = q_out.view(B, L, H, -1)
        keys = k_out.view(B, S, H, -1)
        values = v_out.view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

