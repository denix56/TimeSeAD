# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import AnomalyDetector
from ...models import BaseModel
from ...optim.loss import Loss


class ResTrans1DBlock(torch.nn.Module):
    def __init__(self, channel: int, bias: bool = False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(channel, channel, 3, 1, 1, bias=bias)
        self.in1 = nn.InstanceNorm1d(channel, affine=bias)
        self.conv2 = nn.Conv1d(channel, channel, 3, 1, 1, bias=bias)
        self.in2 = nn.InstanceNorm1d(channel, affine=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 dilation: int = 1, bias: bool = False):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.reflection_pad = nn.ReflectionPad1d(padding)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reflection_pad(x)
        out = self.conv1d(out)
        return out


class SeqTransformNet(nn.Module):
    def __init__(self, x_dim: int, hdim: int, num_layers: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = ConvLayer(x_dim, hdim, 3, 1, bias=False)
        self.in1 = nn.InstanceNorm1d(hdim, affine=False)
        res_blocks = []
        for _ in range(num_layers - 2):
            res_blocks.append(ResTrans1DBlock(hdim, False))
        self.res = nn.Sequential(*res_blocks)
        self.conv2 = ConvLayer(hdim, x_dim, 3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.in1(self.conv1(x)))
        for block in self.res:
            out = block(out)
        out = self.conv2(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, conv_param=None, downsample=None,
                 batchnorm: bool = False, bias: bool = False):
        super().__init__()

        self.conv1 = nn.Conv1d(in_dim, in_dim, 1, 1, 0, bias=bias)
        if conv_param is not None:
            self.conv2 = nn.Conv1d(in_dim, in_dim, conv_param[0], conv_param[1], conv_param[2], bias=bias)
        else:
            self.conv2 = nn.Conv1d(in_dim, in_dim, 3, 1, 1, bias=bias)

        self.conv3 = nn.Conv1d(in_dim, out_dim, 1, 1, 0, bias=bias)
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(in_dim)
            self.bn2 = nn.BatchNorm1d(in_dim)
            self.bn3 = nn.BatchNorm1d(out_dim)
            if downsample:
                self.bn4 = nn.BatchNorm1d(out_dim)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.batchnorm = batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batchnorm:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if self.batchnorm:
                residual = self.bn4(residual)

        out += residual
        out = self.relu(out)

        return out


class SeqEncoder(nn.Module):
    def __init__(self, x_dim: int, x_len: int, h_dim: int, z_dim: int, bias: bool,
                 num_layers: int, batch_norm: bool):
        super().__init__()

        self.bias = bias
        self.batchnorm = batch_norm
        enc = [self._make_layer(x_dim, h_dim, (3, 1, 1))]
        in_dim = h_dim
        window_size = x_len
        for i in range(num_layers - 2):
            out_dim = h_dim * 2 ** i
            enc.append(self._make_layer(in_dim, out_dim, (3, 2, 1)))
            in_dim = out_dim
            window_size = np.floor((window_size + 2 - 3) / 2) + 1

        self.enc = nn.Sequential(*enc)
        self.final_layer = nn.Conv1d(in_dim, z_dim, int(window_size), 1, 0)

    def _make_layer(self, in_dim: int, out_dim: int, conv_param=None):
        downsample = None
        if conv_param is not None:
            downsample = nn.Conv1d(in_dim, out_dim, conv_param[0], conv_param[1], conv_param[2], bias=self.bias)
        elif in_dim != out_dim:
            downsample = nn.Conv1d(in_dim, out_dim, 1, 1, 0, bias=self.bias)

        return ResBlock(in_dim, out_dim, conv_param, downsample=downsample,
                        batchnorm=self.batchnorm, bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        z = self.final_layer(z)
        return z.squeeze(-1)


class SeqNets:
    def _make_nets(self, x_dim: int, config: dict):
        enc_nlayers = config['enc_nlayers']
        enc_hdim = config['enc_hdim']
        z_dim = config['latent_dim']
        x_len = config['x_length']
        trans_nlayers = config['trans_nlayers']
        num_trans = config['num_trans']
        batch_norm = config['batch_norm']

        enc = nn.ModuleList([
            SeqEncoder(x_dim, x_len, enc_hdim, z_dim, config['enc_bias'], enc_nlayers, batch_norm)
            for _ in range(num_trans + 1)
        ])
        trans = nn.ModuleList([
            SeqTransformNet(x_dim, x_dim, trans_nlayers) for _ in range(num_trans)
        ])

        return enc, trans


class NeutralAD(BaseModel):
    def __init__(self, ts_channels: int, seq_len: int, num_trans: int = 4, trans_type: str = 'residual',
                 enc_hdim: int = 32, enc_nlayers: int = 4, trans_nlayers: int = 4, latent_dim: int = 32,
                 batch_norm: bool = False, enc_bias: bool = False):
        super().__init__()

        self.num_trans = num_trans
        self.trans_type = trans_type
        self.z_dim = latent_dim
        config = dict(
            enc_nlayers=enc_nlayers,
            enc_hdim=enc_hdim,
            latent_dim=latent_dim,
            x_length=seq_len,
            trans_nlayers=trans_nlayers,
            num_trans=num_trans,
            batch_norm=batch_norm,
            enc_bias=enc_bias
        )
        self.enc, self.trans = SeqNets()._make_nets(ts_channels, config)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, = inputs
        x = x.float()
        x = x.permute(0, 2, 1)

        x_t = torch.empty(x.shape[0], self.num_trans, x.shape[1], x.shape[2]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_t[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_t[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_t[:, i] = mask + x
            else:
                raise ValueError(f'Unknown trans_type: {self.trans_type}')

        x_cat = torch.cat([x.unsqueeze(1), x_t], 1)
        zs = self.enc[0](x_cat.reshape(-1, x.shape[1], x.shape[2]))
        zs = zs.reshape(x.shape[0], self.num_trans + 1, self.z_dim)

        return zs


def _dcl_score(z: torch.Tensor, temperature: float, eval: bool) -> torch.Tensor:
    z = F.normalize(z, p=2, dim=-1)
    z_ori = z[:, 0]
    z_trans = z[:, 1:]
    batch_size, num_trans, _ = z.shape

    sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / temperature))
    mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
    trans_matrix = sim_matrix[:, 1:].sum(-1)

    pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / temperature)
    k_trans = num_trans - 1
    scale = 1 / np.abs(k_trans * np.log(1.0 / k_trans))

    loss_tensor = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale

    score = loss_tensor.sum(1)
    if eval:
        return score
    return score.mean()


def _eucdcl_score(z: torch.Tensor, temperature: float, eval: bool) -> torch.Tensor:
    batch_size, num_trans, _ = z.shape
    sim_matrix = -torch.cdist(z, z)
    sim_matrix = torch.exp(sim_matrix / temperature)
    mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
    sim_matrix = sim_matrix + 1e-8
    trans_matrix = sim_matrix[:, 1:].sum(-1)
    pos_sim = sim_matrix[:, 1:, 0]

    k_trans = num_trans - 1
    scale = 1 / np.abs(k_trans * np.log(1.0 / k_trans))
    score = (-torch.log(pos_sim) + torch.log(trans_matrix)) * scale
    score = score.sum(1)
    if eval:
        return score
    return score.mean()


class NeutralADLoss(Loss):
    def __init__(self, temperature: float = 0.1, use_euclidean: bool = False):
        super().__init__()
        self.temperature = temperature
        self.use_euclidean = use_euclidean

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...],
                eval: bool = False, *args, **kwargs) -> torch.Tensor:
        z, = predictions
        if self.use_euclidean:
            return _eucdcl_score(z, self.temperature, eval=eval)
        return _dcl_score(z, self.temperature, eval=eval)


class NeutralADAnomalyDetector(AnomalyDetector):
    def __init__(self, model: NeutralAD, loss: NeutralADLoss):
        super().__init__()
        self.model = model
        self.loss = loss

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        with torch.no_grad():
            z = self.model(inputs)
        return self.loss((z,), (), eval=True)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        label, = targets
        return label[:, -1]
