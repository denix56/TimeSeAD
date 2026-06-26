# Implementation from Time Series Library https://github.com/thuml/Time-Series-Library
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionBlockV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        # Cache for the folded (single-conv-equivalent) kernel, populated lazily in
        # eval where the branch weights are frozen. Invalidated on train() and on any
        # device/dtype move (_apply), so it never goes stale.
        self._folded = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _fold_kernels(self):
        # Exact algebraic fold of the num_kernels parallel convs into ONE conv.
        # The block computes (1/K)*sum_i conv(x, w_i, padding=i) over kernel sizes
        # 1,3,...,2K-1 (K=num_kernels). Convolution is linear in the kernel and all
        # branches use zero padding, so this equals a single convolution whose kernel
        # is the average of the per-branch kernels each zero-padded (centered) to the
        # max size (2K-1), with padding=K-1:
        #     (1/K) sum_i conv(x, w_i, pad=i) == conv(x, (1/K) sum_i pad_{K-1-i}(w_i), pad=K-1)
        # Output is identical up to floating-point reduction order (~1e-6); the fold is
        # differentiable, so training gradients reach each branch weight unchanged.
        P = self.num_kernels - 1
        merged_w = F.pad(self.kernels[0].weight, (P, P, P, P))
        for i in range(1, self.num_kernels):
            p = P - i
            w_i = self.kernels[i].weight
            merged_w = merged_w + (F.pad(w_i, (p, p, p, p)) if p > 0 else w_i)
        merged_w = (merged_w / self.num_kernels).contiguous(memory_format=torch.channels_last)

        merged_b = self.kernels[0].bias
        if merged_b is not None:
            for i in range(1, self.num_kernels):
                merged_b = merged_b + self.kernels[i].bias
            merged_b = merged_b / self.num_kernels
        return merged_w, merged_b

    def train(self, mode: bool = True):
        # Branch weights change during training, so drop the folded-kernel cache.
        if mode:
            self._folded = None
        return super().train(mode)

    def _apply(self, *args, **kwargs):
        # Any device/dtype move invalidates the cached folded kernel.
        self._folded = None
        return super()._apply(*args, **kwargs)

    def folded_kernel(self):
        # Return (weight, bias, padding) of the single conv equivalent to this block.
        # In training the fold is rebuilt (live, differentiable weights) -- caching is a
        # Python-side mutation that would force torch.compile graph breaks/recompiles,
        # and the cached graph would be freed after backward. In eval the weights are
        # frozen, so the fold is built once and cached. A caller that invokes the block
        # many times with the same weights (e.g. TimesBlock's per-period loop) can fold
        # once via this method and reuse the kernel, avoiding num_kernels*top_k rebuilds.
        if self.num_kernels == 1:
            k = self.kernels[0]
            return k.weight, k.bias, k.padding[0]
        if self.training or torch.compiler.is_compiling():
            merged_w, merged_b = self._fold_kernels()
        else:
            if self._folded is None:
                with torch.no_grad():
                    self._folded = self._fold_kernels()
            merged_w, merged_b = self._folded
        return merged_w, merged_b, self.num_kernels - 1

    def forward(self, x):
        # One conv launch instead of num_kernels, and fewer FLOPs (a single max-size
        # conv is cheaper than summing all the smaller convs).
        merged_w, merged_b, padding = self.folded_kernel()
        return F.conv2d(x, merged_w, merged_b, padding=padding)