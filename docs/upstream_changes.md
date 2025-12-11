# Differences from upstream TimeSeAD

This repository includes several fixes and performance improvements relative to the upstream [`wagner-d/TimeSeAD`](https://github.com/wagner-d/TimeSeAD) codebase.

## Bug fixes

- **Independent normalization per TimesNet block**: Each TimesNet layer now owns its own `LayerNorm` instance instead of sharing a single module across the stack, keeping affine parameters and statistics local to each block.【F:timesead/models/reconstruction/timesnet.py†L91-L114】
- **Stable non-stationary normalization**: The mean and standard deviation used for the Non-stationary Transformer style normalization are detached from the computation graph and include a datatype-specific epsilon, avoiding gradient flow through the statistics and preventing divide-by-zero issues.【F:timesead/models/reconstruction/timesnet.py†L99-L120】

## Performance improvements

- **On-device period search and padding in TimesNet**: FFT-based period detection now stays on tensors (no NumPy round-trips), zeros the DC component, and uses `topk`/`torch.div` to derive valid periods before padding and folding windows with tensor views, reducing CPU-GPU transfers and reshaping overhead.【F:timesead/models/reconstruction/timesnet.py†L14-L69】
- **FEDformer input handling**: Encoder inputs are materialized as contiguous tensors before embedding, eliminating potential slow paths or correctness issues when receiving strided slices.【F:timesead/models/reconstruction/fedformer.py†L79-L82】
- **AutoCorrelation layer vectorization**: Time-delay aggregation gathers all shifted patterns in a batched fashion and reuses a single fused QKV projection when queries, keys, and values are shared, removing Python loops and redundant projections in the autocorrelation attention used by FEDformer and related models.【F:timesead/models/layers/autocorrelation.py†L27-L242】
- **Fourier attention efficiency**: Fourier blocks and cross-attention store frequency indices as buffers, operate directly on complex weights, and replace per-mode Python loops with batched `matmul`/`einsum` operations, improving throughput for FEDformer’s frequency-domain attention.【F:timesead/models/layers/fourier_correlation.py†L12-L174】
