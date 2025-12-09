# Differences from upstream TimeSeAD

This repository includes several fixes and performance improvements relative to the upstream [`wagner-d/TimeSeAD`](https://github.com/wagner-d/TimeSeAD) codebase.

## Bug fixes

- **Independent normalization per TimesNet block**: Each TimesNet layer now owns its own `LayerNorm` instance instead of sharing a single module across the stack. This prevents the trainable affine parameters from being shared between blocks and keeps their statistics independent.【F:timesead/models/reconstruction/timesnet.py†L94-L117】
- **Stable non-stationary normalization**: The mean and standard deviation used for the Non-stationary Transformer style normalization are detached from the computation graph, avoiding gradient flow through the statistics and preventing inadvertent updates during backpropagation.【F:timesead/models/reconstruction/timesnet.py†L103-L123】

## Performance improvements

- **TimesNet padding and FFT handling**: The FFT period finder and block processing were vectorized to avoid Python-side loops and redundant tensor materialization, reducing overhead when reshaping/padding windows by leveraging `torch.topk`, `torch.div`, and tensor views.【F:timesead/models/reconstruction/timesnet.py†L14-L79】
- **FEDformer input handling**: Encoder inputs are materialized as contiguous tensors before embedding, eliminating potential slow paths or correctness issues when receiving strided slices.【F:timesead/models/reconstruction/fedformer.py†L73-L87】
- **AutoCorrelation layer vectorization**: Time-delay aggregation now gathers all shifted patterns in a batched fashion and reuses a single fused QKV projection when queries/keys/values are shared. This removes Python loops and redundant projections in the autocorrelation attention used by FEDformer and related models.【F:timesead/models/layers/autocorrelation.py†L20-L205】
- **Fourier attention efficiency**: Fourier blocks and cross-attention now store frequency indices as buffers, operate directly on complex weights, and replace per-mode Python loops with batched `einsum`/matmul operations, improving throughput for FEDformer’s frequency-domain attention.【F:timesead/models/layers/fourier_correlation.py†L11-L164】
