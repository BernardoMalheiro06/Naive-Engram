# Naive-Engram (PyTorch reference implementation)

A from-scratch PyTorch implementation of the **Engram** architecture from DeepSeek's paper, covering the hashed embedding lookup, context-aware gating, and short convolution components. This README documents `torch_engram.py`, it is a pure PyTorch reference implementation with no external kernel dependencies.

## Overview

Engram augments a model with a sparse, hash-based memory lookup that retrieves embeddings based on recent local context (n-gram-like "engrams"), then blends that retrieved information back into the residual stream through a gating mechanism and a short causal convolution.

`torch_engram.py` implements four modules:

### `EmbeddingLookup`
Hashes short windows of the input token sequence (of length `engram_size`) into indices, then looks those indices up across `n_tables` independent embedding tables whose outputs are concatenated.

- Builds `n_tables` sets of `engram_size` odd, pairwise-distinct multipliers, each coprime with `hash_vocab_size`, used for a multiplicative hash.
- `hash()` left-pads the sequence, slides a window of size `engram_size` over it (`unfold`), multiplies each window by the multipliers, and XORs the results together before reducing mod `hash_vocab_size`.
- `forward()` looks up the resulting hash indices in each table and concatenates the per-table embeddings along the feature dimension.

### `ContextAwareGating`
A lightweight multi-head attention-style gate that lets each token's hidden state `x` selectively read from its `retrieved_embeds` (the output of `EmbeddingLookup`).

- Splits both `x` and `retrieved_embeds` into `n_heads` heads.
- Projects the retrieved embeddings into key/value pairs (`W_k`, `W_v`); the hidden state itself acts as the query.
- RMS-normalizes queries and keys, computes a per-head scalar gating score (dot product, scaled by `sqrt(D_head)`), and uses it to scale the value.

### `ShortConv`
A causal, dilated depthwise-style convolution (dilation = `engram_size`) applied to the gated signal, with RMSNorm, SiLU activation, and a residual connection.

### `Engram`
Combines the two above: computes the gate output, passes it through the short convolution, and returns `x + short_conv(gate_output) + gate_output`.

## Requirements

- Python 3
- PyTorch (CPU or CUDA)

No Triton, custom CUDA extensions, or other GPU-specific dependencies are needed to run this file.

## Usage

```python
import torch
from torch_engram import EmbeddingLookup, Engram

device = "cuda" if torch.cuda.is_available() else "cpu"

hash_tables = EmbeddingLookup(
    hash_vocab_size=128,
    embed_size=768,
    n_tables=6,
    engram_size=2,
    device=device,
)

engram = Engram(
    embed_size=768,
    n_heads=6,
    kernel_size=4,
    engram_size=2,
    device=device,
)

tokens = torch.randint(low=1, high=128, size=(32, 1024), dtype=torch.long, device=device)

retrieved = hash_tables(tokens)     # (B, S, embed_size)
output = engram(retrieved, retrieved)  # (B, S, embed_size)
```

Note that every operation here is a standard, differentiable PyTorch op, so this module trains end-to-end with ordinary autograd, no custom backward pass is required.

## Status

Complete and functional as a reference implementation. Not yet benchmarked or optimized for throughput.

## Future Work

- **Triton kernels**: port the hashing step and the `ContextAwareGating` forward pass to Triton for GPU throughput (a first pass already exists in `triton_engram.py` / `helpers.py`, including a working forward kernel for the multiplicative XOR hash and for context-aware gating). This is the next thing I plan to work on, including writing the missing custom backward kernel (`cag_bwd_kernel`) so the accelerated version is trainable, not just forward-only.
- Correctness tests comparing the Triton-accelerated outputs (and gradients, once implemented) against this PyTorch reference implementation.
- Benchmark `torch_engram.py` itself against a naive attention-based memory-lookup baseline.
