# Phase-prefill memory transaction (llmcmp port)

Experimental branch: a port of the E06 prefill phase-memory transaction to the
llmcmp production stack for **Qwen3.8-Flash-Next** (qwen4exp) on 8 GPUs.

Original design and reference implementation by [Inovello](https://github.com/Inovello):
branch `flashnext-e06` of `Inovello/llama.cpp` (commits `80aae9ab5` and
`6f31af3b4`), write-ups at https://www.inovello.dev/writeups/. That branch
targets a 2x RTX 3090 box with expert layers pinned in host DDR4 and an LRU
expert cache in VRAM; this port keeps the transaction but drops everything
that depends on the expert cache, and adapts the invariants to an 8-GPU
layer-split box that keeps all weights in VRAM.

This branch is based on the llmcmp production stack (`pooled-key`, `588fef344`):
Qwen3.8 MTP speculative decoding, PLE direct reads, QSA gather attention,
CUDA top-k with CUB DeviceTopK, and the QSA pooled-key cache.

## What the transaction does

During a prompt the decode-only device state sits idle: compute buffers sized
for tiny decode batches, CUDA graph execs, scratch pool blocks. The transaction

1. synchronizes target and draft contexts and every CUDA stream,
2. invalidates CUDA graphs and releases the scheduler compute buffers,
3. drops the CUDA scratch pools (unmaps VMM high-water backing),
4. re-reserves the target context for a larger prefill micro-batch,
5. runs the whole prompt at that micro-batch,
6. restores the original micro-batch (and the scheduler allocations) before
   the first generated token.

Logical batch and sequence state are untouched; only physical compute capacity
changes. Every step is fail-closed: a transition that cannot complete aborts
the server instead of limping on, because the state is not recoverable.

## Files

| file | change |
|---|---|
| `include/llama.h` | `llama_experimental_prefill_begin/end` API |
| `src/llama-context.{h,cpp}` | transition implementation, invariants snapshot, granular pair checks, pipeline-parallel save/restore |
| `ggml/include/ggml-cuda.h` | `ggml_backend_cuda_phase_reset` extension + reset stages |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | phase reset: stream sync, graph invalidation, pool release; local `dynamic_cast` pool-size helper |
| `tools/server/server-context.cpp` | server hook: wrap prompt batches in the transaction |
| `src/llama-memory-hybrid-idx.cpp` | mark the qwen4exp indexer KV cache MLA-shaped so `llama_kv_cache` does not allocate unused V tensors |

## Adaptations versus the original branch

- **No expert cache.** PR #27861 does not exist in this tree and all experts
  are VRAM-resident, so cache quiesce/release/restore and the cache phase,
  generation and capacity guards were removed.
- **N target devices.** The original requires exactly two target CUDA devices;
  this port accepts any number (the box runs eight) and only checks that draft
  devices are a subset of target devices.
- **`dft->cparams.ctx_other == ctx_tgt` is allowed.** The MTP borrow-embeddings
  path in this tree always sets it. The "independent memory" requirement was
  split into granular checks that allow this linkage but still reject fully
  shared memory.
- **Pool size without touching `common.cuh`.** A new virtual on
  `ggml_cuda_pool` would rebuild every CUDA translation unit. The port instead
  uses a file-local `dynamic_cast` helper in `ggml-cuda.cu`.
- **Pipeline parallelism.** Upstream auto-enables PP on this box (8-GPU layer
  split, no `-ot` overrides; the original box disables it implicitly via
  overrides). The begin-side reserve drops PP when the larger buffers do not
  fit, so the port saves the flag at begin and restores it before the
  end-side reserve: prefill runs without PP, decode keeps it.
  `LLAMA_NO_PIPELINE_PARALLEL=1` force-disables PP for A/B.
- **Granular startup checks** with distinct messages instead of one opaque
  failure, easing future adaptation.

## Enabling it

```
export LLAMA_PHASE_PREFILL_UBATCH=512    # or 1024/2048
export LLAMA_PHASE_PREFILL_VERIFY=1      # optional: hash host output buffers around transitions
```

Requirements enforced at startup: single text-generation slot (`--parallel 1`),
`-b` >= 2048, `-ub` equal on target and draft, one MTP draft, CUDA only,
no mmproj/embedding modes.

## Measured on llmcmp

Box: 1x RTX 3090 + 7x CMP 90HX, dual Xeon, CUDA, Qwen3.8-Flash-Next UD-Q3_K_XL
with the Q4_K_M MTP draft (n-max 2), q8_0 KV, `ctx 262144`, `-b 4096` with
`-ub 128` start, single stream, prefill measured from `/completion` timings.

| arm | pp ~6K | pp ~47K | pp ~90K | smoke tg |
|---|---|---|---|---|
| control, PP on, transaction disabled | 219.1 | 217.7 | 200.9 | 17.4 |
| control, PP off | 201.1 | 176.0 | - | 15.6 |
| transaction ub512, PP save/restore | 262.7 | 250.5 | 209.2 | 17.5 |

- Prefill gain versus the prod-like control: **+20% / +15% / +4%** at ~6K /
  47K / 90K. The gain shrinks with context depth because the remaining cost is
  attention/indexer work, not batch-shape overhead.
- Decode is unchanged; MTP acceptance unchanged (smoke `draft=10/9`, deep 31/31).
- 64-token greedy output identical (sha `f1400edf41bdb73d`) across all nine
  measurement arms.
- Transition cost: begin 0.73-0.90 s, end 0.44-0.81 s per prompt; prompts of
  eight tokens or fewer skip the transaction.
- `ub 1024/2048` do not fit at `ctx 262144`: the begin-side reserve wants
  3.7-6.2 GB per device for the sparse indexer score graph. `ub 1024` fits at
  `ctx 160K` but is only +6% over `ub 512`.

## Limitations

Tested envelope: single slot, prompts 6K-90K, MTP enabled, q8_0 KV, full
256K context, PP save/restore. Not tested: prompt-cache reuse across requests,
state save/load, contexts above ~90K prompts, no-MTP configurations, other
quants. The transaction is CUDA-only and fail-closed: unsupported situations
abort the process rather than degrade.

Status: experimental. The llmcmp production service still runs the base
`pooled-key` build; deploying this branch additionally requires `-b 4096` and
`LLAMA_PHASE_PREFILL_UBATCH=512`.
