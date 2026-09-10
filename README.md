# llama.cpp — Rhonstin fork

Personal fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
(MIT) carrying the **Qwen3.8-Flash-Next (qwen4exp)** inference stack used on
the llmcmp box: 8 GPUs (1x RTX 3090 + 7x CMP 90HX), `llama-server`, MTP
speculative decoding, QSA sparse attention.

`main` is the integration branch. `master` mirrors upstream.

## Branches

| branch | what it holds |
|---|---|
| `main` | current llmcmp stack + the phase-prefill port |
| `phase-prefill` | E06 prefill phase-memory transaction, see `PHASE-PREFILL-LLMCMP.md` |
| `qwen4exp-pooled-key-cache` | upstream submission of the QSA pooled-key cache ([ggml-org/llama.cpp#28699](https://github.com/ggml-org/llama.cpp/pull/28699)) |
| `cuda-radix-topk` | archive of the closed upstream PR #28366 (CUDA radix TOP_K fallback); both commits are cherry-picked into `main` and compile out when CUB DeviceTopK is available |
| `master` | mirror of upstream |

## Work carried in this fork

### QSA pooled-key cache (decode)

Qwen3.8-Flash-Next's QSA sparse-attention indexer rescores position blocks on
every decode step. This cache keeps one f32 pooled row per position block per
indexer layer and maintains it incrementally (per-sequence block watermark,
dirty tables instead of full rescans). One buffer per device: a single shared
buffer regressed layer-split decode about 2x because rows crossed PCIe.
Kill switch: `LLAMA_QSA_NO_POOLED_CACHE=1`.

Measured on llmcmp (UD-Q3_K_XL, MTP n-max 2, q8_0 KV, ctx 131072):
decode **+9.3%** at ~63K and **+9.4%** at ~114K, prefill parity, 64-token
greedy output bit-identical to the baseline. Submitted upstream as
[ggml-org/llama.cpp#28699](https://github.com/ggml-org/llama.cpp/pull/28699).

### Phase-prefill memory transaction

Port of Inovello's E06 idea: during a prompt, release the decode-only device
state (scheduler compute buffers, CUDA graphs, scratch pools), re-reserve the
target context for a larger prefill micro-batch, run the prompt, restore
everything before the first generated token. Enabled with
`LLAMA_PHASE_PREFILL_UBATCH=512|1024|2048`; see `PHASE-PREFILL-LLMCMP.md` for
adaptations, invariants and limitations.

Measured on the same box: prefill **+20% / +15% / +4%** at ~6K / 47K / 90K
prompts versus the same binary with the transaction disabled, decode and MTP
acceptance unchanged, greedy output identical across all arms. ub1024/2048 do
not fit at 256K context on 10 GB cards.

### llmcmp production stack (in `main`)

Built on upstream `master` plus the patches this box needed: Qwen3.8 MTP
speculative decoding with an external draft head, PLE direct reads
(`--lazy-mode on-direct`), gather-based QSA attention, CUDA top-k through CUB
DeviceTopK (CCCL 3.x headers against CUDA 12.4), and the 8 MB argsort chunk
for the CUB fallback. Measurement logs live in the operator's private wiki.

## Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build -j -t llama-server
```

For the DeviceTopK path, point CUDA 12.4 at a newer CCCL (CUB >= 3.2):

```bash
cmake ... -DCMAKE_CUDA_FLAGS=-I/path/to/include/cccl
```

Without it the build uses the argsort fallback — which is why the 8 MB chunk
patch exists.

## Credits and license

Upstream llama.cpp: ggml-org and contributors, MIT (see `LICENSE`).
QSA pooled-key cache original implementation: Joakim Hansson (apepojken).
Phase-prefill transaction design: Inovello (`flashnext-e06`).
Everything else here is fork-specific porting, adaptation and measurement.
