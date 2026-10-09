# Text buckets and packed batches (prototype)

Status: prototype on branch `feat/text-buckets-packing`. Nothing here is on
Hugging Face yet; `scripts/download_models.py` still fetches only
`text_embeds_s320`. With no extra packages installed the runtime behaves
exactly as before.

## Why

`text_embeds_s320` has a fixed 320-token shape, so a 15-token query costs
the same 35 ms as a 300-token document, and there was no way to embed many
short texts in one call. Two additions fix that:

- **Buckets**: the same text graph exported at a smaller fixed S
  (`text_embeds_s32`, `s64`, `s128`, `s256`). The host picks the smallest
  bucket that holds the text. Longer texts still use `s320` (and are cut to
  320 tokens, as before).
- **Packed batches**: `text_pack_<N>x<T>` runs up to T short texts laid end
  to end in one N-token sequence. Texts never see each other, so each row
  equals the one-text result.

The packing design (block-diagonal bias, positions that restart per text,
a pooling matrix) follows the published description of
FluidInference/embeddinggemma-2-coreml. No code was taken from that repo; its
card is marked Apache-2.0, and its license should be checked before reusing
anything from it.

## Packed tower inputs (all f16)

| Input | Shape | Meaning |
|---|---|---|
| `inputs_embeds` | `[1, N, 512]` | token rows (host lookup), zeros after the last text |
| `attention_bias` | `[1, 1, N, N]` | 0 inside a text's own block, -1e4 elsewhere; padding tokens see only themselves |
| `positions` | `[N, 1]` | RoPE position, restarting at 0 for each text |
| `pool` | `[T, N]` | row t is 1/len over text t's tokens, an all-zero row is an unused slot |

Output `embedding` `[T, 768]` is the projected mean per text, not normalized
(an empty slot would divide by zero in fp16); the host L2-normalizes the
rows it filled. Every text is at most N tokens, below the 1024-token sliding
window, so the sliding and full attention layers take the same bias.

Graph notes (see `model/text_pack.py`):

- RoPE tables for positions `0..N-1` are constants. Each token's row is
  picked with a one-hot matrix built by arithmetic, `relu(1 - |p - j|)`,
  then a matmul. No sin/cos and no integer gather in the graph.
- Pooling is written as `hidden^T @ pool^T`, then a transpose. The direct
  `pool @ hidden` (and the same as a 4-D matmul) put the whole tower on the
  GPU on macOS 27.0 with no error message; `hidden^T @ pool^T` and an
  elementwise multiply + sum both stay on the ANE.

## Results (M4 Pro, macOS 27.0, fresh Core AI cache)

Every function below is fully on the ANE, both as separate packages and as
one combined package. Cosine is against the shipped `text_embeds_s320` on
the 12 prompts in `tests/fixtures/prompts.json` (each bucket on the
prompts that fit it).

| Function | min cosine vs s320 | warm p50, one call |
|---|---|---|
| `text_embeds_s32` | 0.999993 | 3.2 ms |
| `text_embeds_s64` | 0.999993 | 4.4 ms |
| `text_embeds_s128` | 0.999993 | 7.2 ms |
| `text_embeds_s256` | 0.999993 | 15.0 ms |
| `text_embeds_s320` (shipped) | 1 | 33.9 ms |
| `text_pack_128x8` | 0.999992 | 7.1 ms (8 texts) |
| `text_pack_256x8` | 0.999990 | 14.9 ms (8 texts) |
| `text_pack_256x16` | 0.999993 | 15.4 ms (16 texts) |

A 15-token query through `Embedder.embed_text` (tokenize, lookup, worker
round trip, tower): 37.5 ms on `s320`, 4.2 ms with buckets.

Throughput with `embed_texts` (512 lines of Python from `api/`, median 18
tokens with the document prefix; best of 3):

| Plan | texts/s (wall) | texts/s (tower time only) |
|---|---|---|
| `s320`, one text per call | 27.5 | 29.3 |
| buckets, one text per call | 237.5 | 338.0 |
| packed, `text_pack_256x8` only | 417.7 | 519.6 |
| packed, `text_pack_128x8` only | 611.2 | 860.4 |
| packed, `text_pack_256x16` only | 666.4 | 826.8 |
| auto (picks `text_pack_256x16`) | 662.5 | 818.5 |

Short posts (192, median 21 tokens): 27.7 texts/s on `s320`, 243.6 with
buckets, 623.8 packed (auto). Every packed row matched its `s320` result
with cosine >= 0.999988.

Tower-time figures repeat within a few percent between runs. Wall figures
do not: a later run on the same Mac (combined package, scratch and Core AI
cache on an external SSD instead of the internal one) gave 23.6 / 157.8 /
469.3 texts/s wall for `s320` / buckets / auto on the same lines, with 29.2 /
329 / 758 tower-time. The gap is host-side per-call cost (npz file and JSON
line to the worker), so it shows most with many small calls. Treat the wall
numbers as 15x to 25x over `s320` and the tower-time as the steady part.

Wall time includes about 1.3 ms (bucket) to 2.8 ms (packed feed) of worker
round trip per call (npz file, JSON line, copies).

## Results on Apple M5 Max, macOS 27.2

Independent run on a different chip and OS (reported by a separate test of
draft PR #16, head `0f3b2ee`; M5 Max, macOS 27.2). Same code, fresh Core AI
cache.

- Placement: all eight functions (`text_embeds_s32`, `s64`, `s128`, `s256`,
  `text_pack_128x8`, `text_pack_256x8`, `text_pack_256x16`, and the
  prototype package's own `text_embeds_s320`) are `fullyOnANE`.
- Parity: cosine >= 0.99999 against the shipped `text_embeds_s320`.
- Tests: 181 passed (the same unit suite), hardware tests passed.

| Function | warm p50, one call (M5 Max, macOS 27.2) | M4 Pro, macOS 27.0 |
|---|---|---|
| `text_embeds_s32` | 2.8 ms | 3.0 ms |
| `text_embeds_s64` | 3.6 ms | 4.3 ms |
| `text_embeds_s128` | 6.4 ms | 7.1 ms |
| `text_embeds_s256` | 13.5 ms | 15.4 ms |
| `text_embeds_s320` (shipped) | 31.5 ms | 35.1 ms |

Packed batches ran at about 747-779 texts/s on the M5 Max (as reported; the
report does not say whether that is wall or tower time, and the M5 Max pack
p50 values were not part of it). For comparison, the M4 Pro packed plans
above gave 418-666 texts/s wall and 520-860 tower time in the first run, and
307-481 wall and 485-843 tower time in the later run. The M4 Pro column in
the table is that later run (combined package, external SSD); the first
table above is the earlier run.

## Package sizes and Core AI multi-function packages

Each separate package carries its own copy of the weights:

| Package | `main.mlirb` |
|---|---|
| `text_embeds_s32` | 274.8 MB |
| `text_embeds_s64` | 275.7 MB |
| `text_embeds_s128` | 278.1 MB |
| `text_embeds_s256` | 284.6 MB |
| `text_pack_128x8` | 275.8 MB |
| `text_pack_256x8` | 280.0 MB |
| `text_pack_256x16` | 280.0 MB |
| all seven, separate | 1949 MB |
| shipped `text_embeds_s320` | 290.9 MB |

Core AI supports several functions in one `.aimodel`: `coreai_torch`'s
`TorchConverter.add_exported_program` takes an `entrypoint_name`, can be
called once per program, and `to_coreai()` converts them together
(`model/_coreai_convert_multi.py`). Equal weight blobs are stored once. One
package with `text_embeds_s320`, the four buckets and the three packs is
**311.9 MB**, 21 MB more than `text_embeds_s320` alone. Each function keeps
its own fixed shapes (there is no shape enumeration inside one function);
`AIModel.function_names` lists them and `load_function(name)` loads one.
The first load specializes every function in the package for the chip.

So shipping one combined `text_buckets.aimodel` (or replacing
`text_embeds_s320` with it) costs about 21 MB, not about 2 GB.

### Decision: the combined package does not carry its own `text_embeds_s320`

The first prototype package also contained a re-exported `text_embeds_s320`.
Compared on an M4 Pro (macOS 27.0, fresh cache, 162 texts: the 12 fixtures
plus 150 source lines, host path with extras off, so every text runs on the
one function):

| | shipped `text_embeds_s320` package | `text_embeds_s320` inside the combined package |
|---|---|---|
| placement (per-function manifest) | fullyOnANE | fullyOnANE |
| cosine between the two | 1.0 (min over 162 texts) | |
| end-to-end p50, one short query | 42.4 ms | 42.5 ms |
| bytes in `main.mlirb` | 290.9 MB (own package) | +17.4 MB (311.9 MB vs 294.5 MB without it) |

The outputs are identical and the speed is the same, so the copy adds only
17.4 MB of download and one more function to specialize at first load. The
host never used it (the shipped tower is loaded first and a function with the
same name is skipped). The combined package therefore holds only the four
buckets and the three packs: export with `--buckets 32,64,128,256`
(294.5 MB, about 3.6 MB more than `text_embeds_s320` alone).

### Placement caveat for a fresh multi-function export

On the same M4 Pro, the combined package built on 2026-10-09 10:14 (with
`s320`) is `fullyOnANE` for every function (fresh cache, `scripts/warmup.py
--require-ane` exits 0 again today). Four later exports from the same code
(`--buckets 32,64,128,256` with and without `s320`, with and without
`--skip-check`, buckets only, packs only) all load but report every extra
function as GPU, while a separate single-function `text_embeds_s32.aimodel`
exported today is fully on the ANE. The cause is not found yet; it is not
the extra `s320`, the pack math check, or free disk space on the home volume
(3 GB or more was free during the compiles). `scripts/warmup.py --require-ane`
now catches this per function, and the host ignores extras that are not fully
on the ANE, so such a package is harmless but gives no speed-up. Re-check any
final export with `warmup.py --require-ane` on a fresh cache before it is
uploaded.

## Host

- `api/text_batch.py` (numpy only): bucket choice, pack planning
  (first-fit decreasing), feeds, Matryoshka truncation, and a cost model
  that picks one-text-per-call or one of the packed towers using each
  tower's warm latency (measured by the worker at load).
- `api/coreai_worker.py` loads every function of each extra text package,
  keyed by function name, and reads per-function placement from the
  compiled manifest. An extra that fails to load or is not fully on the ANE
  is ignored; the main three towers are unaffected.
- `api/embedder.py`: `embed_text` uses the smallest loaded bucket,
  `Embedder.embed_texts(texts, role=..., dim=..., pack=True)` embeds a list
  in order. Extras are found next to `text_embeds_s320.aimodel`
  (`text_buckets.aimodel` wins over single-function packages). Set
  `ANEMLL_TEXT_BUCKETS=0` to ignore them.

## Reproduce

```bash
# export (embeddings venv; conversion uses ANEMLL_COREAI_PYTHON)
python model/export_text_buckets.py --artifacts ~/text-buckets \
    --buckets 32,64,128,256 --pack 128x8,256x8,256x16 --layout multi
# then put text_buckets.aimodel next to text_embeds_s320.aimodel, check it
python scripts/warmup.py --require-ane   # every function must say yes, on a fresh cache
# and
CFFIXED_USER_HOME=/tmp/fresh-cache python model/bench_text_buckets.py \
    --artifacts <artifacts dir> --model <host model dir> --json bench.json
```

## Swift API (macOS 27 SDK)

The macOS 27 SDK ships `CoreAI.framework` (re-exports `CoreAIDelegates`) and
`CoreAIRuntime.framework` (under `System/Library/SubFrameworks`), both with
`.swiftinterface` files and no C headers. `CoreAIRuntime` has `AIModel`
(`functionNames`, `functionDescriptor(for:)`), `InferenceFunction`
(`run(inputs: [String: NDArray], ...)` and an async `encode`), `NDArray`,
`NDArrayDescriptor`, `ComputeStream` and Metal-buffer interop. So a Swift app
can list and run the functions of a multi-function package such as
`text_buckets.aimodel` and pick a bucket or pack function per call. The Swift
adapter in this repo has not been changed; this branch only touches the
Python host.

## Open questions

- Ship the extra towers on Hugging Face (combined package or per-tower)?
  Not done: no HF upload from this branch. The download path is staged
  (placeholders in `scripts/download_common.py`, `hf/config.json`,
  `hf/towers.yaml`). Blocker: a fresh export must first land fully on the
  ANE (see the placement caveat above).
- Check the FluidInference repo license before any code reuse.
- Swift adapter: add bucket and pack selection there too?
- Measured on an M4 Pro (macOS 27.0) and an M5 Max (macOS 27.2). M3 Ultra and the base M5 are not measured with these towers.
