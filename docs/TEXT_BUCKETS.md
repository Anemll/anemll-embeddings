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

Wall time includes about 1.3 ms (bucket) to 2.8 ms (packed feed) of worker
round trip per call (npz file, JSON line, copies).

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
    --buckets 32,64,128,256,320 --pack 128x8,256x8,256x16 --layout multi
# then put text_buckets.aimodel next to text_embeds_s320.aimodel and
CFFIXED_USER_HOME=/tmp/fresh-cache python model/bench_text_buckets.py \
    --artifacts <artifacts dir> --model <host model dir> --json bench.json
```
