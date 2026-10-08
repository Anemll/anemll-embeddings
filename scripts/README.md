# Download and warm-up

`download_models.py` fetches the public Neural Engine packages and the Google host checkpoint. `warmup.py` loads each tower once so Core AI specializes them for this Mac and caches the result. There is **no** per-hardware compile to ship.

No Hugging Face login or token is needed. Both repos are public and ungated. The download script needs `huggingface_hub`, which `python -m pip install -e .` (or `.[demo]`) already installs.

Validated on **M4 Pro, macOS 27.0**. On **macOS 27.2 / M5** the ANE pre-check currently rejects vision and text (`invalid MLIR-MPS program`) and they fall back to the GPU; audio still runs on the ANE.

```sh
python scripts/download_models.py
# or: ./scripts/download_models.sh
python scripts/warmup.py
```

Put the printed `export` lines in `~/.zshrc` or a file you `source`.

## What gets downloaded

About **2.8 GB** on disk.

| What | Source | Size |
| --- | --- | --- |
| ANE towers (`vision_s280`, `audio_s280`, `text_embeds_s320`) | [anemll/anemll-embeddinggemma-2-ane](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane) `@ 8ceba04` | **~1.19 GB** (vision 307 MB, audio 589 MB, text_embeds 291 MB) |
| Host checkpoint (tokenizer, processor, embed table) | [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) `@ 914f7f89142e33e77833254d9c9b90c3cef7303b` | **~1.53 GB** (`model.safetensors` 1.49 GB / 1,488,915,288 bytes, plus `tokenizer.json` 32 MB, `tokenizer.model` 4.7 MB, and small config/processor files) |

Weights are Apache-2.0 under Google’s terms. This repo’s code is MIT.

## On-disk layout

Default `--dest` is `~/.anemll-embeddings` (or `$ANEMLL_EMBEDDINGS_HOME`). `api.Embedder` reads `$ANEMLL_EMBEDDINGS_ARTIFACTS/coreai/<name>.aimodel`.

```
~/.anemll-embeddings/
  ane/
    vision_s280/vision_s280.aimodel/     # metadata.json, main.hash, main.mlirb
    audio_s280/audio_s280.aimodel/
    text_embeds_s320/text_embeds_s320.aimodel/
  embeddinggemma-2/                      # Google host checkpoint
  artifacts/coreai/
    vision_s280.aimodel -> …/ane/vision_s280/vision_s280.aimodel
    audio_s280.aimodel -> …/ane/audio_s280/audio_s280.aimodel
    text_embeds_s320.aimodel -> …/ane/text_embeds_s320/text_embeds_s320.aimodel
```

## Flags

`python scripts/download_models.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dest PATH` | `~/.anemll-embeddings` | Parent directory. `$ANEMLL_EMBEDDINGS_HOME` overrides the default. |
| `--force` | off | Re-download even if the pinned revision is already on disk. Otherwise the script skips. |
| `--coreai-python PATH` | `$ANEMLL_COREAI_PYTHON` or the forge venv if present | Value printed for `ANEMLL_COREAI_PYTHON`. |

Revisions are pinned in the script (`8ceba04` and `914f7f8…`). There is no `--revision` flag. Skip-if-present is the default; use `--force` to fetch again.

Custom dest:

```sh
python scripts/download_models.py --dest /Volumes/Models/anemll-embeddings
```

`python scripts/warmup.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--artifacts PATH` | `$ANEMLL_EMBEDDINGS_ARTIFACTS` | Directory that contains `coreai/<name>.aimodel` |
| `--coreai-python PATH` | `$ANEMLL_COREAI_PYTHON` | Interpreter that can `import coreai.runtime` |
| `--compute ane\|cpu` | `ane` | Device for this load |

## Example output

`scripts/download_models.py` (copy these, or add them to `~/.zshrc`):

```sh
# example output of scripts/download_models.py
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Users/you/.anemll-embeddings/artifacts
export ANEMLL_EMBEDDINGS_MODEL=/Users/you/.anemll-embeddings/embeddinggemma-2
export ANEMLL_COREAI_PYTHON=/path/to/anemll-forge/coreai/.venv/bin/python
```

`scripts/warmup.py` (M4 Pro / macOS 27.0; first load is slower than later ones):

```
# example output of scripts/warmup.py
tower                  on ANE                load_ms   first_run_ms
vision_s280            yes                    4521.0          337.0
audio_s280             yes                     890.1           10.8
text_embeds_s320       yes                    2100.4           34.8
overall_placement=fullyOnANE  wall_ms=7800.0
```

## Troubleshooting

- A slow first warmup is normal. Core AI compiles for this chip and writes `~/Library/Caches/coreai-cache`. Later loads reuse that cache.
- On macOS 27.2 / M5, vision and text may print a GPU-fallback / `invalid MLIR-MPS program` message. Audio still runs on the ANE.
- Rerunning either script is safe. Download skips files that already match the pinned revision.
- To force a recompile: `rm -rf ~/Library/Caches/coreai-cache` then run `python scripts/warmup.py` again.
