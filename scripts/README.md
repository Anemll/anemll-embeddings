# Download and warm-up

`download_models.py` fetches the **inference** assets: the public Neural Engine packages plus the slim Google host files (tokenizer, processor, embed table). It does **not** download `model.safetensors`. `download_export_assets.py` is a separate script for the full checkpoint if you will re-convert. `warmup.py` loads each tower once so Core AI specializes them for this Mac and caches the result. There is **no** per-hardware compile to ship.

No Hugging Face login or token is needed. Both repos are public and ungated. The download scripts need `huggingface_hub`, which `python -m pip install -e .` (or `.[demo]`) already installs.

Validated on **M4 Pro, macOS 27.0**. On **macOS 27.2 / M5** the ANE pre-check currently rejects vision and text (`invalid MLIR-MPS program`) and they fall back to the GPU; audio still runs on the ANE.

```sh
python scripts/download_models.py
# or: ./scripts/download_models.sh
python scripts/warmup.py
```

Put the printed `export` lines in `~/.zshrc` or a file you `source`.

## What gets downloaded (inference)

About **1.49 GB** on disk. `scripts/download_models.py` only.

| What | Source | Size |
| --- | --- | --- |
| ANE towers (`vision_s280`, `audio_s280`, `text_embeds_s320`) | [anemll/anemll-embeddinggemma-2-ane](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane) `@ 8ceba04` | **~1.19 GB** (vision 307 MB, audio 589 MB, text_embeds 291 MB) |
| Host tokenizer / processor / configs | [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) `@ 914f7f89142e33e77833254d9c9b90c3cef7303b` | **~37 MB** (`tokenizer.json` 32.2 MB, `tokenizer.model` 4.7 MB, plus `config.json`, processor / preprocessor configs, tokenizer config, chat template) |
| Embed table `embed_tokens.safetensors` | extracted from that checkpoint (HF range request, not a full-weights download) | **256 MiB** (268,435,456 bytes, BF16 `[262144, 512]`, plus Gemma `sqrt(512)` scale) |

Weights are Apache-2.0 under Google’s terms. This repo’s code is MIT.

## On-disk layout

Default `--dest` is `~/.anemll-embeddings` (or `$ANEMLL_EMBEDDINGS_HOME`). `api.Embedder` reads `$ANEMLL_EMBEDDINGS_ARTIFACTS/coreai/<name>.aimodel` and loads `embed_tokens.safetensors` when present (falls back to a full checkpoint).

```
~/.anemll-embeddings/
  ane/
    vision_s280/vision_s280.aimodel/     # metadata.json, main.hash, main.mlirb
    audio_s280/audio_s280.aimodel/
    text_embeds_s320/text_embeds_s320.aimodel/
  embeddinggemma-2/                      # slim host (no model.safetensors)
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

## Re-export the packages yourself

Only if you will run `model/export_coreai_towers.py`. Everyday inference does not need this.

```sh
python scripts/download_export_assets.py
```

**What gets downloaded** (about **1.53 GB** on disk):

| What | Source | Size |
| --- | --- | --- |
| Full Google checkpoint (conversion weights) | [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) `@ 914f7f89142e33e77833254d9c9b90c3cef7303b` | **~1.53 GB** (`model.safetensors` 1.49 GB / 1,488,915,288 bytes, plus tokenizer / processor / configs) |

Writes `~/.anemll-embeddings/embeddinggemma-2-full` (separate from the slim inference dir). Same `--dest` / `--force` / `--coreai-python` flags as the inference script.

```sh
# example output of scripts/download_export_assets.py
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Users/you/.anemll-embeddings/artifacts
export ANEMLL_EMBEDDINGS_MODEL=/Users/you/.anemll-embeddings/embeddinggemma-2-full
export ANEMLL_COREAI_PYTHON=/path/to/anemll-forge/coreai/.venv/bin/python
# conversion also needs torch, transformers, sentence-transformers
# and a Python that can import coreai.runtime (ANEMLL_COREAI_PYTHON)
```

## Troubleshooting

- A slow first warmup is normal. Core AI compiles for this chip and writes `~/Library/Caches/coreai-cache`. Later loads reuse that cache.
- On macOS 27.2 / M5, vision and text may print a GPU-fallback / `invalid MLIR-MPS program` message. Audio still runs on the ANE.
- Rerunning either inference or warmup is safe. Download skips files that already match the pinned revision.
- To force a recompile: `rm -rf ~/Library/Caches/coreai-cache` then run `python scripts/warmup.py` again.
