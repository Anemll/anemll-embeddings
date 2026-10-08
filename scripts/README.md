# Download and warm-up

`download_models.py` fetches the **inference** assets from one repo: the public Neural Engine packages, the root `config.json` package descriptor, and `host/` (tokenizer, processor, embed table) on [anemll/anemll-embeddinggemma-2-ane](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane) `@ 30fe9a770d417c35bedf716075dc9a41a957c9f4`. It does **not** download `model.safetensors`. If `host/` is missing on the pin, it falls back to the slim Google files. `download_export_assets.py` is a separate script for the full checkpoint if you will re-convert. `prepare_hf_host_folder.py` stages `hf/host/` for a Hub upload (writes only under `--dest`; does not upload). `warmup.py` loads each tower once so Core AI specializes them for this Mac and caches the result. There is **no** per-hardware compile to ship.

No Hugging Face login or token is needed. Both repos are public and ungated. The download scripts need `huggingface_hub`, which `python -m pip install -e .` (or `.[demo]`) already installs.

Fully on the ANE on **M4 Pro / M3 Ultra, macOS 27.0**. On **macOS 27.2 / M5** the ANE pre-check currently rejects vision and text (`invalid MLIR-MPS program`) and they fall back to the GPU; audio still runs on the ANE. `warmup.py --require-ane` turns that into a non-zero exit.

The packages run under a separate Core AI interpreter (`coreai-core` 1.0.0b2 on Python 3.13). Setup and lookup order: [README → Core AI runtime](../README.md#core-ai-runtime).

```sh
python scripts/download_models.py
# or: ./scripts/download_models.sh
python scripts/warmup.py
```

With the default `--dest` nothing needs exporting: `api.Embedder`, the samples, `warmup.py`, and the demo fall back to `~/.anemll-embeddings/artifacts` and `~/.anemll-embeddings/embeddinggemma-2`. With a custom `--dest`, put the printed (shell-quoted) `export` lines in `~/.zshrc` or a file you `source`.

Each fresh download is verified against SHA-256 digests tracked in `scripts/download_common.py` (`TOWER_MLIRB_SHA256` for each tower's `main.mlirb`, `ROOT_CONFIG_SHA256` for the root `config.json`, `HOST_SHA256` for every `host/` file) **before** the revision marker is written, so a corrupted or tampered file is never treated as installed. A mismatch stops the script with the file name and both digests. If you override `ANEMLL_ANE_REVISION`, the pinned table does not apply and verification is skipped with a note.

## What gets downloaded (inference)

About **1.49 GB** on disk. `scripts/download_models.py` only.

| What | Source | Size |
| --- | --- | --- |
| ANE towers (`vision_s280`, `audio_s280`, `text_embeds_s320`) | [anemll/anemll-embeddinggemma-2-ane](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane) `@ 30fe9a770d417c35bedf716075dc9a41a957c9f4` | **~1.19 GB** (vision 307 MB, audio 589 MB, text_embeds 291 MB) |
| Host tokenizer / processor / configs | same repo, `host/` (fallback: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) `@ 914f7f89142e33e77833254d9c9b90c3cef7303b`) | **~37 MB** (`tokenizer.json` 32.2 MB, `tokenizer.model` 4.7 MB, plus `config.json`, processor / preprocessor configs, tokenizer config, chat template) |
| Embed table `embed_tokens.safetensors` | same repo, `host/` (extracted from Google’s `model.safetensors`; not a full-weights download) | **256 MiB** (268,435,456 bytes, BF16 `[262144, 512]`, plus Gemma `sqrt(512)` scale) |
| Package descriptor `config.json` | same repo, root (towers, shapes, `host/` paths; not a transformers config) | **~3 KB** |

Weights are Apache-2.0 under Google’s terms. This repo’s code is MIT.

## On-disk layout

Default `--dest` is `~/.anemll-embeddings` (or `$ANEMLL_EMBEDDINGS_HOME`). `api.Embedder` reads `$ANEMLL_EMBEDDINGS_ARTIFACTS/coreai/<name>.aimodel` (default `~/.anemll-embeddings/artifacts`) and loads `embed_tokens.safetensors` when present (falls back to a full checkpoint).

```
~/.anemll-embeddings/
  ane/
    config.json                          # package descriptor (root config.json on the Hub)
    vision_s280/vision_s280.aimodel/     # metadata.json, main.hash, main.mlirb
    audio_s280/audio_s280.aimodel/
    text_embeds_s320/text_embeds_s320.aimodel/
  embeddinggemma-2/                      # slim host (from ane/host/ or Google fallback)
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
| `--verify` | off | Re-hash the towers and host files already on disk against the pinned digests (no download). |
| `--coreai-python PATH` | `$ANEMLL_COREAI_PYTHON`, then `~/.anemll-embeddings/coreai-venv`, then a sibling or `~/anemll-forge/coreai/.venv` | Value printed for `ANEMLL_COREAI_PYTHON`. If none exists, commented setup lines are printed instead. |

Revisions are pinned in `scripts/download_common.py` (`ANE_REVISION=30fe9a770d417c35bedf716075dc9a41a957c9f4`, overridable with `ANEMLL_ANE_REVISION`; Google fallback stays at `914f7f8…`). There is no `--revision` flag. Skip-if-present is the default; use `--force` to fetch again.

Custom dest:

```sh
python scripts/download_models.py --dest /path/to/fast/disk/anemll-embeddings
```

`python scripts/warmup.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--artifacts PATH` | `$ANEMLL_EMBEDDINGS_ARTIFACTS`, else `~/.anemll-embeddings/artifacts` | Directory that contains `coreai/<name>.aimodel` |
| `--coreai-python PATH` | `$ANEMLL_COREAI_PYTHON`, then `~/.anemll-embeddings/coreai-venv`, then a sibling or `~/anemll-forge/coreai/.venv` | Interpreter that can `import coreai.runtime`. A set-but-missing path is an error; with nothing found it stops with setup steps. |
| `--coreai-home DIR` | `$CFFIXED_USER_HOME`, else your home | Core AI's home; the specialization cache is `<DIR>/Library/Caches/coreai-cache`. Created if missing. |
| `--cache-dir PATH` | — | Same as `--coreai-home`, given as the full cache path. Must end in `Library/Caches/coreai-cache` (Core AI has no free-form cache location); anything else is rejected. Mutually exclusive with `--coreai-home`. |
| `--compute ane\|cpu` | `ane` | Device for this load |
| `--require-ane` | off | Exit **3** unless every tower reports fully on the Neural Engine. Cannot be combined with `--compute cpu`. |

## Example output

`scripts/download_models.py` (copy these, or add them to `~/.zshrc`):

```sh
# example output of scripts/download_models.py
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Users/you/.anemll-embeddings/artifacts
export ANEMLL_EMBEDDINGS_MODEL=/Users/you/.anemll-embeddings/embeddinggemma-2
export ANEMLL_COREAI_PYTHON=/Users/you/.anemll-embeddings/coreai-venv/bin/python
```

`scripts/warmup.py` — **cold** vs **warm** (M3 Ultra / macOS 27.0):

```
# cold first load — compile + cache, ~111 s
tower                  on ANE                load_ms   first_run_ms
vision_s280            yes                   58000.0            -
audio_s280             yes                   16000.0            -
text_embeds_s320       yes                   36000.0            -
overall_placement=fullyOnANE  wall_ms=111000.0
```

```
# warm load — cache hit, load ~0.06–0.1 s per tower
tower                  on ANE                load_ms   first_run_ms
vision_s280            yes                      80.0          360.0
audio_s280             yes                      80.0           25.0
text_embeds_s320       yes                      80.0           42.0
overall_placement=fullyOnANE
```

## Stage `host/` for the Hub (maintainers)

Does **not** upload. On the Mac that will push to
[anemll/anemll-embeddinggemma-2-ane](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane):

```sh
python scripts/prepare_hf_host_folder.py
# or: python scripts/prepare_hf_host_folder.py --src ~/.anemll-embeddings/embeddinggemma-2-full
# LICENSE/NOTICE default to hf/LICENSE and hf/NOTICE; override with --license / --notice
```

`--dest` is the Hub staging **root** (default `hf/` in this repo); files land in `<dest>/host/`, and nothing outside `<dest>/host/` is written. Without `--src`, the slim Google files are downloaded into a temporary directory that is deleted when the script exits (also on error). It never writes `embed_tokens.safetensors` into `--src`. `--license` and `--notice` are required files (defaults: `hf/LICENSE`, `hf/NOTICE`) and are copied into `host/`. Upload `hf/` (card, `towers.yaml`, `LICENSE`, `NOTICE`, `host/`) yourself.

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
# conversion also needs: pip install -e ".[runtime,reference]"
# and ANEMLL_COREAI_PYTHON pointing at the forge coreai/.venv
# (coreai-core 1.0.0b2 + coreai-torch 0.4.2 + coreai-opt 0.2.1, see README)
```

## Troubleshooting

- A **cold** first warmup of ~111 s on M3 Ultra is normal (compile + write the cache). A **warm** load is ~0.06–0.1 s per tower.
- If warmup dies while loading, the Core AI cache may be unwritable or a broken symlink (`~/Library/Caches/coreai-cache`). Fix that path, or redirect: `python scripts/warmup.py --coreai-home /path/to/writable/home` (or `export CFFIXED_USER_HOME=/path/to/writable/home`; the cache becomes `<home>/Library/Caches/coreai-cache`). Keep `CFFIXED_USER_HOME` set for later runs so the samples and demo reuse that cache.
- "Core AI Python not found" / "cannot import coreai.runtime": create the Core AI venv ([README → Core AI runtime](../README.md#core-ai-runtime)) or set `ANEMLL_COREAI_PYTHON`.
- `sha256 mismatch …`: the file on disk is not the pinned one. Delete that tower or `host/` folder and rerun with `--force`.
- On macOS 27.2 / M5, vision and text report `no (GPU)` (`invalid MLIR-MPS program`). Audio still runs on the ANE. `--require-ane` exits 3 there.
- Rerunning either inference or warmup is safe. Download skips files that already match the pinned revision.
- To force a recompile: `rm -rf ~/Library/Caches/coreai-cache` then run `python scripts/warmup.py` again.
