# anemll-embeddings

Open-source **EmbeddingGemma 2** → **Core ML / Apple Neural Engine** via an ANEMLL-forge-style workflow.

## Goal

Convert Google’s EmbeddingGemma 2 text embedding models to run efficiently on Apple Silicon (Core ML → ANE), starting with the ~270M text-only checkpoint, then extending toward multimodal (vision/audio) where feasible.

## Sources

- Official model: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (prefer this checkpoint; ~1.53 GB safetensors)
- Community / Unsloth guide & variants: [unsloth/embeddinggemma-2](https://huggingface.co/unsloth/embeddinggemma-2), Unsloth EmbeddingGemma docs
- Conversion workflow reference (read-only): [Anemll/anemll-forge](https://github.com/Anemll/anemll-forge) (`docs/WORKFLOW.md`, `forge.py`)

## Model weights location

**Weights are not stored in this git repo.** Keep this checkout lean (code + small configs only).

Canonical models directory (TrueNAS SMB volume **TB36**):

`/Volumes/TB36/Models/anemll-embeddings`

- Checkpoint: `/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2` (`google/embeddinggemma-2`, ~1.5 GB safetensors)
- Do **not** use flash USB `/Volumes/SAN512` for new downloads (slow). A prior incomplete copy may still exist at `/Volumes/SAN512/MODELS/anemll-embeddings` (~702M partial); leave it unless cleaning up deliberately.
- Prefer not filling the internal SSD when free space is tight.

Set Hugging Face caches on TB36:

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
```

A local `models` symlink to that path is optional and gitignored.

## Reference fixtures (T1)

Fixed Sentence-Transformers text-only embeddings for parity work live under `tests/fixtures/`:

- `prompts.json` — stable SearchQuery / Document / CodeRetrieval / SentenceSimilarity / Classification / Clustering set
- `embeddings.npy` — float32 `[N, 768]` ST reference vectors (L2-normalized)
- `reference_meta.json` — load dtype/device, package versions, per-vector SHA-256 digests

Regenerate (weights on TB36; runtime venv must be **local**, not SMB — torch SIGBUSes from TB36):

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/gen_reference_fixtures.py
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_reference_fixtures.py
```

Python env: `/Volumes/Models/anemll-embeddings/.venv` (sentence-transformers≥6.1, transformers, torch, pillow, torchvision — processor import still pulls image deps even for text-only). Pip cache may stay on TB36. A TB36 `.venv` was attempted but native `torch` imports crash with SIGBUS over SMB.

## Artifacts disk

`.mlpackage` / `.mlmodelc` / TorchScript / build outputs go under:

`ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts`

Do **not** put compile artifacts on TB36 (models + HF caches only), SAN512, or the internal SSD.

## Text-only loader + wrapper (T2 / T3)

- `src/load_text_model.py` — Sentence-Transformers load with `vision_config`/`audio_config=None`, BF16/FP32 only (refuses FP16)
- `src/embed_wrapper.py` — `EmbeddingGemma2Wrapper`: mask-aware mean pool → 512→768 projection → optional L2 (pool-then-project graph ready for later `torch.jit.trace` / coremltools)

Smoke vs T1 fixtures (cosine ≥ 0.999):

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/smoke_wrapper_vs_fixtures.py
```

## Trace + Core ML convert (T4 / T5)

Fixed-S `torch.jit.trace` then public `coremltools==9.0` (see `requirements-conversion.txt`). This is **not** `forge.py convert`.

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_export_utils.py
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/export_torchscript.py --seq-len 512
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/convert_coreml.py --seq-len 512
```

Default `S=512` matches this checkpoint’s `text_config.sliding_window` (PLAN’s 1024 note is the later ladder). Artifacts land under `$ANEMLL_EMBEDDINGS_ARTIFACTS/embeddinggemma2-text-s512/` (gitignored).

Core ML I/O (names are fixed):

| Name | Role | Shape | Dtype |
| --- | --- | --- | --- |
| `input_ids` | token ids (host tokenizer + task prefix) | `[1, S]` | `int32` |
| `attention_mask` | `1` = token, `0` = pad | `[1, S]` | `int32` |
| `embedding` | L2-normalized 768-d vector | `[1, 768]` | `float32` |

`CPU_AND_NE` on convert does **not** prove ANE placement (T7). First package is FP32 compute (`ct.precision.FLOAT32`); FP16 is T10.

## Parity vs T1 fixtures (T6)

Core ML **CPU** vs committed `tests/fixtures/embeddings.npy` (cosine, rel-L2, L2 norms, pairwise `cos(q,d)`). This does **not** prove ANE.

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_parity_metrics.py
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/parity_cosine.py --seq-len 512
```

Gates (PLAN starting numbers): per-prompt cosine ≥ 0.999, rel-L2 ≤ 0.05, pairwise |Δcos| ≤ 0.01. JSON report: `$ANEMLL_EMBEDDINGS_ARTIFACTS/embeddinggemma2-text-s512/embeddinggemma2-text-s512.parity.json`.

## ANE smoke + placement (T7)

Same S=512 `.mlpackage`, scored with `CPU_AND_NE`. **T6 CPU parity is not ANE proof.** T7 dumps `MLComputePlan` preferred devices and fails if zero non-const ops prefer ANE.

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_ane_placement.py
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/ane_smoke.py --seq-len 512
```

Report: `$ANEMLL_EMBEDDINGS_ARTIFACTS/embeddinggemma2-text-s512/embeddinggemma2-text-s512.ane.json`.

## FP16 hazard experiment (T10)

Reuses the T4 FP32 TorchScript. Converts a **separate** `*-fp16` package (`ct.precision.FLOAT16`) so the T5 FLOAT32 artifacts stay put. PyTorch stays BF16/FP32.

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/convert_coreml.py --seq-len 512 --precision FLOAT16
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/parity_cosine.py --seq-len 512 --precision FLOAT16
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/ane_smoke.py --seq-len 512 --precision FLOAT16
```

Artifacts: `$ANEMLL_EMBEDDINGS_ARTIFACTS/embeddinggemma2-text-s512-fp16/`. Record NaN rate, cosine/rel-L2, and ANE compute-plan counts. Do not treat T6/T7 CPU numbers as ANE.

mp4 (M4 Pro) first run: FP16 **does** place on ANE (`2307` ANE / `17` CPU preferred). ANE vs T1 min cosine `0.99988`. CPU_ONLY FP16 vs T1 min cosine `0.920` (no NaNs) — the FP16 hazard is the **CPU** path, not ANE. Convert logged MIL `overflow encountered in cast` warnings.

`ane_smoke.py --precision FLOAT16` also times `CPU_AND_GPU` and `ALL` (warm + p50/p90) and classifies every CPU-preferred op as begin / mid / end. The 17 leftovers are a **begin-of-graph** island (pad/window mask + embedding gather). That is **not** a full-ANE graph: one CPU→ANE switch at the start is expected; mid-graph CPU islands would be a fail. `ALL` may still prefer ANE — use `CPU_AND_GPU` for the GPU number.

mp4 timings (S=512, warmup=2, iters=5): CPU_ONLY p50 **65.0 ms** / CPU_AND_NE **27.4 ms** / CPU_AND_GPU **38.6 ms** / ALL **28.4 ms**. ALL plan is 2246 ANE + 78 GPU (begin GPU island), not a GPU path and not full ANE.

## Multimodal 740M → Core AI (primary)

Text-only Core ML stays as the T4–T10 bonus path. New work targets **Core AI** (`torch.export` → `.aimodel`) for the full checkpoint (vision 170M + audio 300M + text 270M). Do **not** call `forge.py convert`. Use forge’s `coreai/.venv` only as the Core AI toolchain (`ANEMLL_COREAI_PYTHON`).

The TB36 `google-embeddinggemma-2` safetensors already include `vision_tower` + `audio_tower`. Load them with empty `config_kwargs` (not T2’s text-only strip).

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_multimodal_media.py
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/gen_multimodal_fixtures.py
ANEMLL_COREAI_PYTHON=/Users/anemll/anemll-forge/coreai/.venv/bin/python \
  /Volumes/Models/anemll-embeddings/.venv/bin/python scripts/export_coreai_vision.py --probe
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_coreai_towers.py
ANEMLL_COREAI_PYTHON=/Users/anemll/anemll-forge/coreai/.venv/bin/python \
  /Volumes/Models/anemll-embeddings/.venv/bin/python scripts/export_coreai_towers.py
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_coreai_smoke.py
ANEMLL_COREAI_PYTHON=/Users/anemll/anemll-forge/coreai/.venv/bin/python \
  /Volumes/Models/anemll-embeddings/.venv/bin/python scripts/smoke_coreai_towers.py
```

Separate packages under `$ANEMLL_EMBEDDINGS_ARTIFACTS/coreai/`: `vision_s280.aimodel` (pixels → 280×512 soft tokens), `audio_s280.aimodel` (280×128 mel → 70×512; chunked attn windows via `index_select` instead of `aten.unfold`), `text_s128.aimodel` (ids/mask → 768). Audio `.pt2` save can hit TreeSpec; convert uses the live exported program. Host interleaves placeholders. Cast16 can clash on int `div`; convert retries without it.

CPU-only smoke on mp4 (`scripts/smoke_coreai_towers.py`): vision / text / audio all **PASS** (finite, expected shapes). Preferred ANE specialize aborted (`mps_spi.sdpa` grouping; `i1` mask `2520xi1`). No mid-graph CPU island from the CPU smoke path. Results: `$ANEMLL_EMBEDDINGS_ARTIFACTS/coreai/towers.smoke.json`.

Synthetic media + multimodal embeddings land under `$ANEMLL_EMBEDDINGS_ARTIFACTS/fixtures/` (not git, not the FLOAT32 text `.mlpackage` tree). Prefixes are text-only; image/video/audio use `<|image|>` / `<|video|>` / `<|audio|>`.

## Local showcase

`demo/` is a small FastAPI server and three static pages: multimodal search, a cross-modal cosine matrix, and “search what I heard”. The pages are plain HTML. There is no build step.

The server loads one backend at startup:

| Backend | What it runs |
| --- | --- |
| `coreai` | `vision_s280`, `audio_s280`, and `text_embeds_s320`, kept resident. Host embed + scatter matches `scripts/parity_coreai_host.py`. |
| `reference` | Full EmbeddingGemma 2 checkpoint on CPU (sentence-transformers), same encode path as `scripts/gen_multimodal_fixtures.py`. |
| `mock` | Deterministic stand-in vectors so the API and pages run on a machine without Core AI. |

**Core AI defaults to the Neural Engine.** Parity does not: `scripts/_coreai_run_npy.py` leaves `ANEMLL_COREAI_COMPUTE` unset as CPU, and `ANEMLL_COREAI_COMPUTE=ane` opts that runner into the Neural Engine. The showcase sets the variable to `ane` only inside its worker. Override the server with `--compute cpu` or `ANEMLL_DEMO_COMPUTE=cpu`.

Audio uses the processor keep-mask. Pad frames stay masked. An all-ones mask lets fp16 pad noise into real frames on the ANE. The export’s additive mask constant is `-1e4` (fp16-safe); the host passes the 0/1 mask, not that constant.

Packages and the index live outside git:

```sh
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_COREAI_PYTHON=/Users/anemll/anemll-forge/coreai/.venv/bin/python
export ANEMLL_DEMO_DATA=$HOME/.anemll-embeddings/demo
export ANEMLL_DEMO_CORPUS=$HOME/.anemll-embeddings/corpus
```

On a Mac, from the repo root, with the host venv that already has torch and transformers:

```sh
/Volumes/Models/anemll-embeddings/.venv/bin/python -m pip install -r demo/requirements.txt
/Volumes/Models/anemll-embeddings/.venv/bin/python -m demo.server \
  --backend coreai --host 0.0.0.0 --port 8765
```

Then open `http://127.0.0.1:8765`. Other machines on the LAN use `http://<mac-ip>:8765` (the process binds `0.0.0.0`). `GET /health` reports the backend, the three towers, warmup milliseconds, and placement (`fullyOnANE` when the runtime says so). Each page shows that as `on ANE · N ms`.

This VM has no Neural Engine. Use the mock backend:

```sh
python -m pip install -r demo/requirements.txt
python -m demo.server --backend mock --port 8765
python tests/test_demo_api.py
python tests/test_demo_coreai_masks.py
```

Optional corpus (CC0 / CC BY via Openverse, licenses recorded in `manifest.json`, files not committed):

```sh
python demo/scripts/fetch_corpus.py --dest "$ANEMLL_DEMO_CORPUS"
python demo/scripts/seed_index.py --base-url http://127.0.0.1:8765 --corpus "$ANEMLL_DEMO_CORPUS"
```

| Variable | Role |
| --- | --- |
| `ANEMLL_DEMO_BACKEND` | `mock` (default), `reference`, or `coreai` |
| `ANEMLL_DEMO_HOST` / `ANEMLL_DEMO_PORT` | Bind address, default `0.0.0.0:8765` |
| `ANEMLL_DEMO_DATA` | Index `index.npz` + `index.json` + media |
| `ANEMLL_DEMO_CORPUS` | Fetched demo media |
| `ANEMLL_DEMO_COMPUTE` | `ane` (default) or `cpu` for the coreai backend |
| `ANEMLL_EMBEDDINGS_ARTIFACTS` | Directory whose `coreai/` holds the three `.aimodel` packages |
| `ANEMLL_EMBEDDINGS_MODEL` | Checkpoint for the host tokenizer, processor, and embed lookup |
| `ANEMLL_COREAI_PYTHON` | Interpreter with `coreai`. The parity runner still defaults to CPU |

`POST /embed` accepts JSON `{"text": "..."}` or a multipart image/audio file (wav, and browser webm/ogg via ffmpeg, resampled to 16 kHz mono). `POST /index` stores the vector. `POST /search` returns top-k cosine across modalities. Pages: `/` search, `/heatmap` matrix, `/heard` rolling mic chunks.

## License

Apache License 2.0 — see `LICENSE.note`; EmbeddingGemma upstream terms also apply to model weights.
