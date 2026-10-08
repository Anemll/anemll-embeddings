# anemll-embeddings

This project turns Google’s EmbeddingGemma 2 into Apple Core AI packages that run on your Mac’s Neural Engine. You can search photos, sounds, and text together. Everything stays on the Mac — there is no cloud API in the loop.

The Neural Engine is the dedicated chip on Apple Silicon for this kind of work. These packages are built so the hot path stays on that chip: no mid-graph hop to the GPU or CPU.

## What is an embedding?

An embedding is a list of numbers that captures meaning. This model writes a list of 768 numbers for each photo, sound, or sentence. Things that mean the same thing land close together, even across types: a photo of a fox, a bark, and the words “a red fox” can match each other. Search then ranks by how close those lists are (cosine similarity on length-normalized vectors).

![How EmbeddingGemma 2 is used: one vector per input, then compare](docs/images/embedding_flow.png)

The model turns each input into one 768-number vector; similarity is computed afterwards by comparing vectors (grid values are illustrative).

One input (for example the phrase “white house”) always gives exactly one vector, regardless of length.

## What’s included

- Public Neural Engine packages on [Hugging Face](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane), plus `scripts/download_models.py` and `scripts/warmup.py`
- Scripts that export EmbeddingGemma 2’s vision, audio, and text towers to Core AI `.aimodel` packages
- Host code that tokenizes text, runs those packages, and inserts image/audio tokens into the text model
- A local demo: search, a similarity heatmap, “search what I heard”, and camera alerts (see [Try the demo](#try-the-demo) and [demo/README.md](demo/README.md))
- Small test fixtures (prompts and reference vectors)
- An older text-only Core ML path (see [below](#older-text-only-core-ml-path))

## Folder map

| Path | What lives there |
| --- | --- |
| `model/` | Export / convert EmbeddingGemma 2 to Core AI / ANE: wrappers, ANE graph patches, specialize and inspect tools, parity and cosine checks |
| `api/` | Importable Python runtime: `from api import Embedder, cosine` — see [api/README.md](api/README.md) |
| `scripts/` | Download the public Hugging Face packages and warm them up on this Mac |
| `samples/` | Small runnable examples plus corpus / alert fetch scripts and manifests (no large binaries in git) |
| `demo/` | FastAPI showcase server and static pages only (imports `api`) |
| `docs/` | How it works, historical plan, diagrams |
| `tests/` | Unit and API tests |

**Model weights are not in this git repo.** The converted Neural Engine packages are on Hugging Face at [anemll/anemll-embeddinggemma-2-ane](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane) (commit `8ceba04`). The host still needs [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (revision `914f7f89142e33e77833254d9c9b90c3cef7303b`) for the tokenizer, processor, and embedding lookup. Both are Apache-2.0 under Google’s terms, not MIT.

## Requirements

- An Apple Silicon Mac. The numbers below were measured on an **M4 Pro, macOS 27.0**.
- Python with `torch`, `transformers`, `sentence-transformers`, and `pillow`
- A Python that can `import coreai.runtime`, pointed at by `ANEMLL_COREAI_PYTHON` (typically the `coreai/.venv` from [Anemll/anemll-forge](https://github.com/Anemll/anemll-forge))
- The official checkpoint: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (~740M, vision + audio + text) and the converted packages at [anemll/anemll-embeddinggemma-2-ane](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane)

## Quick start

Validated on an **M4 Pro, macOS 27.0**. On **macOS 27.2 / M5** the Core AI ANE pre-check currently rejects the vision and text packages (`invalid MLIR-MPS program`) and they fall back to the GPU; audio still runs on the Neural Engine.

You need a Python that can `import coreai.runtime` (typically `coreai/.venv` from [Anemll/anemll-forge](https://github.com/Anemll/anemll-forge)).

1. **Install** (in a virtualenv):

   ```sh
   python -m pip install -e ".[demo]"
   python -m pip install torch transformers sentence-transformers pillow
   export ANEMLL_COREAI_PYTHON=/path/to/anemll-forge/coreai/.venv/bin/python
   ```

2. **Download** the public ANE packages and the Google host checkpoint (idempotent). Layout on disk is `ane/<name>/<name>.aimodel/` plus a host copy of `google/embeddinggemma-2`; the script then creates the `artifacts/coreai/<name>.aimodel` symlinks `api.Embedder` expects:

   ```sh
   python scripts/download_models.py
   # or: ./scripts/download_models.sh
   ```

   Copy the `export` lines it prints (or run them). Default dest is `~/.anemll-embeddings` (`--dest` to override). That fetches [anemll/anemll-embeddinggemma-2-ane](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane) at `8ceba04` (`vision_s280`, `audio_s280`, `text_embeds_s320`) and [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) at `914f7f89142e33e77833254d9c9b90c3cef7303b`.

3. **Warm up** once on this Mac. There is **no** per-hardware compile to ship: Core AI specializes each tower for the local chip on first load and caches it (`~/Library/Caches/coreai-cache`). The first run is slow; later loads are fast. The script reports whether each tower is fully on the ANE, plus load and first-run time:

   ```sh
   python scripts/warmup.py
   ```

4. **Run a sample or the demo:**

   ```sh
   python samples/embed_sentence.py "a red fox"
   python -m demo.server --backend coreai --host 0.0.0.0 --port 8766
   ```

To re-export packages yourself (optional; not needed if you used the download script):

```sh
python model/export_coreai_towers.py \
  --tower vision --tower audio --tower text --tower text_embeds
```

That writes `vision_s280.aimodel` (pixels → 280 × 512 tokens), `audio_s280.aimodel` (280 × 128 mel frames → 70 × 512 tokens), `text_s128.aimodel` (ids-only 768-d; not in the public HF pack), and `text_embeds_s320.aimodel` (looked-up tokens, including image/audio, → 768-d). This is not `forge.py convert`.

## Python usage

Copy-paste examples (text–text, image–text, audio–text, camera-alert threshold) live in **[api/README.md](api/README.md)**. Short version:

```python
from api import Embedder, cosine

embedder = Embedder(compute="ane")  # or Embedder(backend="mock") without Core AI
q = embedder.embed_text("a red fox")                 # shape (768,), L2 == 1
d = embedder.embed_text("a red fox in snow", role="document")
print(cosine(q, d))                                  # float in [-1, 1]
embedder.close()
```

`python -m pip install -e .` makes `from api import Embedder` work from any working directory. From a repo checkout, keep the repo root on `PYTHONPATH` (the samples do this). The demo pages call this same `Embedder`. `Embedder(compute="ane")` reads `ANEMLL_EMBEDDINGS_ARTIFACTS`, `ANEMLL_EMBEDDINGS_MODEL`, and `ANEMLL_COREAI_PYTHON` when those constructor arguments are omitted.

The host path in `api/coreai_host.py` runs `vision_s280` / `audio_s280`, scatters those tokens into the text sequence, then runs `text_embeds_s320`. `model/parity_coreai_host.py` is the full loop. The older Sentence-Transformers wrapper still lives at `model/embed_wrapper.py` for export and fixture work.

Examples: `python samples/embed_sentence.py --backend mock`, `python samples/image_text_search.py --backend mock`, `python samples/sound_matching.py --backend mock`, `python samples/camera_alert_rule.py --backend mock`. The samples also take `--artifacts`, `--model`, and `--coreai-python`.

## Results

Measured on an **M4 Pro, macOS 27.0**. Cosine is the Neural Engine package versus the patched PyTorch model in FP32 on CPU. Inputs: the aurora fixture image, the `tone_a4` fixture clip, and a caption plus image tokens (269 tokens).

| Package | Placement | Cosine vs FP32 CPU | p50 |
| --- | --- | --- | --- |
| `vision_s280` | fully ANE, 1 region | 0.999954 (row min 0.99979) | 337 ms |
| `audio_s280` | fully ANE, 1 region | 0.999927 (25 valid rows, row min 0.99944) | 10.8 ms |
| `text_embeds_s320` | fully ANE, 1 region | 0.999963 | 34.8 ms |

Each tower is one ANE region: `mps.fullyPlacedOnANE` and `mps.noGPUActivity`, with no GPU or CPU regions and no ANE validation messages.

End-to-end `Embedder(compute="ane")` scores on the same M4 Pro (cosine between two embeddings, after warmup):

| Pair | Cosine |
| --- | --- |
| text `a red fox` vs `a red fox in the snow` | 0.901 |
| text `a red fox` vs `a delivery truck` | 0.694 |
| UPS photo vs `a brown UPS delivery truck` | 0.727 |
| UPS photo vs `a cat` | 0.513 |
| bark vs `a dog barking` | 0.721 |
| bark vs `a cat meowing` | 0.661 |

About **35 ms** per sentence, **380 ms** per photo, **50 ms** per sound.

## How it works

Deeper notes: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). In short:

- The model is three towers (vision ~170M, audio ~300M, text ~270M). Each becomes its own package. The host stitches them; the compiled graphs stay simple.
- The Neural Engine wants 16-bit (fp16) math. The original PyTorch model forbids fp16 because it produces NaNs. Export still uses an fp16 graph (`cast16`), and rewrites RMSNorm / LayerNorm as `x / max|x|` before squaring so large activations (around 900) do not overflow.
- Attention and masks were rewritten so every op is legal on the Neural Engine: no fused attention with mismatched key/value shapes, no 1-bit masks, no `-inf` (that is NaN in fp16; we use `-1e4`), no `aten.unfold`, no 64-bit gathers. Those were the old GPU/CPU leftovers. They are gone on the M4 Pro packages above.

The original conversion plan is in [docs/PLAN.md](docs/PLAN.md) (historical).

## Limitations

- Validated on an M4 Pro running macOS 27.0.
- On macOS 27.2 (M5, newer Neural Engine) the Core AI ANE pre-check currently rejects the vision and text packages (`invalid MLIR-MPS program`) and they fall back to the GPU. Audio still runs on the Neural Engine.
- Audio clips must produce at least one mel frame (about 9 ms at 16 kHz). Shorter clips error instead of returning a bad vector.
- There is no video package yet. `<|video|>` fixtures are skipped.
- Weights are not in git. Use `scripts/download_models.py`. Check the [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) terms and the [ANE package card](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane) before you download.
- The demo’s microphone needs `http://127.0.0.1` or HTTPS. A plain `http://<lan-ip>` page cannot record.

## Try the demo

This demo shows EmbeddingGemma 2 running on your Mac’s Neural Engine. It turns photos, sounds, and text into embeddings — lists of numbers that capture meaning — so things that mean the same thing land close together, even across types. Everything stays on your Mac.

```sh
python scripts/download_models.py
python scripts/warmup.py
python -m pip install -r demo/requirements.txt
python -m demo.server --backend coreai --host 0.0.0.0 --port 8766
```

Then open **http://127.0.0.1:8766**. Default port is **8766**. Without `--backend coreai` the server uses `mock` (fake vectors, no Neural Engine).

Load a small public corpus (optional):

```sh
export ANEMLL_DEMO_CORPUS=$HOME/.anemll-embeddings/corpus
python samples/fetch_corpus.py --dest "$ANEMLL_DEMO_CORPUS"
python samples/seed_index.py --base-url http://127.0.0.1:8766 --corpus "$ANEMLL_DEMO_CORPUS"
```

| Page | What it does |
| --- | --- |
| [Search](http://127.0.0.1:8766/) `/` | Drop photos, audio, or text on the left to add them. Type “a red fox” or drop an image or sound on the right to see the closest matches. |
| [Heatmap](http://127.0.0.1:8766/heatmap) `/heatmap` | A grid of every selected item vs every other. Higher / brighter means more alike. |
| [Search what I heard](http://127.0.0.1:8766/heard) `/heard` | Record a few seconds, then type what you heard. It searches **this session’s audio chunks** (not the main library’s photos). |
| [Camera alert](http://127.0.0.1:8766/alert) `/alert` | Click a camera frame or a sound and see which preset alerts fire. |

Camera-alert photos and clips are not in git. Fetch them beside the corpus:

```sh
python samples/fetch_alert.py --dest "${ANEMLL_DEMO_ALERT:-$HOME/.anemll-embeddings/alert}"
```

Full walkthrough, backends, and things to try: **[demo/README.md](demo/README.md)**.

## Older text-only Core ML path

An earlier track exports the ~270M text tower through `torch.jit.trace` and public `coremltools==9.0` (not Core AI). It is still here; the multimodal packages above are the main path.

```sh
python -m pip install -r requirements-conversion.txt
python model/export_torchscript.py --seq-len 512
python model/convert_coreml.py --seq-len 512
python model/parity_cosine.py --seq-len 512
python model/ane_smoke.py --seq-len 512
```

I/O names are fixed: `input_ids` and `attention_mask` in (`[1, S]`, int32), `embedding` out (`[1, 768]`, float32). `CPU_AND_NE` on convert does not prove Neural Engine placement — `ane_smoke.py` reads the compute plan. FP16 Core ML (`--precision FLOAT16`) is a separate artifact tree; on the M4 Pro the FP16 *CPU* path was the weak one (min cosine 0.920 vs the fixtures), not the Neural Engine path.

## License

The software in this repository is [MIT](LICENSE), Copyright (c) 2026 Anemll LLC.

The model weights — EmbeddingGemma 2 and the Core AI conversions at [anemll/anemll-embeddinggemma-2-ane](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane) — are **Apache-2.0 under Google’s terms** for the base model, not MIT. See that card’s `LICENSE` and `NOTICE`.
