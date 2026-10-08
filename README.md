# anemll-embeddings

This project turns Google’s EmbeddingGemma 2 into Apple Core AI packages that run on your Mac’s Neural Engine. You can search photos, sounds, and text together. Everything stays on the Mac — there is no cloud API in the loop.

The Neural Engine is the dedicated chip on Apple Silicon for this kind of work. These packages are built so the hot path stays on that chip: no mid-graph hop to the GPU or CPU.

## What is an embedding?

An embedding is a list of numbers that captures meaning. This model writes a list of 768 numbers for each photo, sound, or sentence. Things that mean the same thing land close together, even across types: a photo of a fox, a bark, and the words “a red fox” can match each other. Search then ranks by how close those lists are (cosine similarity on length-normalized vectors).

![How EmbeddingGemma 2 is used: one vector per input, then compare](docs/images/embedding_flow.png)

The model turns each input into one 768-number vector; similarity is computed afterwards by comparing vectors (grid values are illustrative).

One input (for example the phrase “white house”) always gives exactly one vector, regardless of length.

## What’s included

- Scripts that export EmbeddingGemma 2’s vision, audio, and text towers to Core AI `.aimodel` packages
- Host code that tokenizes text, runs those packages, and inserts image/audio tokens into the text model
- A local demo: search, a similarity heatmap, “search what I heard”, and camera alerts (see [Try the demo](#try-the-demo) and [demo/README.md](demo/README.md))
- Small test fixtures (prompts and reference vectors)
- An older text-only Core ML path (see [below](#older-text-only-core-ml-path))

## Folder map

| Path | What lives there |
| --- | --- |
| `model/` | Export / convert EmbeddingGemma 2 to Core AI / ANE: wrappers, ANE graph patches, specialize and inspect tools, parity and cosine checks |
| `api/` | Importable Python runtime: load the `.aimodel` towers and embed text / image / audio (`from api import Embedder`) |
| `samples/` | Small runnable examples plus corpus / alert fetch scripts and manifests (no large binaries in git) |
| `demo/` | FastAPI showcase server and static pages only (imports `api`) |
| `docs/` | How it works, historical plan, diagrams |
| `tests/` | Unit and API tests |

**Model weights are not in this repo.** You download EmbeddingGemma 2 yourself and check the license and terms on the model card.

## Requirements

- An Apple Silicon Mac. The numbers below were measured on an **M4 Pro, macOS 27.0**.
- Python with `torch`, `transformers`, `sentence-transformers`, and `pillow`
- A Python that can `import coreai.runtime`, pointed at by `ANEMLL_COREAI_PYTHON` (typically the `coreai/.venv` from [Anemll/anemll-forge](https://github.com/Anemll/anemll-forge))
- The official checkpoint: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (~740M, vision + audio + text)

## Quick start

Set these before any download or export. If you skip them, the scripts fall back to paths on one developer machine and will not find your files.

```sh
export ANEMLL_EMBEDDINGS_MODEL=/path/to/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/path/to/artifacts
export ANEMLL_COREAI_PYTHON=/path/to/anemll-forge/coreai/.venv/bin/python
export HF_HOME=$HOME/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface
```

1. **Install** (in a virtualenv):

   ```sh
   python -m pip install -e ".[demo]"
   python -m pip install torch transformers sentence-transformers pillow
   ```

2. **Get the weights** (not committed here; read the Hugging Face terms first):

   ```sh
   huggingface-cli download google/embeddinggemma-2 --local-dir "$ANEMLL_EMBEDDINGS_MODEL"
   ```

3. **Export the packages**, or skip this if you already have them under `$ANEMLL_EMBEDDINGS_ARTIFACTS/coreai/`:

   ```sh
   python model/export_coreai_towers.py \
     --tower vision --tower audio --tower text --tower text_embeds
   ```

   With no `--tower` flags the script exports vision, audio, and the ids-only text package. Add `--tower text_embeds` if you want captions or mixed image/audio. That writes:

   - `vision_s280.aimodel` — pixels → 280 × 512 image tokens
   - `audio_s280.aimodel` — 280 × 128 mel frames → 70 × 512 audio tokens
   - `text_s128.aimodel` — token ids → 768-d vector (plain text)
   - `text_embeds_s320.aimodel` — already-looked-up tokens, including image/audio, → 768-d vector

   This is not `forge.py convert`. The Core AI Python is used only as the converter.

4. **Run a first embedding** (load each package and do one forward on the Neural Engine):

   ```sh
   python model/smoke_coreai_towers.py --compute ane \
     --tower vision --tower audio --tower text --tower text_embeds
   ```

   Or compare a few fixture cases against the original PyTorch model:

   ```sh
   ANEMLL_COREAI_COMPUTE=ane python model/parity_coreai_host.py
   ```

## Python usage

The public runtime loads the three `.aimodel` towers and returns one 768-d unit vector. Set `ANEMLL_EMBEDDINGS_MODEL` and `ANEMLL_EMBEDDINGS_ARTIFACTS` first (and `ANEMLL_COREAI_PYTHON` on a Mac).

```python
from api import Embedder

embedder = Embedder(compute="ane")  # or Embedder(backend="mock") without Core AI
text = embedder.embed_text("a red fox")          # role="query" (SearchQuery)
image = embedder.embed_image(pil_image)
audio = embedder.embed_audio(wav_f32, 16000)
embedder.close()
```

`python -m pip install -e .` makes `from api import Embedder` work from any working directory. From a repo checkout, keep the repo root on `PYTHONPATH` (the samples do this).

`SearchQuery` / `Document` are task prefixes the original model expects. `embed_text(..., role="query")` and `role="SearchQuery"` are the same; `role="document"` matches `Document`.

The host path in `api/coreai_host.py` runs `vision_s280` / `audio_s280`, scatters those tokens into the text sequence, then runs `text_embeds_s320`. Plain text can use `text_s128` (ids only). `model/parity_coreai_host.py` is the full loop. The older Sentence-Transformers wrapper still lives at `model/embed_wrapper.py` for export and fixture work.

Examples: `python samples/embed_sentence.py --backend mock`, `python samples/image_text_search.py --backend mock`, `python samples/sound_matching.py --backend mock`, `python samples/camera_alert_rule.py --backend mock`.

## Results

Measured on an **M4 Pro, macOS 27.0**. Cosine is the Neural Engine package versus the patched PyTorch model in FP32 on CPU. Inputs: the aurora fixture image, the `tone_a4` fixture clip, and a caption plus image tokens (269 tokens).

| Package | Placement | Cosine vs FP32 CPU | p50 |
| --- | --- | --- | --- |
| `vision_s280` | fully ANE, 1 region | 0.999954 (row min 0.99979) | 337 ms |
| `audio_s280` | fully ANE, 1 region | 0.999927 (25 valid rows, row min 0.99944) | 10.8 ms |
| `text_embeds_s320` | fully ANE, 1 region | 0.999963 | 34.8 ms |

Each tower is one ANE region: `mps.fullyPlacedOnANE` and `mps.noGPUActivity`, with no GPU or CPU regions and no ANE validation messages.

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
- Weights are not in git. Check the [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) terms before you download.
- Default script paths (`ANEMLL_EMBEDDINGS_MODEL`, artifacts, Hugging Face cache, `ANEMLL_COREAI_PYTHON`) point at one developer machine. Set the environment variables above.
- The demo’s microphone needs `http://127.0.0.1` or HTTPS. A plain `http://<lan-ip>` page cannot record.

## Try the demo

This demo shows EmbeddingGemma 2 running on your Mac’s Neural Engine. It turns photos, sounds, and text into embeddings — lists of numbers that capture meaning — so things that mean the same thing land close together, even across types. Everything stays on your Mac.

```sh
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

Apache License 2.0 — see `LICENSE.note`. EmbeddingGemma upstream terms also apply to the model weights.
