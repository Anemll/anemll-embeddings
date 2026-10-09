---
license: apache-2.0
base_model: google/embeddinggemma-2
base_model_relation: quantized
library_name: coreai
pipeline_tag: feature-extraction
tags:
- coreai
- apple-neural-engine
- ane
- embeddings
- multimodal
- embeddinggemma
- anemll
language:
- multilingual
---

# EmbeddingGemma 2 for the Apple Neural Engine (Core AI)

**Unofficial** Core AI conversion of Google DeepMind's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) by
[Anemll](https://github.com/Anemll). All credit for the model goes to Google
DeepMind. This repo only changes the format so it runs on the Apple Neural
Engine (ANE). It is not affiliated with or endorsed by Google.

EmbeddingGemma 2 turns text, images, and audio into one 768-d vector in a
shared space, so you can compare any of them with cosine similarity.

## What's here

| Folder | Package | Input → output | p50 on M4 Pro (ANE) |
|---|---|---|---|
| `vision_s280/` | `vision_s280.aimodel` | 2520 image patches → 280 soft tokens | ~414 ms |
| `audio_s280/` | `audio_s280.aimodel` | 280 mel frames → 70 soft tokens | ~11-26 ms |
| `text_embeds_s320/` | `text_embeds_s320.aimodel` | 320 token embeddings → 768-d embedding | ~35 ms |
| `host/` | tokenizer, processor, `embed_tokens.safetensors` | host-side lookup for `api.Embedder` | - |

An optional multi-function package `text_buckets/text_buckets.aimodel` (short-text buckets and packed batch towers) is **staged but not uploaded**: `config.json` and `towers.yaml` describe it with placeholder digests, and nothing downloads it until it is published and pinned.

Exact input/output names, shapes, dtypes, and checksums are in
[`towers.yaml`](towers.yaml). The root [`config.json`](config.json) is a short JSON
descriptor of the same package (not a transformers config). Per-file origin for `host/` is in
[`host/SOURCE.md`](host/SOURCE.md).

Images and audio go through their tower first. Their soft tokens are then
placed into the token sequence and run through `text_embeds_s320`, which
produces the final embedding. Text goes straight to `text_embeds_s320`.
End to end that is about 450 ms per image, 45-60 ms per audio clip, and
35 ms per sentence on an M4 Pro.

`host/` is the slim Google host side (~305 MB): tokenizer, processor /
preprocessor configs, `config.json`, and an extracted 256 MiB embed table.
It does **not** include the full 1.49 GB `model.safetensors`. Files there
are unmodified copies of
[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
at `914f7f89142e33e77833254d9c9b90c3cef7303b`, except the embedding table,
which is extracted.

## What changed from the original

Converted from `google/embeddinggemma-2` at revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`. No retraining or fine-tuning.

- float16 weights and activations (the original is float32)
- explicit attention (matmul + softmax) instead of fused SDPA, tiled so it stays on the ANE
- softmax spelled out as max/sub/exp/sum/reciprocal/mul, so macOS 27.2 does not re-fuse attention into an SDPA op the ANE rejects (that check runs when Core AI first loads a tower on the Mac, not at export, and prints only `Failed to import MPS module` before falling back to the GPU)
- fp16-safe RMSNorm
- audio relative-position keys precomputed at export
- vision and audio feed the text backbone as soft tokens interleaved with the text tokens (the host builds `inputs_embeds`)
- fixed sequence lengths: 2520 patches / 280 frames / 320 tokens

## Validation (M4 Pro and M3 Ultra, macOS 27.0; M5, macOS 27.2)

- All three towers run **fully on the ANE** (no GPU or CPU regions) on
  M4 Pro and M3 Ultra with macOS 27.0 and on M5 with macOS 27.2.
- Cosine of the final embedding vs the same host path with the patched
  towers in FP32 on CPU (minimum, M4 Pro): photos **0.999918**, sounds
  **0.998951**, text **0.999934**.
- Warm p50 on the ANE: M4 Pro vision 414 ms, audio 11.2 ms, text 34.8 ms;
  M3 Ultra 466 / 11.6 / 35.2 ms; M5 426 / 13.8 / 33 ms. On the M5 the GPU
  is faster for vision (about 99 ms) and text (about 24-27 ms).

## Limitations

- Fully-ANE placement is validated on **M4 Pro and M3 Ultra with macOS 27.0**
  and **M5 with macOS 27.2**.
- The vision tower is about 22-34% slower on macOS 27.0 than the earlier
  export (414-466 ms instead of 339-349 ms), because its softmax is spelled
  out so that it stays on the ANE on macOS 27.2.
- Video is not converted.

## Usage

The runtime lives in [Anemll/anemll-embeddings](https://github.com/Anemll/anemll-embeddings).
One download from this repo is enough for inference (towers + `host/`).

```bash
git clone https://github.com/Anemll/anemll-embeddings && cd anemll-embeddings
python -m pip install -e ".[runtime]" -c constraints.txt   # host venv, Python 3.11+

# Core AI runtime (separate interpreter that runs the .aimodel packages)
python3.13 -m venv ~/.anemll-embeddings/coreai-venv
~/.anemll-embeddings/coreai-venv/bin/python -m pip install "coreai-core==1.0.0b2" numpy

python scripts/download_models.py   # verifies SHA-256 of towers + host/
python scripts/warmup.py --require-ane
```

With the default `~/.anemll-embeddings` layout no exports are needed. With a
custom `--dest`, add the printed export lines to `~/.zshrc` or source them.
`--require-ane` exits non-zero on Macs where a tower is not fully on the
Neural Engine. It exits 0 on M4 Pro / M3 Ultra (macOS 27.0) and M5
(macOS 27.2).

`vision_s280` and `text_embeds_s320` were re-exported with an unfused
attention softmax at this revision. `audio_s280` and the `host/` files are
byte-identical to commit `47d05aa218a227e887858fe571f8deb2f2a1d532`. The GitHub repo pins an exact revision of this repo (`ANE_REVISION` in
`scripts/download_common.py`) and checks every tower and `host/` file against
SHA-256 digests on download. `download_models.py` prefers `host/` here. If a pin does not have that
folder yet, it falls back to the slim files on
`google/embeddinggemma-2` (still not the full `model.safetensors`).

```python
from PIL import Image
from api import Embedder, cosine

embedder = Embedder(compute="ane")
photo = embedder.embed_image(Image.open("truck.jpg"))
text = embedder.embed_text("a brown UPS delivery truck", role="document")
print(photo.shape, cosine(photo, text))   # (768,) ~0.73 for a UPS truck photo
embedder.close()
```

`embed_audio(wav, 16000)` works the same way for mono 16 kHz audio. See the
repo's `api/README.md`, `samples/`, and the browser demo in `demo/`.

To re-export the packages yourself, use `scripts/download_export_assets.py`
(full Google checkpoint) and `model/export_coreai_towers.py`.

## License

The model weights in this repo are a converted form of
`google/embeddinggemma-2` and are distributed under the same license as the
base model: **Apache License 2.0**, as published by Google for Gemma at
<https://ai.google.dev/gemma/docs/gemma_4_license>. The full text is in
[`LICENSE`](LICENSE), and attribution is in [`NOTICE`](NOTICE). Apache-2.0
allows this redistribution. `host/` files are unmodified copies of the
upstream objects, except the embedding table, which is extracted (see
[`host/SOURCE.md`](host/SOURCE.md)). As the base model card states, use
must also follow the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

The conversion and runtime code at
[github.com/Anemll/anemll-embeddings](https://github.com/Anemll/anemll-embeddings)
is separate software under the MIT License.
