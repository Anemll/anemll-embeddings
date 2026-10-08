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
| `vision_s280/` | `vision_s280.aimodel` | 2520 image patches → 280 soft tokens | ~337 ms |
| `audio_s280/` | `audio_s280.aimodel` | 280 mel frames → 70 soft tokens | ~11–26 ms |
| `text_embeds_s320/` | `text_embeds_s320.aimodel` | 320 token embeddings → 768-d embedding | ~35 ms |
| `host/` | tokenizer, processor, `embed_tokens.safetensors` | host-side lookup for `api.Embedder` | — |

Exact input/output names, shapes, dtypes, and checksums are in
[`towers.yaml`](towers.yaml). Per-file origin for `host/` is in
[`host/SOURCE.md`](host/SOURCE.md).

Images and audio go through their tower first. Their soft tokens are then
placed into the token sequence and run through `text_embeds_s320`, which
produces the final embedding. Text goes straight to `text_embeds_s320`.
End to end that is about 370 ms per image, 50–60 ms per audio clip, and
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
- fp16-safe RMSNorm
- audio relative-position keys precomputed at export
- vision and audio feed the text backbone as soft tokens interleaved with the text tokens (the host builds `inputs_embeds`)
- fixed sequence lengths: 2520 patches / 280 frames / 320 tokens

## Validation (M4 Pro, macOS 27.0)

- All three towers run **fully on the ANE** (no GPU or CPU regions).
- Cosine of each tower's ANE output vs its FP32 CPU reference:
  vision **0.999954**, audio **0.999927**, text_embeds **0.999963**.

## Limitations

- Validated only on **M4 Pro with macOS 27.0**.
- On **M5 with macOS 27.2**, the ANE pre-check rejects the vision and text
  packages ("Parsing failed, invalid MLIR-MPS program"), and Core AI falls
  back to the GPU.
- Video is not converted.

## Usage

The runtime lives in [Anemll/anemll-embeddings](https://github.com/Anemll/anemll-embeddings).
One download from this repo is enough for inference (towers + `host/`).

```bash
git clone https://github.com/Anemll/anemll-embeddings && cd anemll-embeddings
python scripts/download_models.py
# prints the export lines; add them to ~/.zshrc or source them
python scripts/warmup.py
```

`download_models.py` prefers `host/` here. If a pin does not have that
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
