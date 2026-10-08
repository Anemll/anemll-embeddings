<!-- DRAFT. Not published. Tagging v0.1.0 and publishing a GitHub release need Anemll's OK. -->
# anemll-embeddings v0.1.0

Google's EmbeddingGemma 2 as Apple Core AI packages: one 768-d embedding space
for photos, sounds, and text, computed on the Mac.

## Platform support

| Mac | macOS | Placement |
| --- | --- | --- |
| M4 Pro, M3 Ultra | 27.0 | All three towers fully on the Neural Engine |
| M5 | 27.2 | Audio on the Neural Engine; vision and text fall back to the GPU (Core AI ANE pre-check rejects them). Embeddings still match the reference (cosine 0.99994–0.99997). |

Check yours with `python scripts/warmup.py --require-ane`.

## Install

```sh
python -m pip install -e ".[runtime]" -c constraints.txt
python3.13 -m venv ~/.anemll-embeddings/coreai-venv
~/.anemll-embeddings/coreai-venv/bin/python -m pip install "coreai-core==1.0.0b2" numpy
python scripts/download_models.py
python scripts/warmup.py --require-ane
```

## Packages

Hugging Face: `anemll/anemll-embeddinggemma-2-ane` @ `<ANE_REVISION>` (towers +
`host/`), verified by SHA-256 on download.

## Highlights

See [CHANGELOG.md](../CHANGELOG.md).
