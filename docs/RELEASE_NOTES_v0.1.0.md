# anemll-embeddings v0.1.0

Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) as
Apple Core AI packages: one 768-d embedding space for photos, sounds, and text,
computed on the Mac's Neural Engine.

## Platform support

| Mac | macOS | Placement |
| --- | --- | --- |
| M4 Pro, M3 Ultra | 27.0 | All three towers fully on the Neural Engine |
| M5 | 27.2 | Audio on the Neural Engine; vision and text fall back to the GPU (the Core AI ANE pre-check rejects them). Embeddings still match the reference (cosine 0.99994–0.99997). |

Check your Mac with `python scripts/warmup.py --require-ane` (exits 3 unless
every tower is fully on the Neural Engine).

## Install

Host venv: Python 3.11 or newer (3.12 tested). Core AI runtime: a separate
Python 3.13 venv with `coreai-core` 1.0.0b2.

```sh
git clone --branch v0.1.0 https://github.com/Anemll/anemll-embeddings && cd anemll-embeddings
python -m pip install -e ".[runtime]" -c constraints.txt
python3.13 -m venv ~/.anemll-embeddings/coreai-venv
~/.anemll-embeddings/coreai-venv/bin/python -m pip install "coreai-core==1.0.0b2" numpy
python scripts/download_models.py
python scripts/warmup.py --require-ane
```

The first warmup compiles each tower for your chip and caches it (~111 s in
total cold on an M3 Ultra; ~0.1 s per tower warm). Full walkthrough:
[README](https://github.com/Anemll/anemll-embeddings/blob/v0.1.0/README.md).

## Packages

Hugging Face:
[`anemll/anemll-embeddinggemma-2-ane`](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane/tree/90d2ab497d423bba4ee29947b274c787bb4a1f0a)
@ `90d2ab497d423bba4ee29947b274c787bb4a1f0a`: `vision_s280`, `audio_s280`,
`text_embeds_s320`, and `host/` (about 1.49 GB). `download_models.py` checks
every tower and `host/` file against SHA-256 digests tracked in this repo.

## Highlights

- Reproducible install: tested dependency ranges, exact `constraints.txt`, and
  a documented Core AI runtime venv with a clear error when it is missing.
- Placement stated per platform; `warmup.py --require-ane` for a strict check.
- Demo binds to `127.0.0.1` by default (LAN only via explicit
  `--host 0.0.0.0`, with a warning), XSS fixes, request and media limits.
- Downloads verified by SHA-256; no `trust_remote_code`.
- CI on Python 3.11 and 3.12 (unit suite, ruff, wheel contents).

Full list: [CHANGELOG.md](https://github.com/Anemll/anemll-embeddings/blob/v0.1.0/CHANGELOG.md).

## Verify this release

`MANIFEST-v0.1.0.json` (attached) lists the SHA-256 of every Hugging Face
package file at the pin and of the key repo files. It is signed keyless with
Sigstore by this repo's release workflow:

```sh
cosign verify-blob MANIFEST-v0.1.0.json \
  --bundle MANIFEST-v0.1.0.json.sigstore.json \
  --certificate-identity https://github.com/Anemll/anemll-embeddings/.github/workflows/release.yml@refs/tags/v0.1.0 \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
python scripts/release_manifest.py verify --version v0.1.0
```

Details: [release/README.md](https://github.com/Anemll/anemll-embeddings/blob/v0.1.0/release/README.md).

## Known limitations

- M5 / macOS 27.2: vision and text run on the GPU (see above).
- No video package yet.
- The demo has no authentication; keep it on `127.0.0.1` or a trusted LAN.

## License

Code: MIT. Model weights: converted from `google/embeddinggemma-2`, Apache-2.0
under Google's terms, with the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
