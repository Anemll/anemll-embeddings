# Changelog

All notable changes to this project are documented here.

## Unreleased

### Added
- `Embedder.embed_texts(texts, role=..., dim=..., pack=True)`: embed a list
  of texts in order, with optional Matryoshka `dim` (128/256/512/768, also on
  `embed_text`). `samples/grep_embed.py`: semantic grep over the lines of a
  file.
- Prototype short-text towers (not published): fixed-shape buckets
  `text_embeds_s32`/`s64`/`s128`/`s256` and packed towers `text_pack_<N>x<T>`
  (block-diagonal bias, per-text positions, pooling matrix), exported with
  `model/export_text_buckets.py`, alone or as one multi-function
  `text_buckets.aimodel`. The runtime uses them when they sit next to
  `text_embeds_s320.aimodel`; without them nothing changes. See
  `docs/TEXT_BUCKETS.md`. Measured on an M4 Pro (macOS 27.0) and an M5 Max
  (macOS 27.2): every function fully on the Neural Engine, cosine >= 0.99999
  against the shipped `text_embeds_s320`. The combined package no longer
  carries its own `text_embeds_s320` (identical output, same speed, 17.4 MB
  smaller); export with `--buckets 32,64,128,256`.
- `scripts/warmup.py` loads `text_buckets.aimodel` when it is present, lists
  each bucket / pack function with its own placement (read from the compiled
  manifest), and `--require-ane` covers them. `--no-text-buckets` skips them.
- `scripts/download_models.py` knows the optional combined package
  `text_buckets/text_buckets.aimodel` (fetch, SHA-256 check, symlink). It is
  not on the Hub yet: its revision and digests are marked placeholders in
  `scripts/download_common.py`, `hf/config.json` and `hf/towers.yaml`, and the
  download is skipped (`text_buckets=not-published`) until they are filled in.
  `--no-text-buckets` skips it. `hf/config.json` is staged for the next Hub
  revision (`ROOT_CONFIG_STAGED_SHA256`); the published pin is unchanged.
- Root `config.json` on the Hugging Face package (`hf/config.json` here): a
  small JSON descriptor of the Core AI towers (paths, input/output shapes),
  the base model pin, the 768-d embedding, and where the `host/` files live.
  It is not a transformers config. The Hub's default download counter counts
  requests to a root `config.json`, so installs now show up as downloads.
- New `vision_s280` and `text_embeds_s320` towers on Hugging Face at
  `18e1b7e85cdf0c58d924c5d270c7a4be1a40159a`, exported with the unfused
  attention softmax (`main.mlirb` sha256 `5ebb5342...` vision, `8181d927...`
  text). One export for every Mac: fully on the Neural Engine on M4 Pro /
  M3 Ultra (macOS 27.0) and M5 (macOS 27.2). Vision is 22-34% slower on
  macOS 27.0 than the previous export (414-466 ms vs 339-349 ms); text is
  unchanged there. `audio_s280`, `host/` and `config.json` are unchanged.

### Changed
- Export: attention softmax is spelled out (`softmax_unfused` in
  `model/trace_patches.py`: `amax`/`sub`/`exp`/`sum`/`reciprocal`/`mul`) in the
  text and vision towers. A plain `softmax` between the two attention matmuls
  is fused by MPSGraph into `mps_spi.sdpa`, which the macOS 27.2 ANE check
  rejects when Core AI first loads the tower on that Mac (not at export; the
  only console output is `Failed to import MPS module`), so those towers fell
  back to the GPU. Re-exported text and vision
  towers run fully on the ANE on M5 / macOS 27.2 and on M4 Pro / M3 Ultra,
  macOS 27.0. Text latency is unchanged on 27.0; vision is about 22-34% slower
  there (414-466 ms vs 339-349 ms). On the M5 the ANE is slower than the GPU
  for vision (426 vs 99 ms) and text (33 vs 24-27 ms). Added
  `model/parity_text_embeds_ab.py` (text-only package A/B parity) and
  `tests/test_softmax_unfused.py`. See `docs/M5_ANE_SOFTMAX_FIX.md`.
- `ANE_REVISION` pinned to Hugging Face commit
  `18e1b7e85cdf0c58d924c5d270c7a4be1a40159a` with the re-exported `vision_s280`
  (`main.mlirb` sha256 `5ebb5342...`) and `text_embeds_s320` (`8181d927...`);
  `audio_s280` and `host/` byte-identical to `47d05aa`. Earlier in this
  release cycle the pin was `1cbb580` (adds the root `config.json`).
  Existing installs see the new pin and download the two new towers.
- README, HF card, `towers.yaml`, and docs: all three towers fully on the
  Neural Engine on M4 Pro / M3 Ultra (macOS 27.0) and M5 (macOS 27.2) with the
  re-exported vision and text towers, with the measured timings and cosines.
- Em dashes in the Markdown docs replaced with plain dashes.
- `download_models.py` fetches the root `config.json` with the towers and
  verifies it against `ROOT_CONFIG_SHA256` like the other pinned files.
  Existing installs see the new pin and refresh; unchanged files are not
  downloaded again.

### Fixed
- Warm Core AI cache: `scripts/warmup.py` (and the demo's `/health`) reported
  `vision_s280` as `unknown` on a second run, so `--require-ane` exited 3,
  although the cached manifest shows vision fully on the Neural Engine. The
  placement lookup stripped whitespace bytes from the binary `main.hash`;
  vision's hash ends in `0x0c`, so the cache folder name came out one byte
  short. The raw bytes are now used as is. Also on `main`.

## 0.1.0

First public release. Notes: [docs/RELEASE_NOTES_v0.1.0.md](docs/RELEASE_NOTES_v0.1.0.md).

### Security
- Demo binds to `127.0.0.1` by default. Binding all interfaces needs an explicit
  `--host 0.0.0.0` / `ANEMLL_DEMO_HOST=0.0.0.0` and prints a warning.
- Fixed stored XSS in the Heatmap and Heard pages (labels are rendered as text)
  and escaped media URLs; alert rule colors are validated.
- Request size limits (413) for uploads and JSON; decoded image pixel and audio
  duration caps; `ffmpeg` timeout.
- Downloads are verified against SHA-256 digests tracked in the repo before the
  revision marker is written; new `download_models.py --verify`.
- `ANE_REVISION` pinned to Hugging Face commit
  `90d2ab497d423bba4ee29947b274c787bb4a1f0a` (updated model card; towers and
  `host/` byte-identical to `47d05aa`).
- `trust_remote_code` disabled everywhere.

### Added
- Reproducible Core AI runtime setup (`coreai-core==1.0.0b2` venv) and a clear
  `CoreAIPythonNotFound` error with setup steps.
- `warmup.py --require-ane` (exit 3 unless every tower is fully on the ANE) and
  `--coreai-home`.
- `constraints.txt`, `reference` and `test` extras, CI workflow, hardware-gated
  `tests/test_hardware_ane.py`.
- Console scripts: `anemll-embeddings-download`, `-warmup`, `-embed`, `-demo`.
- Signed release manifest: `release/MANIFEST-v0.1.0.json` (SHA-256 of the
  Hugging Face package at the pin and key repo files), regenerated and signed
  keyless with Sigstore by `.github/workflows/release.yml` on the tag;
  `scripts/release_manifest.py verify` checks a checkout and download.
- Sample downloads record license URL, final URL, SHA-256, size, fetch time, and
  modifications, and are size-capped.

### Changed
- Dependencies pinned to tested ranges (torch 2.14, torchvision 0.29,
  transformers 5.19; sentence-transformers 6.1 moved to the `reference` extra).
- Requires Python 3.11 or newer for the host venv (`requires-python >= 3.11`;
  the tests use `tomllib`). CI runs the unit suite and `ruff check` on 3.11
  and 3.12.
- Default `~/.anemll-embeddings` paths work without exporting environment
  variables. Developer-specific paths removed from code and docs.
- `warmup.py --cache-dir` rejects paths Core AI cannot use instead of silently
  ignoring them.
- README states placement per platform: fully on the ANE on M4 Pro / M3 Ultra
  (macOS 27.0); on M5 / macOS 27.2 only audio is on the ANE.

### Fixed
- Core AI worker IPC files are deleted after every request.
- `prepare_hf_host_folder.py` removes its temporary download; `--dest` docs.
- Printed shell exports are shell-quoted.
- Wheel now includes `demo/static`.
