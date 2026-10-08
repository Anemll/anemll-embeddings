# Changelog

All notable changes to this project are documented here.

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
