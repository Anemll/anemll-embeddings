# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Setup

```sh
python3.12 -m venv .venv && source .venv/bin/activate   # 3.11 or newer
python -m pip install -e ".[test]" -c constraints.txt
```

For real (non-mock) embeddings also set up the Core AI interpreter described in
[README → Core AI runtime](README.md#core-ai-runtime) and run
`python scripts/download_models.py`.

## Before you open a PR

```sh
python -m compileall -q api model scripts samples demo tests
python -m pip install ruff==0.16.10 && ruff check .
python -m pytest tests -q
```

The unit suite needs no Neural Engine: it uses the mock backend and stubbed
workers. CI (`.github/workflows/ci.yml`) runs the same commands on Python 3.11 and 3.12
plus a dependency check and a wheel-content check.

If you touch the Core AI path, also run the hardware checks on a Mac and paste
the output in the PR:

```sh
python scripts/warmup.py --require-ane        # exit 3 = a tower is not fully on the ANE
ANEMLL_HW_TESTS=1 python -m pytest tests/test_hardware_ane.py -q -s
```

Say which chip and macOS version you ran on. On M5 / macOS 27.2 all three
towers work and match the reference, but vision and text currently run on the
GPU (the macOS 27.2 ANE pre-check rejects them), so `--require-ane` exits 3
there. That is expected, not a regression.

## Guidelines

- Keep paths generic: no user names, home directories, or internal volumes in
  code or docs. Use `~/.anemll-embeddings` or environment variables.
- Pin new dependencies to a tested minor range in `pyproject.toml` and record
  the exact version in `constraints.txt`.
- Changes to the published packages (`hf/`, `towers.yaml`, digests in
  `scripts/download_common.py`) need a new Hugging Face revision and a matching
  `ANE_REVISION` bump in the same PR.
- Code is MIT. Model weights stay under Google's terms (Apache-2.0).
