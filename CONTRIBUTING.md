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

Say which chip and macOS version you ran on. `--require-ane` should exit 0 on
M4 Pro / M3 Ultra (macOS 27.0) and M5 (macOS 27.2). If you change attention,
keep the softmax spelled out (`softmax_unfused`): a plain `softmax` between the
attention matmuls puts vision and text on the GPU on macOS 27.2.

## Guidelines

- Keep paths generic: no user names, home directories, or internal volumes in
  code or docs. Use `~/.anemll-embeddings` or environment variables.
- Pin new dependencies to a tested minor range in `pyproject.toml` and record
  the exact version in `constraints.txt`.
- Changes to the published packages (`hf/`, `towers.yaml`, digests in
  `scripts/download_common.py`) need a new Hugging Face revision and a matching
  `ANE_REVISION` bump in the same PR.
- Code is MIT. Model weights stay under Google's terms (Apache-2.0).
