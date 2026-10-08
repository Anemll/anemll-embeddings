# Security

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/Anemll/anemll-embeddings/security/advisories/new)
rather than a public issue. Include the affected file or endpoint, steps to
reproduce, and the impact you see. If private reporting is not available, open
an issue that asks for a security contact without including details.

## Scope and known limits

- **The demo server has no authentication.** It binds to `127.0.0.1` by
  default. `--host 0.0.0.0` / `ANEMLL_DEMO_HOST=0.0.0.0` exposes add, delete,
  upload, and alert-rule endpoints to everyone who can reach the port; use it
  only on a network you trust, never on the internet. Request bodies are capped
  (`ANEMLL_DEMO_MAX_UPLOAD_MB`, `ANEMLL_DEMO_MAX_JSON_KB`) and decoded media is
  bounded, but the demo is not hardened for hostile clients.
- **Downloads** are pinned to immutable Hugging Face revisions and checked
  against SHA-256 digests tracked in `scripts/download_common.py` before they are
  marked installed. `python scripts/download_models.py --verify` re-checks files
  already on disk.
- **No remote code.** Transformers models and processors load with
  `trust_remote_code=False`.
- The Core AI worker is a local subprocess that exchanges `.npz` / `.npy` files
  in a private temporary directory; each request's files are deleted after use.
