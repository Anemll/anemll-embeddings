# Release manifests

`MANIFEST-<version>.json` lists the SHA-256 and size of every release artifact:

- every file of the Hugging Face package
  [`anemll/anemll-embeddinggemma-2-ane`](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane)
  at the pinned revision (`ANE_REVISION` in `scripts/download_common.py`):
  the three towers, `host/`, the model card, and the license files
- the key files of this repo: `api/`, the download / warmup / manifest scripts
  in `scripts/`, `hf/towers.yaml`, `pyproject.toml`, `constraints.txt`, and
  `LICENSE`

It is built by `scripts/release_manifest.py` and committed before the tag.

## How it is signed

There is no long-lived signing key. When a `v*` tag is pushed,
[`.github/workflows/release.yml`](../.github/workflows/release.yml) regenerates
the manifest from the tagged checkout and the Hugging Face revision, fails if it
differs from the committed copy, and signs it **keyless with Sigstore**
(`cosign sign-blob`). Sigstore's Fulcio CA issues a short-lived certificate to
that workflow run through GitHub OIDC, and the signature is logged in the
public Rekor transparency log. The workflow attaches these files to the GitHub
release:

| File | What |
| --- | --- |
| `MANIFEST-<tag>.json` | the manifest (same bytes as the committed `release/MANIFEST-<tag>.json`) |
| `MANIFEST-<tag>.json.sigstore.json` | Sigstore bundle: certificate, signature, transparency-log proof |
| `SHA256SUMS-<tag>.txt` | the same digests as a `shasum` listing (`hf/…` and `repo/…` paths) |
| `SHA256SUMS-<tag>.txt.sigstore.json` | its Sigstore bundle |

## How to verify

Needs [cosign](https://docs.sigstore.dev/cosign/system_config/installation/) 3.1.3
or newer (`brew install cosign`).

```sh
TAG=v0.1.0
gh release download "$TAG" -R Anemll/anemll-embeddings -p 'MANIFEST-*'

# 1. The manifest was signed by this repo's release workflow at this tag.
cosign verify-blob "MANIFEST-$TAG.json" \
  --bundle "MANIFEST-$TAG.json.sigstore.json" \
  --certificate-identity "https://github.com/Anemll/anemll-embeddings/.github/workflows/release.yml@refs/tags/$TAG" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
# prints: Verified OK

# 2. It is the manifest committed at the tag.
git checkout "$TAG"
cmp "MANIFEST-$TAG.json" "release/MANIFEST-$TAG.json"

# 3. Your checkout and your download match it.
python scripts/download_models.py          # downloads and checks the pinned package
python scripts/release_manifest.py verify --version "$TAG"   # --dest if you used one
```

`verify` hashes the repo files listed in the manifest and every package file
found under `~/.anemll-embeddings/ane` (or `<dest>/ane`), and exits non-zero on
any mismatch. To pin the exact commit too, add
`--certificate-github-workflow-sha <commit>` to `cosign verify-blob` (the tag's
commit, `git rev-parse "$TAG^{commit}"`).
