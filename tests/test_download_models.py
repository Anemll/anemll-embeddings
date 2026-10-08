#!/usr/bin/env python3
"""Unit checks for inference / export download scripts (no network, no Core AI)."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.host_embed import EMBED_SCALE as HOST_EMBED_SCALE  # noqa: E402
from api.host_embed import SLIM_EMBED_NAME as HOST_SLIM_NAME  # noqa: E402
from api.host_embed import load_slim_host  # noqa: E402
from scripts.download_common import (  # noqa: E402
    ANE_ALLOW,
    ANE_BYTES,
    ANE_INFERENCE_ALLOW,
    ANE_REVISION,
    EMBED_SCALE,
    EMBED_TABLE_BYTES,
    EMBED_TENSOR_KEY,
    EXPORT_HOST_ALLOW,
    FULL_SAFE_TENSORS_BYTES,
    HOST_FOLDER,
    HOST_FOLDER_ALLOW,
    INFERENCE_HOST_ALLOW,
    INFERENCE_HOST_IGNORE,
    SLIM_EMBED_NAME,
    TOWERS,
    all_bundles_complete,
    bundle_complete,
    ensure_slim_embed,
    ensure_symlink,
    env_exports,
    export_download_bytes,
    extract_embed_from_bytes,
    extract_embed_from_file,
    inference_download_bytes,
    link_coreai,
    matches_hf_patterns,
    revision_matches,
    snapshot,
    write_revision,
    write_safetensors,
)
from scripts.download_export_assets import download_full_host  # noqa: E402
from scripts.download_models import download_host  # noqa: E402
from scripts.prepare_hf_host_folder import (  # noqa: E402
    ORIGIN_COPIED,
    ORIGIN_EXTRACTED,
    stage_host,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _write_bundle(ane: Path, name: str) -> None:
    root = ane / name / f"{name}.aimodel"
    root.mkdir(parents=True, exist_ok=True)
    for filename in ("metadata.json", "main.hash", "main.mlirb"):
        (root / filename).write_text("ok\n", encoding="utf-8")


def test_bundle_complete(tmp: Path) -> None:
    ane = tmp / "ane"
    if all_bundles_complete(ane):
        _fail("empty dir should be incomplete")
    _write_bundle(ane, "vision_s280")
    if not bundle_complete(ane, "vision_s280"):
        _fail("vision bundle should be complete")
    if all_bundles_complete(ane):
        _fail("two towers still missing")
    for name in TOWERS:
        _write_bundle(ane, name)
    if not all_bundles_complete(ane):
        _fail("all three bundles should be complete")


def test_symlink_idempotent(tmp: Path) -> None:
    ane = tmp / "ane"
    artifacts = tmp / "artifacts"
    for name in TOWERS:
        _write_bundle(ane, name)
    first = link_coreai(artifacts, ane)
    if set(first) != set(TOWERS) or any(state != "created" for state in first.values()):
        _fail(f"first link {first}")
    again = link_coreai(artifacts, ane)
    if any(state != "ok" for state in again.values()):
        _fail(f"second link {again}")
    other = tmp / "other"
    _write_bundle(other, "vision_s280")
    link = artifacts / "coreai" / "vision_s280.aimodel"
    if ensure_symlink(link, other / "vision_s280" / "vision_s280.aimodel") != "replaced":
        _fail("expected replaced")
    if link.resolve() != (other / "vision_s280" / "vision_s280.aimodel").resolve():
        _fail("symlink did not retarget")


def test_revision_marker(tmp: Path) -> None:
    folder = tmp / "ane"
    if revision_matches(folder, ANE_REVISION):
        _fail("missing marker should not match")
    write_revision(folder, ANE_REVISION)
    if not revision_matches(folder, ANE_REVISION):
        _fail("marker should match")
    if revision_matches(folder, "deadbeef"):
        _fail("wrong revision should not match")


def test_env_exports() -> None:
    text = env_exports(
        artifacts=Path("/tmp/artifacts"),
        model=Path("/tmp/model"),
        coreai_python="/tmp/coreai/bin/python",
    )
    want = (
        "export ANEMLL_EMBEDDINGS_ARTIFACTS=/tmp/artifacts\n"
        "export ANEMLL_EMBEDDINGS_MODEL=/tmp/model\n"
        "export ANEMLL_COREAI_PYTHON=/tmp/coreai/bin/python"
    )
    if text != want:
        _fail(f"exports {text!r}")


def test_existing_dir_is_not_replaced(tmp: Path) -> None:
    target = tmp / "real.aimodel"
    target.mkdir(parents=True)
    (target / "main.hash").write_text("x\n", encoding="utf-8")
    link = tmp / "coreai" / "vision_s280.aimodel"
    link.parent.mkdir()
    link.mkdir()
    try:
        ensure_symlink(link, target)
    except FileExistsError:
        return
    _fail("expected FileExistsError for a real directory")


def test_inference_allow_excludes_full_weights() -> None:
    if "model.safetensors" in INFERENCE_HOST_ALLOW:
        _fail("inference allow-list must not include model.safetensors")
    if any(name.endswith(".safetensors") for name in INFERENCE_HOST_ALLOW):
        _fail(f"inference allow-list has a safetensors file: {INFERENCE_HOST_ALLOW}")
    if not matches_hf_patterns("config.json", INFERENCE_HOST_ALLOW, INFERENCE_HOST_IGNORE):
        _fail("config.json should be allowed")
    if not matches_hf_patterns("tokenizer.json", INFERENCE_HOST_ALLOW, INFERENCE_HOST_IGNORE):
        _fail("tokenizer.json should be allowed")
    if not matches_hf_patterns("preprocessor_config.json", INFERENCE_HOST_ALLOW, INFERENCE_HOST_IGNORE):
        _fail("preprocessor_config.json should be allowed")
    if matches_hf_patterns("model.safetensors", INFERENCE_HOST_ALLOW, INFERENCE_HOST_IGNORE):
        _fail("model.safetensors must be excluded by inference allow/ignore")
    if matches_hf_patterns("model.safetensors", INFERENCE_HOST_ALLOW, None):
        _fail("model.safetensors must not match the allow-list even without ignore")
    for name in TOWERS:
        path = f"{name}/{name}.aimodel/main.mlirb"
        if not matches_hf_patterns(path, ANE_ALLOW, None):
            _fail(f"ANE allow missed {path}")
    if matches_hf_patterns("README.md", ANE_ALLOW, None):
        _fail("ANE allow should not pull the repo README")
    if matches_hf_patterns("host/config.json", ANE_ALLOW, None):
        _fail("tower allow should not match host/")
    if not matches_hf_patterns("host/config.json", ANE_INFERENCE_ALLOW, None):
        _fail("inference allow should include host/")
    if not matches_hf_patterns("host/embed_tokens.safetensors", HOST_FOLDER_ALLOW, None):
        _fail("host allow should include the slim embed table")
    if matches_hf_patterns("model.safetensors", HOST_FOLDER_ALLOW, None):
        _fail("host allow should not match a root model.safetensors")
    if ANE_REVISION != "90d2ab497d423bba4ee29947b274c787bb4a1f0a":
        _fail(f"ANE_REVISION {ANE_REVISION} is not the uploaded host/ pin")
    if SLIM_EMBED_NAME != HOST_SLIM_NAME:
        _fail("slim filename drifted between host and download")
    if abs(EMBED_SCALE - HOST_EMBED_SCALE) > 1e-9:
        _fail("embed scale drifted between host and download")


def test_export_allow_is_full_checkpoint() -> None:
    if EXPORT_HOST_ALLOW is not None:
        _fail(f"export allow should be None (all files), got {EXPORT_HOST_ALLOW}")
    if not matches_hf_patterns("model.safetensors", EXPORT_HOST_ALLOW, None):
        _fail("export must download model.safetensors")
    if not matches_hf_patterns("config.json", EXPORT_HOST_ALLOW, None):
        _fail("export must download config.json")
    if export_download_bytes() < FULL_SAFE_TENSORS_BYTES:
        _fail("export size should include the full safetensors")
    # Inference is ANE + slim host + 256 MiB table, not ANE + the 1.49 GB weights.
    if inference_download_bytes() >= sum(ANE_BYTES.values()) + FULL_SAFE_TENSORS_BYTES:
        _fail("inference size still counts the full 1.49 GB weights")
    if inference_download_bytes() < EMBED_TABLE_BYTES + 1_000_000_000:
        _fail("inference size should include ANE towers plus the embed table")


def test_snapshot_passes_inference_patterns(tmp: Path) -> None:
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)

    with patch("scripts.download_common._hf_snapshot_download", fake):
        snapshot(
            "google/embeddinggemma-2",
            "rev",
            tmp / "model",
            allow_patterns=INFERENCE_HOST_ALLOW,
            ignore_patterns=INFERENCE_HOST_IGNORE,
        )
    if captured.get("allow_patterns") != list(INFERENCE_HOST_ALLOW):
        _fail(f"allow_patterns {captured.get('allow_patterns')}")
    if captured.get("ignore_patterns") != list(INFERENCE_HOST_IGNORE):
        _fail(f"ignore_patterns {captured.get('ignore_patterns')}")


def test_snapshot_export_omits_allow(tmp: Path) -> None:
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)

    with patch("scripts.download_common._hf_snapshot_download", fake):
        snapshot("google/embeddinggemma-2", "rev", tmp / "full", allow_patterns=EXPORT_HOST_ALLOW)
    if "allow_patterns" in captured:
        _fail(f"export snapshot should omit allow_patterns, got {captured}")


def _write_slim_host(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ):
        (folder / name).write_text("{}", encoding="utf-8")
    (folder / SLIM_EMBED_NAME).write_bytes(b"slim")


def test_download_host_prefers_mirrored(tmp: Path) -> None:
    ane = tmp / "ane"
    _write_slim_host(ane / HOST_FOLDER)
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)

    # Dummy payload: checksum verification is covered in test_release_hardening.
    with patch("scripts.download_common._hf_snapshot_download", fake), patch(
        "scripts.download_models.verify_mirrored_host", lambda *a, **k: "skipped"
    ):
        status = download_host(tmp / "embeddinggemma-2", ane_dir=ane, force=True)
    if status != "anemll-host":
        _fail(f"status {status}")
    if calls:
        _fail(f"mirrored host should not snapshot, got {calls}")
    model = tmp / "embeddinggemma-2"
    if not (model / SLIM_EMBED_NAME).is_file():
        _fail("model dir missing slim embed")


def test_download_host_falls_back_to_google(tmp: Path) -> None:
    ane = tmp / "ane"
    ane.mkdir(parents=True)
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(dict(kwargs))
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)

    with patch("scripts.download_common._hf_snapshot_download", fake):
        status = download_host(tmp / "embeddinggemma-2", ane_dir=ane, force=True)
    if status != "google-fallback":
        _fail(f"status {status}")
    if not calls:
        _fail("expected snapshot calls")
    last = calls[-1]
    if last.get("repo_id") != "google/embeddinggemma-2":
        _fail(f"fallback repo {last.get('repo_id')}")
    if last.get("allow_patterns") != list(INFERENCE_HOST_ALLOW):
        _fail(f"download_host allow {last.get('allow_patterns')}")
    if last.get("ignore_patterns") != list(INFERENCE_HOST_IGNORE):
        _fail(f"download_host ignore {last.get('ignore_patterns')}")
    first = calls[0]
    if first.get("allow_patterns") != list(HOST_FOLDER_ALLOW):
        _fail(f"first snapshot should be host/, got {first.get('allow_patterns')}")


def test_download_export_uses_full_allow(tmp: Path) -> None:
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        dest = Path(kwargs["local_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.safetensors").write_bytes(b"x")
        (dest / "config.json").write_text("{}", encoding="utf-8")

    with patch("scripts.download_common._hf_snapshot_download", fake):
        download_full_host(tmp / "embeddinggemma-2-full", force=True)
    if "allow_patterns" in captured:
        _fail(f"export download should omit allow_patterns, got {captured}")


def _tiny_full_checkpoint(src: Path) -> torch.Tensor:
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.bfloat16)
    data = weight.contiguous().view(torch.uint16).cpu().numpy().tobytes()
    extra = struct.pack("<ff", 9.0, 8.0)
    write_safetensors(
        src,
        {
            EMBED_TENSOR_KEY: {"dtype": "BF16", "shape": list(weight.shape), "data": data},
            "other.weight": {"dtype": "F32", "shape": [2], "data": extra},
        },
    )
    return weight.float()


def test_extract_and_load_slim(tmp: Path) -> None:
    src = tmp / "model.safetensors"
    weight = _tiny_full_checkpoint(src)
    dest = tmp / SLIM_EMBED_NAME
    extract_embed_from_file(src, dest)
    if not dest.is_file():
        _fail("slim file missing")
    (tmp / "config.json").write_text(json.dumps({"pad_token_id": 0}), encoding="utf-8")
    (tmp / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": {"SearchQuery": "task: search result | query: "}}),
        encoding="utf-8",
    )
    loaded = load_slim_host(tmp)
    if loaded is None:
        _fail("load_slim_host returned None")
    table, prompts = loaded
    if prompts.get("SearchQuery") != "task: search result | query: ":
        _fail(f"prompts {prompts}")
    ids = torch.tensor([[1]])
    got = table.get_input_embeddings()(ids)
    want = weight[1] * EMBED_SCALE
    if got.shape != (1, 1, 2):
        _fail(f"shape {tuple(got.shape)}")
    if not torch.allclose(got[0, 0].float(), want, atol=1e-3):
        _fail(f"scaled row {got[0, 0]} != {want}")


def test_extract_from_bytes_matches_file(tmp: Path) -> None:
    src = tmp / "model.safetensors"
    _tiny_full_checkpoint(src)
    a = tmp / "a.safetensors"
    b = tmp / "b.safetensors"
    extract_embed_from_file(src, a)
    extract_embed_from_bytes(src.read_bytes(), b)
    if a.read_bytes() != b.read_bytes():
        _fail("file and bytes extract diverged")


def test_prepare_hf_host_folder(tmp: Path) -> None:
    src = tmp / "src"
    dest = tmp / "hf"
    dest.mkdir(parents=True)
    (dest / "LICENSE").write_text("apache\n", encoding="utf-8")
    (dest / "NOTICE").write_text("notice\n", encoding="utf-8")
    for name in INFERENCE_HOST_ALLOW:
        (src).mkdir(parents=True, exist_ok=True)
        (src / name).write_text(f"{name}-bytes\n", encoding="utf-8")
    _tiny_full_checkpoint(src / "model.safetensors")
    host = dest / "host"
    rows = stage_host(src=src, host=host, license_src=dest / "LICENSE", notice_src=dest / "NOTICE")
    by_name = {str(row["file"]): row for row in rows}
    if by_name["config.json"]["origin"] != ORIGIN_COPIED:
        _fail("config.json should be copied verbatim")
    if by_name[SLIM_EMBED_NAME]["origin"] != ORIGIN_EXTRACTED:
        _fail("embed table should be extracted")
    source_md = (host / "SOURCE.md").read_text(encoding="utf-8")
    if "google/embeddinggemma-2" not in source_md:
        _fail("SOURCE.md missing upstream repo")
    if "914f7f89142e33e77833254d9c9b90c3cef7303b" not in source_md:
        _fail("SOURCE.md missing upstream revision")
    if "copied verbatim" not in source_md or "extracted" not in source_md:
        _fail("SOURCE.md missing origin labels")
    sums = (host / "SHA256SUMS").read_text(encoding="utf-8")
    if "config.json" not in sums or SLIM_EMBED_NAME not in sums:
        _fail(f"SHA256SUMS {sums}")
    if (host / "config.json").read_text(encoding="utf-8") != "config.json-bytes\n":
        _fail("verbatim copy drifted")
    if not (host / SLIM_EMBED_NAME).is_file():
        _fail("extracted embed missing")
    if (host / "LICENSE").read_text(encoding="utf-8") != "apache\n":
        _fail("LICENSE not copied into host/")
    if (src / SLIM_EMBED_NAME).exists():
        _fail("prepare must not write embed_tokens.safetensors into --src")


def test_ensure_slim_prefers_local_full(tmp: Path) -> None:
    _tiny_full_checkpoint(tmp / "model.safetensors")
    with patch("scripts.download_common.extract_embed_from_hf") as hf:
        status = ensure_slim_embed(tmp, force=True)
    if status != "extracted-local":
        _fail(f"status {status}")
    if hf.called:
        _fail("range extract should not run when a local full file exists")
    if not (tmp / SLIM_EMBED_NAME).is_file():
        _fail("slim embed missing after local extract")
    if load_slim_host(tmp) is None:
        _fail("slim host should load after extract")


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="anemll-dl-") as raw:
        tmp = Path(raw)
        test_bundle_complete(tmp / "bundles")
        test_symlink_idempotent(tmp / "links")
        test_revision_marker(tmp / "rev")
        test_env_exports()
        test_existing_dir_is_not_replaced(tmp / "clash")
        test_inference_allow_excludes_full_weights()
        test_export_allow_is_full_checkpoint()
        test_snapshot_passes_inference_patterns(tmp / "snap-inf")
        test_snapshot_export_omits_allow(tmp / "snap-exp")
        test_download_host_prefers_mirrored(tmp / "host-mirror")
        test_download_host_falls_back_to_google(tmp / "host-fallback")
        test_download_export_uses_full_allow(tmp / "export")
        test_prepare_hf_host_folder(tmp / "prepare")
        test_extract_and_load_slim(tmp / "slim")
        test_extract_from_bytes_matches_file(tmp / "bytes")
        test_ensure_slim_prefers_local_full(tmp / "local-full")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
