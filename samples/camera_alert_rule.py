#!/usr/bin/env python3
"""Score a frame the way the camera-alert page does.

    python samples/camera_alert_rule.py --backend mock
    python samples/camera_alert_rule.py --image truck.jpg --street street.jpg --rule "a UPS truck"

A text rule is cosine(frame, rule-text). "Anything significant" is
``1 - cosine(frame, empty-street)`` — a change from the baseline photo,
not ``1 - cosine`` against the rule text. The empty-street frame itself is 0.

``--street`` defaults to ``$ANEMLL_DEMO_ALERT/frames/street.jpg`` when that
file exists (after ``samples/fetch_alert.py``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api import cosine  # noqa: E402
from samples.cli import add_embedder_args, make_embedder  # noqa: E402


def _default_street() -> Path | None:
    raw = os.environ.get("ANEMLL_DEMO_ALERT")
    root = Path(raw) if raw else Path.home() / ".anemll-embeddings" / "alert"
    path = root / "frames" / "street.jpg"
    return path if path.is_file() else None


def _same_file(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except OSError:
        return left == right


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--street",
        type=Path,
        default=None,
        help="empty-street baseline photo (Anything significant)",
    )
    parser.add_argument("--rule", default="a UPS truck")
    parser.add_argument("--threshold", type=float, default=0.65)
    add_embedder_args(parser)
    args = parser.parse_args(argv)

    street_path = args.street if args.street is not None else _default_street()
    if args.image is None:
        image = Image.new("RGB", (64, 64), (80, 40, 20))
    else:
        image = Image.open(args.image).convert("RGB")
    if street_path is None:
        street_image = Image.new("RGB", (64, 64), (140, 150, 160))
        street_label = "generated empty-street placeholder"
    else:
        street_image = Image.open(street_path).convert("RGB")
        street_label = str(street_path)

    embedder = make_embedder(args)
    frame = embedder.embed_image(image)
    street = embedder.embed_image(street_image)
    rule = embedder.embed_text(args.rule, role="query")
    embedder.close()

    score = cosine(frame, rule)
    fired = score >= float(args.threshold)
    if _same_file(args.image, street_path):
        significance = 0.0
    else:
        significance = max(0.0, 1.0 - cosine(frame, street))

    print(f"rule={args.rule!r}")
    print(f"cosine(frame, rule)={score:.4f}  threshold={args.threshold:.3f}  fired={fired}")
    print(
        f"1-cosine(frame, empty-street)={significance:.4f}  "
        f"(Anything significant; street={street_label})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
