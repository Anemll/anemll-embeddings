#!/usr/bin/env python3
"""Score a frame against a text rule the way the camera-alert page does.

    python samples/camera_alert_rule.py --backend mock
    python samples/camera_alert_rule.py --image truck.jpg --rule "a UPS truck"

The score is cosine(frame, rule-text). The camera-alert demo also has a
separate "significant" line that is 1 - cosine(frame, empty-street).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api import Embedder  # noqa: E402
from demo.alert_score import change_score  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--rule", default="a UPS truck")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--backend", default="coreai", choices=("coreai", "mock", "reference"))
    args = parser.parse_args(argv)

    if args.image is None:
        image = Image.new("RGB", (64, 64), (80, 40, 20))
    else:
        image = Image.open(args.image).convert("RGB")

    embedder = Embedder(backend=args.backend)
    frame = embedder.embed_image(image)
    rule = embedder.embed_text(args.rule, role="query")
    embedder.close()

    score = float(frame @ rule)
    significance = change_score(score, baseline=False)
    fired = score >= float(args.threshold)
    print(f"rule={args.rule!r}")
    print(f"cosine={score:.4f}  threshold={args.threshold:.3f}  fired={fired}")
    print(f"1-cosine (significance-style)={significance:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
