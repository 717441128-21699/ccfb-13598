"""Synthetic test-recording generator (numpy-only).

Usage:
    py -m app.synth.generate --scenario severe --minutes 30 --out data/samples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..wavio import write_wav
from .scenarios import SCENARIOS, generate_scenario
from .signals import SR


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic night recording")
    ap.add_argument("--scenario", choices=SCENARIOS, default="severe")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ahi", type=float, default=None, help="override target AHI")
    ap.add_argument("--out", type=str, default="data/samples")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = args.minutes * 60.0
    track, gt = generate_scenario(args.scenario, duration=duration,
                                  seed=args.seed, target_ahi=args.ahi)
    stem = f"{args.scenario}_{int(args.minutes)}min_seed{args.seed}"
    wav_path = out_dir / f"{stem}.wav"
    write_wav(wav_path, track, SR)
    meta_path = out_dir / f"{stem}.json"
    payload = gt.to_dict()
    payload["wav"] = wav_path.name
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"wrote {wav_path} ({duration:.0f}s, target AHI {gt.target_ahi:.1f}, "
          f"{len(gt.apneas)} apnea events)")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
