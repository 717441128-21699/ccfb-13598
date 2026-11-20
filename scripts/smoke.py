"""Quick local smoke test: synth -> WAV -> full pipeline -> summary."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import analyze
from app.synth.scenarios import generate_scenario
from app.wavio import write_wav
from app.synth.signals import SR


def main(scenario: str = "severe", minutes: float = 5.0) -> None:
    track, gt = generate_scenario(scenario, duration=minutes * 60, seed=42)
    tmp = Path("data/samples") / f"_smoke_{scenario}.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    write_wav(tmp, track, SR)

    t0 = time.perf_counter()
    result = analyze(str(tmp), job_id=f"smoke-{scenario}", metadata={"positions": gt.positions})
    dt = time.perf_counter() - t0

    print(f"scenario={scenario} minutes={minutes}")
    print(f"target AHI={gt.target_ahi:.1f} true_apneas={len(gt.apneas)} "
          f"true_partner={len(gt.partner_snores)}")
    print(f"detected: {result['ahi']}")
    print(f"counts: {result['counts']}")
    print(f"clips: {len(result['clips'])} hourly_rows={len(result['hourly'])} "
          f"positions={[p['position_label'] for p in result['positions']]}")
    print(f"wall={dt:.2f}s realtime_factor={result['realtime_factor']}")
    if result["apneas"]:
        durs = [a["duration"] for a in result["apneas"]]
        print(f"apnea durations min/mean/max = {min(durs):.1f}/"
              f"{sum(durs)/len(durs):.1f}/{max(durs):.1f}")


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "severe"
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    main(scenario, minutes)
