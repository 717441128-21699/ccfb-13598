"""Measure analysis-only throughput and extrapolate to an 8-hour night."""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import analyze
from app.synth.scenarios import generate_scenario
from app.synth.signals import SR
from app.wavio import write_wav

mins = int(sys.argv[1]) if len(sys.argv) > 1 else 120
track, gt = generate_scenario("severe", duration=mins * 60, seed=42,
                              target_ahi=45.0)
d = tempfile.mkdtemp()
p = os.path.join(d, f"{mins}min.wav")
write_wav(p, track, SR)
t0 = time.perf_counter()
r = analyze(p, job_id=f"perf{mins}", metadata={"positions": gt.positions})
dt = time.perf_counter() - t0
factor = 8 * 60 / mins
print(f"{mins}min analysis: {dt:.1f}s; projected 8h ~ {dt * factor:.0f}s "
      f"(limit 600s); RTF={r['realtime_factor']}")
print("AHI =", r["ahi"]["ahi"], "apneas =", r["ahi"]["apnea_count"],
      "clips =", len(r["clips"]))
