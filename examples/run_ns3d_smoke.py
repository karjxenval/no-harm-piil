#!/usr/bin/env python
"""Convenience wrapper for the industrial 3D Navier--Stokes smoke test."""
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
script = repo / "scripts" / "no_harm_piil_ns3d_industrial.py"
out = repo / "runs" / "ns3d_smoke"
cmd = [sys.executable, str(script), "--mode", "smoke", "--out", str(out)]
print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd))
