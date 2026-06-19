#!/usr/bin/env python
"""Run lightweight checks for the no-harm PIIL repository."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd):
    print("\n$", " ".join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], cwd=ROOT)

code = 0
code |= run([sys.executable, "-m", "pytest"])
code |= run([sys.executable, "-m", "py_compile",
             "scripts/no_harm_piil_validation.py",
             "scripts/no_harm_piil_sufficiency_map.py",
             "scripts/no_harm_piil_ns3d_industrial.py"])
code |= run([sys.executable, "scripts/no_harm_piil_ns3d_industrial.py",
             "--mode", "smoke", "--out", "runs/ns3d_smoke"])
raise SystemExit(code)
