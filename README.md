# No-Harm PIIL: Industrial-Grade Validation Suite

This repository contains reproducible validation code for **no-harm physics-informed inverse learning (PIIL)**.

The central rule is simple:

```text
A learned physics-informed reconstruction replaces a robust baseline only when
its residual-calibrated certificate is no worse than the baseline certificate.

R_learn <= R_base + eps_safe
```

This makes the repository different from ordinary PINN demonstration code. It does not only show attractive plots. It records **data fit, PDE residuals, stability diagnostics, optimization/energy defects, uncertainty radii, and fallback decisions**.

## What is included

```text
src/noharm_piil/                  Reusable certificate and no-harm decision utilities
scripts/no_harm_piil_validation.py
                                  Multi-problem inverse-learning validation suite
scripts/no_harm_piil_sufficiency_map.py
                                  Sufficiency-threshold sweep and diagnostic maps
scripts/no_harm_piil_ns3d_industrial.py
                                  Full 3D incompressible Navier--Stokes validation
examples/                         Minimal examples and smoke-test wrapper
tests/                            Lightweight unit tests
docs/                             Technical notes and run guide
results_examples/                 Uploaded quick/smoke run tables and metadata
```

## Industrial 3D Navier--Stokes case

The third script runs a real 3D incompressible Navier--Stokes validation case on a periodic box:

```text
u_t + (u . grad)u = -grad p + nu Laplacian u,
div u = 0.
```

Numerics:

- Fourier pseudo-spectral discretization,
- Leray projection for incompressibility,
- 2/3 de-aliasing,
- explicit RK4 time stepping,
- Taylor--Green vortex initial condition.

It reports scientific-computing KPIs that matter in serious PDE work:

- sensor RMSE,
- momentum residual,
- divergence residual,
- kinetic-energy budget defect,
- high-wavenumber energy contamination,
- final DNS error,
- CFL,
- wall-clock cost,
- certificate radius,
- no-harm fallback decision.

## Install with Anaconda Prompt

```bat
conda create -n noharm_piil python=3.11 numpy pandas matplotlib scipy pytest -y
conda activate noharm_piil
pip install -e .
```

Or use the environment file:

```bat
conda env create -f environment.yml
conda activate noharm-piil
```

## Smoke test

```bat
python scripts\no_harm_piil_ns3d_industrial.py --mode smoke --out runs\ns3d_smoke
```

## Quick serious run

```bat
python scripts\no_harm_piil_ns3d_industrial.py --mode quick --out runs\ns3d_quick
```

## Main validation suite

```bat
python scripts\no_harm_piil_validation.py --out runs\validation_suite --seed 123
```

## Sufficiency map

```bat
python scripts\no_harm_piil_sufficiency_map.py --out runs\sufficiency_map --seed 2026 --reps 25
```

For a faster local check:

```bat
python scripts\no_harm_piil_sufficiency_map.py --out runs\sufficiency_smoke --seed 2026 --reps 2
```

## Test the repository

```bat
python -m pytest
python -m py_compile scripts\no_harm_piil_validation.py scripts\no_harm_piil_sufficiency_map.py scripts\no_harm_piil_ns3d_industrial.py
```

## Expected no-harm behavior

The 3D Navier--Stokes script is designed to distinguish candidates:

- `baseline_robust_dns`: conservative, safe fallback;
- `learned_calibrated_dns`: calibrated learned parameter DNS, usually accepted;
- `learned_aggressive_dns`: risky extrapolation, usually rejected;
- `learned_sensor_overfit_field`: sensor-fitting high-wavenumber field, rejected by residual/energy/spectrum checks.

The uploaded quick run selected:

```text
safe_choice = learned_calibrated_dns
```

## Important scope note

This repository is not a replacement for production CFD solvers such as OpenFOAM, Nek5000, Dedalus, or spectralDNS. It is a compact, reproducible, dependency-light validation platform for **no-harm certification logic** in physics-informed inverse learning.

## Citation

Use `CITATION.cff` or cite the associated manuscript/software release.
