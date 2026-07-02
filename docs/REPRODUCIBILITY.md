# Reproducibility guide

## Environment

Recommended Python version: 3.10 or later.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Public PDE benchmark validation

```bash
python scripts/download_light_piil_pde_data.py --out data/PIIL_LIGHT_PDE_DATA --set tiny
python scripts/check_piil_data.py
python scripts/piil_public_pde_validation_v3_gated.py --data data/PIIL_LIGHT_PDE_DATA --out results/public_pde_v3 --trials 5 --seed 2026
```

Expected main output tables are written to:

```text
results/public_pde_v3/tables/
```

The uploaded repository also includes the manuscript-aligned V3 tables under:

```text
outputs/public_pde_v3/tables/
```

## Controlled validation

```bash
python scripts/no_harm_piil_controlled_validation.py --out results/controlled --seed 123
```

## Sufficiency sweep

```bash
python scripts/no_harm_piil_sufficiency_map.py --out results/sufficiency_map --seed 2026
```

## Optional 3D Navier--Stokes stress test

```bash
python scripts/no_harm_piil_ns3d_industrial.py --mode smoke --out results/ns3d_smoke
```

Use `--mode quick` or `--mode full` for heavier runs.
