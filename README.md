# No-Harm Physics-Informed Inverse Learning (PIIL)

Repository for the numerical validation code accompanying the manuscript:

**No-Harm Physics-Informed Inverse Learning with Residual-Certified Selection**

The core principle is simple: a learned physics-informed inverse reconstruction is allowed to replace a baseline only when its residual-calibrated certificate is no worse than the baseline certificate:

```math
R_{\mathrm{learn}} \le R_{\mathrm{base}} + \varepsilon_{\mathrm{safe}}.
```

If this condition fails, the method falls back to the baseline. The repository is therefore organized around **reconstruct, certify, and select**, not around a single neural architecture.

## Repository structure

```text
scripts/
  download_light_piil_pde_data.py          Download public PDE benchmark .mat files.
  check_piil_data.py                       Inspect downloaded benchmark files.
  piil_public_pde_validation_v3_gated.py   Main public-PDE validation used in the manuscript.
  piil_public_pde_validation_v1.py         Earlier public-PDE validation script.
  no_harm_piil_controlled_validation.py    Controlled finite-dimensional inverse-problem validation.
  no_harm_piil_sufficiency_map.py          21,000-trial no-harm sufficiency sweep.
  no_harm_piil_ns3d_industrial.py          Optional 3D Navier--Stokes industrial/scientific-computing stress test.

outputs/public_pde_v3/tables/
  dataset_summary.csv
  candidate_summary.csv
  physics_necessity_summary.csv
  noharm_decisions.csv
  piil_trials_raw.csv
  run_metadata.json
  manuscript_findings.txt
  manuscript_main_table.tex
  manuscript_necessity_table.tex

docs/
  REPRODUCIBILITY.md
  PAPER_RESULTS_SUMMARY.md
  GITHUB_DESCRIPTION.md
```

## Quick start

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Download the public PDE benchmark data:

```bash
python scripts/download_light_piil_pde_data.py --out data/PIIL_LIGHT_PDE_DATA --set tiny
python scripts/check_piil_data.py
```

Run the main public-PDE validation:

```bash
python scripts/piil_public_pde_validation_v3_gated.py \
  --data data/PIIL_LIGHT_PDE_DATA \
  --out results/public_pde_v3 \
  --trials 5 \
  --seed 2026
```

Run the controlled inverse-problem validation:

```bash
python scripts/no_harm_piil_controlled_validation.py --out results/controlled --seed 123
```

Run the no-harm sufficiency sweep:

```bash
python scripts/no_harm_piil_sufficiency_map.py --out results/sufficiency_map --seed 2026
```

Optional 3D Navier--Stokes smoke test:

```bash
python scripts/no_harm_piil_ns3d_industrial.py --mode smoke --out results/ns3d_smoke
```

## Main manuscript numbers

The included V3 public-PDE tables correspond to:

- Five public PDE benchmark fields: Burgers shock, Allen--Cahn, Korteweg--de Vries, Kuramoto--Sivashinsky, and nonlinear Schrödinger.
- 400 dataset/regime cases.
- Mean no-harm selected-output gain of 12.98% relative to the baseline.
- Physics-informed candidates necessary in 68.5% of regimes.
- Certificate-driven fallback/rejection rate for physics-filtered candidates of 4.2%.
- Unsafe selected-output rate of 6.0%.

These are empirical diagnostics for the stated candidate family, certificate weights, sparsity/noise grid, and random seed. They are not universal constants.

## Data note

The `.mat` benchmark files are not stored in this repository. Use `scripts/download_light_piil_pde_data.py` to download them from the public PINNs repository. Generated summary CSVs and manuscript-ready tables are included under `outputs/public_pde_v3/tables/` for reproducibility and review.

## Citation

If you use this repository, please cite the accompanying manuscript and the public PINNs benchmark source used for the PDE data.
