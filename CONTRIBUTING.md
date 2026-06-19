# Contributing

This repository is research software. Contributions should preserve three principles:

1. **Reproducibility**: every experiment must expose a seed and write machine-readable tables.
2. **No-harm logic**: learned candidates must be compared to a baseline by certified radius, not visual quality alone.
3. **Scientific-computing discipline**: report residuals, stability diagnostics, and failure modes.

Before opening a pull request, run:

```bash
python -m pytest
python -m py_compile scripts/no_harm_piil_validation.py scripts/no_harm_piil_sufficiency_map.py scripts/no_harm_piil_ns3d_industrial.py
```
