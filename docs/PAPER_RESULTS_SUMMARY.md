# Paper results summary

This repository contains the code and selected output tables for the paper's no-harm PIIL validation.

## Public PDE benchmark validation

The main public-PDE validation uses five small benchmark fields:

1. Burgers shock
2. Allen--Cahn
3. Korteweg--de Vries
4. Kuramoto--Sivashinsky
5. Nonlinear Schrödinger

The manuscript-aligned V3 run uses:

- 5 trials
- observation fractions: 0.03, 0.05, 0.10, 0.20
- noise fractions: 0.00, 0.01, 0.03, 0.05
- holdout fraction: 0.20
- random seed: 2026
- no-harm tolerance: 0.0

Headline empirical diagnostics:

- 400 dataset/regime cases
- 12.98% mean no-harm selected-output gain relative to the baseline
- 68.5% physics-necessary rate
- 4.2% certificate-driven fallback/rejection rate for physics-filtered candidates
- 6.0% unsafe selected-output rate

These diagnostics are conditional on the benchmark fields, candidate family, certificate weights, sparsity/noise grid, and seed.

## Controlled appendix stress tests

The controlled validation includes:

- Poisson inverse source recovery
- inverse heat initial-condition reconstruction
- limited-angle tomography
- elliptic coefficient identification
- stochastic residual validation
- a 21,000-trial no-harm sufficiency sweep

These experiments are intended as controlled certificate stress tests, not as evidence that any one neural architecture is necessary.
