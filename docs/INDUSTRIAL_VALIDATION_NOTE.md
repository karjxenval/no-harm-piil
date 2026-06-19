# Industrial validation note

The repository has three validation layers.

## 1. Canonical inverse problems

`no_harm_piil_validation.py` validates the certificate logic on Poisson source recovery, inverse heat recovery, limited-angle tomography, elliptic coefficient identification, and stochastic collocation residual checks.

## 2. Sufficiency threshold maps

`no_harm_piil_sufficiency_map.py` explores the operating regime where learned physics-informed candidates are sufficient to replace a baseline. It records acceptance rate, unsafe acceptance, false rejection, and safe improvement.

## 3. 3D Navier--Stokes industrial stress test

`no_harm_piil_ns3d_industrial.py` runs a full 3D incompressible Navier--Stokes pseudo-spectral solver and evaluates whether learned/data-assisted candidates satisfy the no-harm replacement rule.

The important point is that a candidate is not accepted because it is visually smooth or sensor-fitting. It is accepted only when the certificate radius is no worse than the baseline radius.
