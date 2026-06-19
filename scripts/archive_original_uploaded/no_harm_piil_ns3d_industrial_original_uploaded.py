#!/usr/bin/env python3
"""
Industrial-grade no-harm PIIL validation on the full 3D incompressible
Navier--Stokes equations in a periodic box.

This is the third script in the no-harm PIIL validation family. Unlike the
smaller inverse-problem demonstrations, this script runs an actual 3D
pseudo-spectral incompressible Navier--Stokes simulation and then evaluates
whether learned / data-assisted candidates are safe enough to replace a robust
baseline.

Core idea
---------
A learned physics-informed candidate is allowed to replace the baseline only if
its residual-calibrated operational certificate is no worse than the baseline:

    R_candidate <= R_baseline + eps_safe.

This script evaluates that rule using industrial/scientific-computing KPIs:
    - sensor-data fit,
    - 3D momentum residual,
    - incompressibility residual,
    - kinetic-energy budget defect,
    - high-wavenumber energy contamination,
    - final-state error against a reference DNS,
    - CFL and wall-clock cost,
    - no-harm fallback decision.

Model
-----
Full 3D incompressible Navier--Stokes on [0, 2*pi]^3 with periodic boundary:

    u_t + (u . grad)u = -grad p + nu Laplacian u,
    div u = 0.

Numerics
--------
    - Fourier pseudo-spectral discretization.
    - Leray projection to enforce incompressibility.
    - 2/3 de-aliasing for nonlinear terms.
    - Explicit RK4 time stepping.
    - Taylor--Green vortex initial condition.

The script is intentionally dependency-light: numpy, pandas, matplotlib.
It is not a replacement for OpenFOAM, Nek5000, Dedalus, or spectralDNS. Its
purpose is to provide a rigorous, reproducible, self-contained 3D DNS-level
validation case for the no-harm PIIL decision principle.

Outputs
-------
<out>/
    tables/
        ns3d_candidate_summary.csv
        ns3d_noharm_decisions.csv
        ns3d_timeseries.csv
        ns3d_run_metadata.json
    figures/
        ns3d_energy_enstrophy_decay.png/pdf
        ns3d_error_vs_certificate_radius.png/pdf
        ns3d_industrial_kpis.png/pdf
        ns3d_velocity_slice_panel.png/pdf
        ns3d_vorticity_slice_panel.png/pdf
        ns3d_energy_spectra_final.png/pdf

Examples
--------
Smoke test, small and fast:

    python no_harm_piil_ns3d_industrial.py --mode smoke --out results_ns3d_smoke

Quick serious run:

    python no_harm_piil_ns3d_industrial.py --mode quick --out results_ns3d_quick

Heavier run:

    python no_harm_piil_ns3d_industrial.py --mode full --out results_ns3d_full

Custom run:

    python no_harm_piil_ns3d_industrial.py --N 32 --T 0.12 --dt 0.001 --n-sensors 96 --out results_custom
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class Grid3D:
    N: int
    L: float
    dx: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    KX: np.ndarray
    KY: np.ndarray
    KZ: np.ndarray
    K2: np.ndarray
    K2_safe: np.ndarray
    Kmag: np.ndarray
    dealias: np.ndarray
    kmax_abs: float


@dataclass
class SimulationResult:
    name: str
    model_kind: str
    nu: float
    amplitude: float
    dt: float
    T: float
    wall_seconds: float
    u_prev: np.ndarray
    u_final: np.ndarray
    diagnostics: pd.DataFrame
    notes: str


@dataclass
class CandidateMetrics:
    candidate: str
    model_kind: str
    accepted_by_noharm: Optional[bool]
    safe_choice: Optional[str]
    radius: float
    baseline_radius: float
    certificate_ratio: float
    final_rel_l2_error: float
    sensor_rmse_normalized: float
    momentum_residual_normalized: float
    divergence_l2: float
    divergence_max: float
    energy_budget_defect_normalized: float
    high_k_energy_ratio: float
    kinetic_energy_final: float
    energy_error_percent: float
    enstrophy_final: float
    dissipation_final: float
    max_cfl: float
    wall_seconds: float
    C_stab: float
    sigma_min_observability: float
    observability_condition_number: float
    notes: str


# =============================================================================
# General utilities
# =============================================================================

def ensure_dirs(out: str) -> Tuple[str, str]:
    fig_dir = os.path.join(out, "figures")
    tab_dir = os.path.join(out, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    return fig_dir, tab_dir


def savefig(fig: plt.Figure, fig_dir: str, name: str) -> None:
    # Some multi-axis figures with shared colorbars emit harmless tight-layout warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), bbox_inches="tight", dpi=250)
    fig.savefig(os.path.join(fig_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def rms_vector_field(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(v * v, axis=0))))


def rel_l2(u: np.ndarray, v: np.ndarray, eps: float = 1e-14) -> float:
    return float(np.linalg.norm((u - v).ravel()) / (np.linalg.norm(v.ravel()) + eps))


def make_grid(N: int, L: float = 2.0 * np.pi) -> Grid3D:
    if N < 8:
        raise ValueError("N must be at least 8 for a meaningful 3D spectral run.")
    dx = L / N
    one = np.linspace(0.0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(one, one, one, indexing="ij")

    k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    K2 = KX * KX + KY * KY + KZ * KZ
    K2_safe = K2.copy()
    K2_safe[K2_safe == 0.0] = 1.0
    Kmag = np.sqrt(K2)

    # 2/3 de-aliasing: for L=2*pi, these are approximately integer modes.
    cutoff = N / 3.0
    dealias = (
        (np.abs(KX) <= cutoff)
        & (np.abs(KY) <= cutoff)
        & (np.abs(KZ) <= cutoff)
    )

    return Grid3D(
        N=N,
        L=L,
        dx=dx,
        x=one,
        y=one,
        z=one,
        X=X,
        Y=Y,
        Z=Z,
        KX=KX,
        KY=KY,
        KZ=KZ,
        K2=K2,
        K2_safe=K2_safe,
        Kmag=Kmag,
        dealias=dealias,
        kmax_abs=float(np.max(np.abs(k1))),
    )


# =============================================================================
# Fourier Navier--Stokes solver
# =============================================================================

def fft_vec(u: np.ndarray) -> np.ndarray:
    return np.stack([np.fft.fftn(u[i]) for i in range(3)], axis=0)


def ifft_vec(uhat: np.ndarray) -> np.ndarray:
    return np.stack([np.fft.ifftn(uhat[i]).real for i in range(3)], axis=0)


def apply_dealias(uhat: np.ndarray, grid: Grid3D) -> np.ndarray:
    out = uhat.copy()
    out[:, ~grid.dealias] = 0.0
    return out


def project_div_free(uhat: np.ndarray, grid: Grid3D) -> np.ndarray:
    """Leray projection in Fourier space."""
    div_hat = 1j * (grid.KX * uhat[0] + grid.KY * uhat[1] + grid.KZ * uhat[2])
    phi_hat = div_hat / grid.K2_safe
    out = uhat.copy()
    out[0] += 1j * grid.KX * phi_hat
    out[1] += 1j * grid.KY * phi_hat
    out[2] += 1j * grid.KZ * phi_hat

    # Remove any numerical mean pressure artefact at zero mode.
    zero = (grid.K2 == 0.0)
    out[:, zero] = uhat[:, zero]
    return out


def spectral_derivative(uhat_component: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.fft.ifftn(1j * K * uhat_component).real


def ns_rhs_hat(uhat: np.ndarray, grid: Grid3D, nu: float) -> np.ndarray:
    """Fourier RHS for incompressible Navier--Stokes."""
    uhat = project_div_free(apply_dealias(uhat, grid), grid)
    u = ifft_vec(uhat)

    grad = np.empty((3, 3, grid.N, grid.N, grid.N), dtype=float)
    Ks = [grid.KX, grid.KY, grid.KZ]
    for i in range(3):
        for j in range(3):
            grad[i, j] = spectral_derivative(uhat[i], Ks[j])

    nonlinear = np.empty_like(u)
    for i in range(3):
        nonlinear[i] = u[0] * grad[i, 0] + u[1] * grad[i, 1] + u[2] * grad[i, 2]

    nlin_hat = fft_vec(nonlinear)
    nlin_hat = apply_dealias(nlin_hat, grid)
    rhs = project_div_free(-nlin_hat, grid)
    rhs -= nu * grid.K2[None, :, :, :] * uhat
    rhs = apply_dealias(rhs, grid)
    return rhs


def rk4_step(uhat: np.ndarray, grid: Grid3D, nu: float, dt: float) -> np.ndarray:
    k1 = ns_rhs_hat(uhat, grid, nu)
    k2 = ns_rhs_hat(uhat + 0.5 * dt * k1, grid, nu)
    k3 = ns_rhs_hat(uhat + 0.5 * dt * k2, grid, nu)
    k4 = ns_rhs_hat(uhat + dt * k3, grid, nu)
    unew = uhat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    unew = project_div_free(apply_dealias(unew, grid), grid)
    return unew


def taylor_green_initial_condition(
    grid: Grid3D,
    amplitude: float = 1.0,
    perturbation: float = 0.0,
) -> np.ndarray:
    """Divergence-free Taylor--Green vortex on [0, 2*pi]^3."""
    X, Y, Z = grid.X, grid.Y, grid.Z
    u = np.zeros((3, grid.N, grid.N, grid.N), dtype=float)
    u[0] = amplitude * np.sin(X) * np.cos(Y) * np.cos(Z)
    u[1] = -amplitude * np.cos(X) * np.sin(Y) * np.cos(Z)
    u[2] = 0.0

    if perturbation > 0.0:
        # Small smooth divergence-free perturbation from a vector potential.
        psi = perturbation * np.sin(2.0 * X + 0.3) * np.sin(3.0 * Y + 0.4) * np.sin(2.0 * Z + 0.7)
        psihat = np.fft.fftn(psi)
        dpsi_dy = np.fft.ifftn(1j * grid.KY * psihat).real
        dpsi_dx = np.fft.ifftn(1j * grid.KX * psihat).real
        u[0] += dpsi_dy
        u[1] -= dpsi_dx
        uhat = project_div_free(apply_dealias(fft_vec(u), grid), grid)
        u = ifft_vec(uhat)
    return u


def divergence_field(u: np.ndarray, grid: Grid3D) -> np.ndarray:
    uhat = fft_vec(u)
    div = np.fft.ifftn(1j * (grid.KX * uhat[0] + grid.KY * uhat[1] + grid.KZ * uhat[2])).real
    return div


def vorticity_field(u: np.ndarray, grid: Grid3D) -> np.ndarray:
    uhat = fft_vec(u)
    wx = spectral_derivative(uhat[2], grid.KY) - spectral_derivative(uhat[1], grid.KZ)
    wy = spectral_derivative(uhat[0], grid.KZ) - spectral_derivative(uhat[2], grid.KX)
    wz = spectral_derivative(uhat[1], grid.KX) - spectral_derivative(uhat[0], grid.KY)
    return np.stack([wx, wy, wz], axis=0)


def kinetic_energy(u: np.ndarray) -> float:
    return float(0.5 * np.mean(np.sum(u * u, axis=0)))


def enstrophy(u: np.ndarray, grid: Grid3D) -> float:
    w = vorticity_field(u, grid)
    return float(0.5 * np.mean(np.sum(w * w, axis=0)))


def dissipation_rate(u: np.ndarray, grid: Grid3D, nu: float) -> float:
    # For divergence-free periodic flow, dissipation = nu * mean(|curl u|^2).
    return float(2.0 * nu * enstrophy(u, grid))


def max_cfl(u: np.ndarray, grid: Grid3D, dt: float) -> float:
    speed = np.sqrt(np.sum(u * u, axis=0))
    return float(np.max(speed) * dt / grid.dx)


def compute_diagnostics(
    u: np.ndarray,
    grid: Grid3D,
    nu: float,
    dt: float,
    t: float,
    step: int,
    candidate: str,
) -> Dict[str, float]:
    div = divergence_field(u, grid)
    ens = enstrophy(u, grid)
    return {
        "candidate": candidate,
        "step": int(step),
        "time": float(t),
        "kinetic_energy": kinetic_energy(u),
        "enstrophy": ens,
        "dissipation": 2.0 * nu * ens,
        "divergence_l2": float(np.sqrt(np.mean(div * div))),
        "divergence_max": float(np.max(np.abs(div))),
        "max_cfl": max_cfl(u, grid, dt),
    }


def integrate_navier_stokes(
    name: str,
    model_kind: str,
    grid: Grid3D,
    nu: float,
    amplitude: float,
    T: float,
    dt: float,
    perturbation: float,
    diagnostics_stride: int,
    notes: str,
    verbose: bool = False,
) -> SimulationResult:
    n_steps = int(round(T / dt))
    if n_steps < 2:
        raise ValueError("T/dt must give at least two time steps.")
    T_eff = n_steps * dt

    u0 = taylor_green_initial_condition(grid, amplitude=amplitude, perturbation=perturbation)
    uhat = project_div_free(apply_dealias(fft_vec(u0), grid), grid)

    records: List[Dict[str, float]] = []
    u_prev = ifft_vec(uhat)
    t0 = time.perf_counter()

    records.append(compute_diagnostics(u_prev, grid, nu, dt, 0.0, 0, name))
    for step in range(1, n_steps + 1):
        if verbose and (step == 1 or step == n_steps or step % max(1, n_steps // 5) == 0):
            print(f"[{name}] step {step}/{n_steps}")
        u_prev = ifft_vec(uhat)
        uhat = rk4_step(uhat, grid, nu, dt)
        if step % diagnostics_stride == 0 or step == n_steps:
            u_now = ifft_vec(uhat)
            records.append(compute_diagnostics(u_now, grid, nu, dt, step * dt, step, name))

    wall = time.perf_counter() - t0
    u_final = ifft_vec(uhat)
    return SimulationResult(
        name=name,
        model_kind=model_kind,
        nu=nu,
        amplitude=amplitude,
        dt=dt,
        T=T_eff,
        wall_seconds=wall,
        u_prev=u_prev,
        u_final=u_final,
        diagnostics=pd.DataFrame(records),
        notes=notes,
    )


# =============================================================================
# Sensors, randomized observability, and overfit-field construction
# =============================================================================

def choose_sensors(grid: Grid3D, n_sensors: int, rng: np.random.Generator) -> np.ndarray:
    total = grid.N ** 3
    if n_sensors > total:
        raise ValueError("n_sensors cannot exceed N^3.")
    flat = rng.choice(total, size=n_sensors, replace=False)
    return np.column_stack(np.unravel_index(flat, (grid.N, grid.N, grid.N))).astype(int)


def observe(u: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    vals = []
    for i, j, k in sensors:
        vals.extend([u[0, i, j, k], u[1, i, j, k], u[2, i, j, k]])
    return np.asarray(vals, dtype=float)


def random_div_free_field(
    grid: Grid3D,
    rng: np.random.Generator,
    low: float,
    high: float,
    normalize: bool = True,
) -> np.ndarray:
    """Random divergence-free spectral field restricted to a wavenumber shell."""
    raw = rng.normal(size=(3, grid.N, grid.N, grid.N))
    uhat = fft_vec(raw)
    shell = (grid.Kmag >= low) & (grid.Kmag <= high)
    uhat[:, ~shell] = 0.0
    uhat = project_div_free(uhat, grid)
    u = ifft_vec(uhat)
    if normalize:
        nrm = rms_vector_field(u)
        if nrm > 1e-14:
            u = u / nrm
    return u


def randomized_observability_constant(
    grid: Grid3D,
    sensors: np.ndarray,
    rng: np.random.Generator,
    n_basis: int,
    k_low: float,
    k_high: float,
    floor: float = 1e-10,
) -> Tuple[float, float, float]:
    """
    Approximate a conditional stability constant on a randomized smooth,
    divergence-free admissible subspace.

    This is not a theorem-level constant. It is an operational observability
    diagnostic: small sigma_min means the sensor layout poorly observes the
    chosen admissible flow class.
    """
    obs_dim = 3 * len(sensors)
    n_basis = int(min(n_basis, max(2, obs_dim - 1)))
    A = np.zeros((obs_dim, n_basis), dtype=float)
    for j in range(n_basis):
        phi = random_div_free_field(grid, rng, low=k_low, high=k_high, normalize=True)
        A[:, j] = observe(phi, sensors)
    svals = np.linalg.svd(A, compute_uv=False)
    sigma_min = float(np.min(svals))
    sigma_max = float(np.max(svals))
    Cstab = float(1.0 / max(sigma_min, floor))
    cond = float(sigma_max / max(sigma_min, floor))
    return Cstab, sigma_min, cond


def fit_sensor_overfit_field(
    base: SimulationResult,
    y: np.ndarray,
    sensors: np.ndarray,
    grid: Grid3D,
    rng: np.random.Generator,
    n_basis: int,
    ridge: float,
    max_correction_fraction: float,
) -> SimulationResult:
    """
    Build a deliberately risky data-assisted final field. It fits sensors using
    random high-wavenumber divergence-free basis functions but is not obtained
    by time integration of Navier--Stokes. The no-harm certificate should catch
    this through momentum residual, energy budget defect, and spectral pollution.
    """
    obs_dim = len(y)
    n_basis = int(min(n_basis, max(4, obs_dim - 1)))
    fields: List[np.ndarray] = []
    A = np.zeros((obs_dim, n_basis), dtype=float)

    # High but resolvable modes. Do not de-alias them: this is a risky learned field.
    low = max(2.0, grid.N / 4.0)
    high = max(low + 1.0, grid.N / 2.0 - 1.0)
    for j in range(n_basis):
        phi = random_div_free_field(grid, rng, low=low, high=high, normalize=True)
        fields.append(phi)
        A[:, j] = observe(phi, sensors)

    residual = y - observe(base.u_final, sensors)
    ATA = A.T @ A + ridge * np.eye(n_basis)
    coeff = np.linalg.solve(ATA, A.T @ residual)
    correction = np.zeros_like(base.u_final)
    for c, phi in zip(coeff, fields):
        correction += c * phi

    base_rms = rms_vector_field(base.u_final)
    corr_rms = rms_vector_field(correction)
    max_corr = max_correction_fraction * max(base_rms, 1e-14)
    if corr_rms > max_corr:
        correction *= max_corr / corr_rms

    u_overfit = base.u_final + correction
    uhat = project_div_free(fft_vec(u_overfit), grid)
    u_overfit = ifft_vec(uhat)

    records = base.diagnostics.copy()
    last = records.iloc[-1].to_dict()
    div = divergence_field(u_overfit, grid)
    ens = enstrophy(u_overfit, grid)
    records.loc[len(records)] = {
        "candidate": "learned_sensor_overfit_field",
        "step": int(last["step"]),
        "time": float(last["time"]),
        "kinetic_energy": kinetic_energy(u_overfit),
        "enstrophy": ens,
        "dissipation": 2.0 * base.nu * ens,
        "divergence_l2": float(np.sqrt(np.mean(div * div))),
        "divergence_max": float(np.max(np.abs(div))),
        "max_cfl": max_cfl(u_overfit, grid, base.dt),
    }

    return SimulationResult(
        name="learned_sensor_overfit_field",
        model_kind="data-assisted field correction, not time-integrated DNS",
        nu=base.nu,
        amplitude=base.amplitude,
        dt=base.dt,
        T=base.T,
        wall_seconds=0.0,
        u_prev=base.u_prev,
        u_final=u_overfit,
        diagnostics=records,
        notes="Risky high-wavenumber sensor-fitting field; useful as a no-harm stress test.",
    )


# =============================================================================
# Industrial no-harm metrics
# =============================================================================

def momentum_residual_bdf1(
    u_prev: np.ndarray,
    u_curr: np.ndarray,
    grid: Grid3D,
    nu: float,
    dt: float,
) -> float:
    uhat_curr = project_div_free(fft_vec(u_curr), grid)
    rhs_curr = ifft_vec(ns_rhs_hat(uhat_curr, grid, nu))
    residual = (u_curr - u_prev) / dt - rhs_curr
    return rms_vector_field(residual)


def energy_budget_defect(
    u_prev: np.ndarray,
    u_curr: np.ndarray,
    grid: Grid3D,
    nu: float,
    dt: float,
) -> float:
    dE_dt = (kinetic_energy(u_curr) - kinetic_energy(u_prev)) / dt
    diss = dissipation_rate(u_curr, grid, nu)
    return float(abs(dE_dt + diss))


def high_k_energy_ratio(u: np.ndarray, grid: Grid3D, cutoff_fraction: float = 2.0 / 3.0) -> float:
    uhat = fft_vec(u)
    energy_modes = np.sum(np.abs(uhat) ** 2, axis=0)
    total = float(np.sum(energy_modes))
    if total <= 1e-30:
        return 0.0
    cutoff = cutoff_fraction * grid.kmax_abs
    high = float(np.sum(energy_modes[grid.Kmag >= cutoff]))
    return high / total


def energy_spectrum(u: np.ndarray, grid: Grid3D) -> pd.DataFrame:
    uhat = fft_vec(u)
    modal_energy = 0.5 * np.sum(np.abs(uhat) ** 2, axis=0) / (grid.N ** 6)
    bins = np.floor(grid.Kmag + 0.5).astype(int)
    kmax = int(np.max(bins))
    rows = []
    for k in range(kmax + 1):
        mask = bins == k
        if np.any(mask):
            rows.append({"k": k, "energy": float(np.sum(modal_energy[mask]))})
    return pd.DataFrame(rows)


def compute_operational_radius(
    Cstab: float,
    data_residual: float,
    momentum_residual: float,
    divergence_residual: float,
    energy_budget_residual: float,
    high_k_residual: float,
    delta_bound: float,
    alpha_momentum: float,
    alpha_divergence: float,
    alpha_energy: float,
    alpha_highk: float,
    p: float = 1.0,
) -> float:
    total = (
        data_residual
        + alpha_momentum * momentum_residual
        + alpha_divergence * divergence_residual
        + alpha_energy * energy_budget_residual
        + alpha_highk * high_k_residual
        + delta_bound
    )
    return float(Cstab * max(total, 0.0) ** p)


def evaluate_candidates(
    reference: SimulationResult,
    baseline: SimulationResult,
    candidates: List[SimulationResult],
    grid: Grid3D,
    sensors: np.ndarray,
    y: np.ndarray,
    delta_normalized: float,
    Cstab: float,
    sigma_min: float,
    observability_cond: float,
    eps_safe: float,
    alpha_momentum: float,
    alpha_divergence: float,
    alpha_energy: float,
    alpha_highk: float,
) -> pd.DataFrame:
    y_scale = float(np.sqrt(np.mean(y * y)) + 1e-14)
    rhs_ref_scale = momentum_residual_bdf1(reference.u_prev, reference.u_final, grid, reference.nu, reference.dt)
    rhs_ref_scale = max(rhs_ref_scale, rms_vector_field(ifft_vec(ns_rhs_hat(fft_vec(reference.u_final), grid, reference.nu))), 1e-12)
    div_scale = max(rms_vector_field(reference.u_final) / grid.L, 1e-12)
    energy_def_ref = energy_budget_defect(reference.u_prev, reference.u_final, grid, reference.nu, reference.dt)
    diss_ref = max(dissipation_rate(reference.u_final, grid, reference.nu), energy_def_ref, 1e-12)

    raw: List[Dict[str, float]] = []
    baseline_radius: Optional[float] = None

    for sim in candidates:
        data_norm = float(np.sqrt(np.mean((observe(sim.u_final, sensors) - y) ** 2)) / y_scale)
        mom_norm = momentum_residual_bdf1(sim.u_prev, sim.u_final, grid, sim.nu, sim.dt) / rhs_ref_scale
        div = divergence_field(sim.u_final, grid)
        div_l2 = float(np.sqrt(np.mean(div * div)))
        div_max = float(np.max(np.abs(div)))
        div_norm = div_l2 / div_scale
        e_def_norm = energy_budget_defect(sim.u_prev, sim.u_final, grid, sim.nu, sim.dt) / diss_ref
        highk = high_k_energy_ratio(sim.u_final, grid)
        radius = compute_operational_radius(
            Cstab=Cstab,
            data_residual=data_norm,
            momentum_residual=mom_norm,
            divergence_residual=div_norm,
            energy_budget_residual=e_def_norm,
            high_k_residual=highk,
            delta_bound=delta_normalized,
            alpha_momentum=alpha_momentum,
            alpha_divergence=alpha_divergence,
            alpha_energy=alpha_energy,
            alpha_highk=alpha_highk,
        )
        if sim.name == baseline.name:
            baseline_radius = radius

        e_final = kinetic_energy(sim.u_final)
        e_ref = kinetic_energy(reference.u_final)
        raw.append({
            "candidate": sim.name,
            "model_kind": sim.model_kind,
            "accepted_by_noharm": None,
            "safe_choice": None,
            "radius": radius,
            "baseline_radius": np.nan,
            "certificate_ratio": np.nan,
            "final_rel_l2_error": rel_l2(sim.u_final, reference.u_final),
            "sensor_rmse_normalized": data_norm,
            "momentum_residual_normalized": mom_norm,
            "divergence_l2": div_l2,
            "divergence_max": div_max,
            "energy_budget_defect_normalized": e_def_norm,
            "high_k_energy_ratio": highk,
            "kinetic_energy_final": e_final,
            "energy_error_percent": 100.0 * (e_final - e_ref) / max(abs(e_ref), 1e-14),
            "enstrophy_final": enstrophy(sim.u_final, grid),
            "dissipation_final": dissipation_rate(sim.u_final, grid, sim.nu),
            "max_cfl": float(sim.diagnostics["max_cfl"].max()),
            "wall_seconds": sim.wall_seconds,
            "C_stab": Cstab,
            "sigma_min_observability": sigma_min,
            "observability_condition_number": observability_cond,
            "notes": sim.notes,
        })

    if baseline_radius is None:
        raise RuntimeError("Baseline radius was not computed. Baseline must be included in candidates.")

    # Apply the no-harm rule.
    safe_choice = baseline.name
    best_radius = baseline_radius
    for row in raw:
        row["baseline_radius"] = baseline_radius
        row["certificate_ratio"] = row["radius"] / max(baseline_radius, 1e-14)
        if row["candidate"] == baseline.name:
            row["accepted_by_noharm"] = True
            row["safe_choice"] = baseline.name
        else:
            accepted = bool(row["radius"] <= baseline_radius + eps_safe)
            row["accepted_by_noharm"] = accepted
            row["safe_choice"] = row["candidate"] if accepted else baseline.name
            if accepted and row["radius"] < best_radius:
                best_radius = row["radius"]
                safe_choice = row["candidate"]

    for row in raw:
        row["safe_choice"] = safe_choice if row["candidate"] != "reference_dns" else "reference_dns"

    # Reference is truth, not a replacement candidate; keep it clearly marked.
    for row in raw:
        if row["candidate"] == reference.name:
            row["accepted_by_noharm"] = True
            row["safe_choice"] = "reference_dns"

    return pd.DataFrame(raw)


# =============================================================================
# Plotting
# =============================================================================

def plot_energy_enstrophy(timeseries: pd.DataFrame, fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for name, g in timeseries.groupby("candidate"):
        if "overfit" in name:
            continue
        ax.plot(g["time"], g["kinetic_energy"], marker="o", markersize=2, label=name)
    ax.set_xlabel("time")
    ax.set_ylabel("kinetic energy")
    ax.set_title("3D Navier--Stokes: kinetic-energy decay")
    ax.legend(fontsize=7)
    savefig(fig, fig_dir, "ns3d_energy_enstrophy_decay")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for name, g in timeseries.groupby("candidate"):
        if "overfit" in name:
            continue
        ax.plot(g["time"], g["enstrophy"], marker="o", markersize=2, label=name)
    ax.set_xlabel("time")
    ax.set_ylabel("enstrophy")
    ax.set_title("3D Navier--Stokes: enstrophy evolution")
    ax.legend(fontsize=7)
    savefig(fig, fig_dir, "ns3d_enstrophy_evolution")


def plot_error_vs_radius(summary: pd.DataFrame, fig_dir: str) -> None:
    sub = summary[summary["candidate"] != "reference_dns"].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.scatter(sub["final_rel_l2_error"], sub["radius"], s=80)
    for _, r in sub.iterrows():
        ax.annotate(r["candidate"], (r["final_rel_l2_error"], r["radius"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("final relative L2 error against reference DNS")
    ax.set_ylabel("operational no-harm certificate radius")
    ax.set_title("No-harm NS3D: error versus certificate radius")
    ax.set_yscale("log")
    savefig(fig, fig_dir, "ns3d_error_vs_certificate_radius")


def plot_industrial_kpis(summary: pd.DataFrame, fig_dir: str) -> None:
    sub = summary[summary["candidate"] != "reference_dns"].copy()
    metrics = [
        "sensor_rmse_normalized",
        "momentum_residual_normalized",
        "energy_budget_defect_normalized",
        "high_k_energy_ratio",
        "final_rel_l2_error",
    ]
    labels = [
        "sensor RMSE",
        "momentum residual",
        "energy-budget defect",
        "high-k energy",
        "DNS error",
    ]

    x = np.arange(len(sub))
    width = 0.14
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for j, (metric, label) in enumerate(zip(metrics, labels)):
        vals = sub[metric].to_numpy(dtype=float)
        ax.bar(x + (j - 2) * width, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["candidate"], rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("normalized KPI value, log scale")
    ax.set_title("Industrial no-harm KPIs for 3D Navier--Stokes candidates")
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, fig_dir, "ns3d_industrial_kpis")


def velocity_magnitude(u: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(u * u, axis=0))


def vorticity_magnitude(u: np.ndarray, grid: Grid3D) -> np.ndarray:
    w = vorticity_field(u, grid)
    return np.sqrt(np.sum(w * w, axis=0))


def plot_slice_panel(
    sims: Dict[str, SimulationResult],
    grid: Grid3D,
    fig_dir: str,
    safe_choice: str,
) -> None:
    names = ["reference_dns", "baseline_robust_dns", "learned_calibrated_dns", safe_choice]
    unique_names: List[str] = []
    for n in names:
        if n in sims and n not in unique_names:
            unique_names.append(n)
    k = grid.N // 2

    fields = [velocity_magnitude(sims[n].u_final)[:, :, k] for n in unique_names]
    vmin = min(float(np.min(f)) for f in fields)
    vmax = max(float(np.max(f)) for f in fields)
    fig, axs = plt.subplots(1, len(unique_names), figsize=(3.3 * len(unique_names), 3.2))
    if len(unique_names) == 1:
        axs = [axs]
    for ax, n, f in zip(axs, unique_names, fields):
        im = ax.imshow(f.T, origin="lower", extent=[0, grid.L, 0, grid.L], vmin=vmin, vmax=vmax)
        ax.set_title(n)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axs, shrink=0.8)
    savefig(fig, fig_dir, "ns3d_velocity_slice_panel")

    vort_fields = [vorticity_magnitude(sims[n].u_final, grid)[:, :, k] for n in unique_names]
    vmin = min(float(np.min(f)) for f in vort_fields)
    vmax = max(float(np.max(f)) for f in vort_fields)
    fig, axs = plt.subplots(1, len(unique_names), figsize=(3.3 * len(unique_names), 3.2))
    if len(unique_names) == 1:
        axs = [axs]
    for ax, n, f in zip(axs, unique_names, vort_fields):
        im = ax.imshow(f.T, origin="lower", extent=[0, grid.L, 0, grid.L], vmin=vmin, vmax=vmax)
        ax.set_title(n)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axs, shrink=0.8)
    savefig(fig, fig_dir, "ns3d_vorticity_slice_panel")


def plot_energy_spectra(sims: Dict[str, SimulationResult], grid: Grid3D, fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for name, sim in sims.items():
        if name not in {"reference_dns", "baseline_robust_dns", "learned_calibrated_dns", "learned_aggressive_dns", "learned_sensor_overfit_field"}:
            continue
        spec = energy_spectrum(sim.u_final, grid)
        spec = spec[spec["energy"] > 0]
        if len(spec) == 0:
            continue
        ax.loglog(spec["k"] + 1e-12, spec["energy"], marker="o", markersize=3, label=name)
    ax.set_xlabel("wavenumber shell k")
    ax.set_ylabel("spectral kinetic energy")
    ax.set_title("Final 3D kinetic-energy spectra")
    ax.legend(fontsize=7)
    savefig(fig, fig_dir, "ns3d_energy_spectra_final")


# =============================================================================
# CLI and main workflow
# =============================================================================

def apply_mode_presets(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "smoke":
        args.N = args.N or 16
        args.T = args.T or 0.040
        args.dt = args.dt or 0.0020
        args.n_sensors = args.n_sensors or 48
        args.stability_basis = args.stability_basis or 32
        args.diagnostics_stride = args.diagnostics_stride or 2
    elif args.mode == "quick":
        args.N = args.N or 24
        args.T = args.T or 0.080
        args.dt = args.dt or 0.0015
        args.n_sensors = args.n_sensors or 96
        args.stability_basis = args.stability_basis or 64
        args.diagnostics_stride = args.diagnostics_stride or 5
    elif args.mode == "full":
        args.N = args.N or 48
        args.T = args.T or 0.160
        args.dt = args.dt or 0.0010
        args.n_sensors = args.n_sensors or 192
        args.stability_basis = args.stability_basis or 96
        args.diagnostics_stride = args.diagnostics_stride or 10
    else:
        args.N = args.N or 24
        args.T = args.T or 0.080
        args.dt = args.dt or 0.0015
        args.n_sensors = args.n_sensors or 96
        args.stability_basis = args.stability_basis or 64
        args.diagnostics_stride = args.diagnostics_stride or 5
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-harm PIIL 3D Navier--Stokes industrial validation script.")
    parser.add_argument("--mode", choices=["smoke", "quick", "full", "custom"], default="smoke")
    parser.add_argument("--out", type=str, default="results_noharm_ns3d")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--N", type=int, default=None, help="Grid points per direction.")
    parser.add_argument("--T", type=float, default=None, help="Final time.")
    parser.add_argument("--dt", type=float, default=None, help="Time step.")
    parser.add_argument("--nu", type=float, default=0.01, help="Reference viscosity.")
    parser.add_argument("--n-sensors", dest="n_sensors", type=int, default=None)
    parser.add_argument("--noise-fraction", type=float, default=0.01)
    parser.add_argument("--stability-basis", type=int, default=None)
    parser.add_argument("--diagnostics-stride", type=int, default=None)
    parser.add_argument("--eps-safe", type=float, default=0.0)
    parser.add_argument("--alpha-momentum", type=float, default=0.25)
    parser.add_argument("--alpha-divergence", type=float, default=0.10)
    parser.add_argument("--alpha-energy", type=float, default=0.20)
    parser.add_argument("--alpha-highk", type=float, default=0.15)
    parser.add_argument("--baseline-nu-factor", type=float, default=1.20)
    parser.add_argument("--baseline-amplitude", type=float, default=0.970)
    parser.add_argument("--learned-good-amplitude", type=float, default=1.005)
    parser.add_argument("--learned-aggressive-amplitude", type=float, default=1.100)
    parser.add_argument("--learned-aggressive-nu-factor", type=float, default=0.70)
    parser.add_argument("--overfit-basis", type=int, default=96)
    parser.add_argument("--overfit-ridge", type=float, default=1e-5)
    parser.add_argument("--overfit-max-correction-fraction", type=float, default=0.35)
    parser.add_argument("--perturbation", type=float, default=0.01)
    parser.add_argument("--verbose", action="store_true")
    return apply_mode_presets(parser.parse_args())


def main() -> None:
    args = parse_args()
    fig_dir, tab_dir = ensure_dirs(args.out)
    rng = np.random.default_rng(args.seed)

    grid = make_grid(args.N)
    n_steps = int(round(args.T / args.dt))
    if abs(n_steps * args.dt - args.T) > 1e-12:
        args.T = n_steps * args.dt

    print("No-harm PIIL NS3D validation")
    print(f"mode={args.mode}, N={args.N}, steps={n_steps}, T={args.T}, dt={args.dt}, nu={args.nu}")
    print(f"output={args.out}")

    # ------------------------------------------------------------------
    # Full 3D Navier--Stokes runs
    # ------------------------------------------------------------------
    reference = integrate_navier_stokes(
        name="reference_dns",
        model_kind="reference 3D DNS",
        grid=grid,
        nu=args.nu,
        amplitude=1.0,
        T=args.T,
        dt=args.dt,
        perturbation=args.perturbation,
        diagnostics_stride=args.diagnostics_stride,
        notes="Reference pseudo-spectral 3D Navier--Stokes run.",
        verbose=args.verbose,
    )

    baseline = integrate_navier_stokes(
        name="baseline_robust_dns",
        model_kind="robust conservative DNS baseline",
        grid=grid,
        nu=args.nu * args.baseline_nu_factor,
        amplitude=args.baseline_amplitude,
        T=args.T,
        dt=args.dt,
        perturbation=args.perturbation,
        diagnostics_stride=args.diagnostics_stride,
        notes="Conservative baseline: slightly dissipative and amplitude-biased but fully PDE-integrated.",
        verbose=args.verbose,
    )

    learned_good = integrate_navier_stokes(
        name="learned_calibrated_dns",
        model_kind="calibrated learned-parameter DNS",
        grid=grid,
        nu=args.nu,
        amplitude=args.learned_good_amplitude,
        T=args.T,
        dt=args.dt,
        perturbation=args.perturbation,
        diagnostics_stride=args.diagnostics_stride,
        notes="Learned/calibrated candidate close to the reference physical parameters; fully PDE-integrated.",
        verbose=args.verbose,
    )

    learned_aggressive = integrate_navier_stokes(
        name="learned_aggressive_dns",
        model_kind="aggressive learned extrapolation DNS",
        grid=grid,
        nu=args.nu * args.learned_aggressive_nu_factor,
        amplitude=args.learned_aggressive_amplitude,
        T=args.T,
        dt=args.dt,
        perturbation=args.perturbation,
        diagnostics_stride=args.diagnostics_stride,
        notes="Aggressive learned extrapolation: lower viscosity and higher initial energy; fully PDE-integrated but risky.",
        verbose=args.verbose,
    )

    # ------------------------------------------------------------------
    # Sensors and overfit stress-test field
    # ------------------------------------------------------------------
    sensors = choose_sensors(grid, args.n_sensors, rng)
    y_clean = observe(reference.u_final, sensors)
    y_scale = float(np.sqrt(np.mean(y_clean * y_clean)) + 1e-14)
    noise = args.noise_fraction * y_scale * rng.normal(size=y_clean.shape)
    y = y_clean + noise
    delta_normalized = float(np.sqrt(np.mean(noise * noise)) / y_scale)

    Cstab, sigma_min, obs_cond = randomized_observability_constant(
        grid=grid,
        sensors=sensors,
        rng=rng,
        n_basis=args.stability_basis,
        k_low=1.0,
        k_high=max(2.0, args.N / 5.0),
    )

    overfit = fit_sensor_overfit_field(
        base=baseline,
        y=y,
        sensors=sensors,
        grid=grid,
        rng=rng,
        n_basis=args.overfit_basis,
        ridge=args.overfit_ridge,
        max_correction_fraction=args.overfit_max_correction_fraction,
    )

    sims = {
        s.name: s
        for s in [reference, baseline, learned_good, learned_aggressive, overfit]
    }

    # ------------------------------------------------------------------
    # No-harm evaluation
    # ------------------------------------------------------------------
    summary = evaluate_candidates(
        reference=reference,
        baseline=baseline,
        candidates=[reference, baseline, learned_good, learned_aggressive, overfit],
        grid=grid,
        sensors=sensors,
        y=y,
        delta_normalized=delta_normalized,
        Cstab=Cstab,
        sigma_min=sigma_min,
        observability_cond=obs_cond,
        eps_safe=args.eps_safe,
        alpha_momentum=args.alpha_momentum,
        alpha_divergence=args.alpha_divergence,
        alpha_energy=args.alpha_energy,
        alpha_highk=args.alpha_highk,
    )

    noharm = summary[summary["candidate"] != "reference_dns"].copy()
    safe_choice = str(noharm.sort_values("radius").iloc[0]["candidate"])
    # However, the rule is fallback against baseline; among accepted choices, choose the smallest radius.
    accepted = noharm[noharm["accepted_by_noharm"] == True].copy()  # noqa: E712
    if not accepted.empty:
        safe_choice = str(accepted.sort_values("radius").iloc[0]["candidate"])
    else:
        safe_choice = "baseline_robust_dns"
    summary.loc[summary["candidate"] != "reference_dns", "safe_choice"] = safe_choice

    # ------------------------------------------------------------------
    # Save tables and metadata
    # ------------------------------------------------------------------
    timeseries = pd.concat([sim.diagnostics for sim in sims.values()], ignore_index=True)
    summary.to_csv(os.path.join(tab_dir, "ns3d_candidate_summary.csv"), index=False)
    noharm.to_csv(os.path.join(tab_dir, "ns3d_noharm_decisions.csv"), index=False)
    timeseries.to_csv(os.path.join(tab_dir, "ns3d_timeseries.csv"), index=False)

    metadata = {
        "script": "no_harm_piil_ns3d_industrial.py",
        "mode": args.mode,
        "N": args.N,
        "grid_points_total": args.N ** 3,
        "velocity_degrees_of_freedom": 3 * args.N ** 3,
        "L": grid.L,
        "dx": grid.dx,
        "T": args.T,
        "dt": args.dt,
        "n_steps": n_steps,
        "nu_reference": args.nu,
        "n_sensors": args.n_sensors,
        "sensor_components": int(3 * args.n_sensors),
        "noise_fraction": args.noise_fraction,
        "delta_normalized": delta_normalized,
        "C_stab_randomized_observability": Cstab,
        "sigma_min_observability": sigma_min,
        "observability_condition_number": obs_cond,
        "safe_choice": safe_choice,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wall_seconds_total_solver_only": float(sum(sim.wall_seconds for sim in sims.values())),
        "candidate_notes": {name: sim.notes for name, sim in sims.items()},
    }
    with open(os.path.join(tab_dir, "ns3d_run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_energy_enstrophy(timeseries, fig_dir)
    plot_error_vs_radius(summary, fig_dir)
    plot_industrial_kpis(summary, fig_dir)
    plot_slice_panel(sims, grid, fig_dir, safe_choice=safe_choice)
    plot_energy_spectra(sims, grid, fig_dir)

    print("Done.")
    print(f"Safe no-harm choice: {safe_choice}")
    print(f"Tables:  {tab_dir}")
    print(f"Figures: {fig_dir}")
    print("Main table: ns3d_candidate_summary.csv")


if __name__ == "__main__":
    main()
