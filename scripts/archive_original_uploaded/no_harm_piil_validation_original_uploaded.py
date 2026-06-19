#!/usr/bin/env python3
"""
No-harm physics-informed inverse learning validation suite.

This script produces reproducible numerical evidence for the manuscript idea:
    reconstruction + residual certificate + no-harm decision + uncertainty radius.

The experiments are deliberately finite-dimensional and reproducible. They do not claim
that a neural network is required to demonstrate the certificate. Instead, they validate
what the paper proves: any computed inverse reconstruction, including a physics-informed
learned one, can be checked using data, physics/model, boundary, stochastic residual, and
optimization residual information.

Experiments included
--------------------
1. Poisson inverse source recovery on [0,1].
2. Inverse heat initial-condition recovery.
3. Limited-angle tomography with a simple Radon-like projection operator.
4. 1D elliptic coefficient identification as a geophysical-style inverse problem.
5. Stochastic collocation residual certificate sweep.

Outputs
-------
results_noharm_piil/
    tables/
        summary_all_experiments.csv
        noharm_decisions.csv
        stability_constants.csv
        stochastic_residual_sweep.csv
    figures/
        *.png, *.pdf

Dependencies
------------
numpy, scipy, pandas, matplotlib

Run
---
python no_harm_piil_validation.py --out results_noharm_piil --seed 123
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy.linalg import eigh
from scipy.optimize import least_squares
from scipy import ndimage


# -----------------------------
# Global plotting configuration
# -----------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 140,
    "savefig.dpi": 250,
    "lines.linewidth": 1.8,
})


@dataclass
class CandidateResult:
    experiment: str
    scenario: str
    candidate: str
    rel_error: float
    coeff_error: float
    data_residual: float
    physics_residual: float
    boundary_residual: float
    optimization_residual: float
    delta_bound: float
    stability_constant: float
    radius: float
    covered: bool
    radius_sharpness: float
    accepted_by_noharm: Optional[bool] = None
    safe_choice: Optional[str] = None
    notes: str = ""


def ensure_dirs(out: str) -> Tuple[str, str]:
    fig_dir = os.path.join(out, "figures")
    tab_dir = os.path.join(out, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    return fig_dir, tab_dir


def savefig(fig, fig_dir: str, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def rel_norm(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + eps))


def stable_inverse_constant(F: np.ndarray, floor: float = 1e-10) -> Tuple[float, float, np.ndarray]:
    """Return C=1/sigma_min for a finite-dimensional admissible subspace."""
    s = np.linalg.svd(F, compute_uv=False)
    sigma_min = float(np.min(s))
    return 1.0 / max(sigma_min, floor), sigma_min, s


def ridge_solution(F: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n = F.shape[1]
    return np.linalg.solve(F.T @ F + lam * np.eye(n), F.T @ y)


def normalize_columns(B: np.ndarray) -> np.ndarray:
    B = B.copy()
    for j in range(B.shape[1]):
        nrm = np.linalg.norm(B[:, j])
        if nrm > 0:
            B[:, j] /= nrm
    return B


def sine_basis_1d(n: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, n + 2)[1:-1]
    B = np.column_stack([np.sin(np.pi * (j + 1) * x) for j in range(k)])
    return x, normalize_columns(B)


def fd_poisson_matrix(n: int) -> Tuple[np.ndarray, float]:
    h = 1.0 / (n + 1)
    main = 2.0 * np.ones(n) / h**2
    off = -1.0 * np.ones(n - 1) / h**2
    K = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    return K, h


def solve_poisson_source(q: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.linalg.solve(K, q)


def observation_matrix(n: int, m: int, rng: np.random.Generator, mode: str = "spread") -> np.ndarray:
    if mode == "spread":
        idx = np.linspace(0, n - 1, m, dtype=int)
    elif mode == "random":
        idx = np.sort(rng.choice(n, size=m, replace=False))
    else:
        raise ValueError("mode must be 'spread' or 'random'")
    H = np.zeros((m, n))
    H[np.arange(m), idx] = 1.0
    return H


def compute_radius(
    Cstab: float,
    p: float,
    data_res: float,
    pde_res: float,
    bc_res: float,
    delta: float,
    opt_res: float = 0.0,
    alpha_pde: float = 1.0,
    alpha_bc: float = 1.0,
    alpha_opt: float = 0.0,
) -> float:
    total = data_res + alpha_pde * pde_res + alpha_bc * bc_res + delta + alpha_opt * opt_res
    return float(Cstab * max(total, 0.0) ** p)


def add_candidate_row(
    rows: List[CandidateResult],
    experiment: str,
    scenario: str,
    candidate: str,
    q: np.ndarray,
    q_true: np.ndarray,
    c: np.ndarray,
    c_true: np.ndarray,
    data_res: float,
    pde_res: float,
    bc_res: float,
    opt_res: float,
    delta: float,
    Cstab: float,
    radius: float,
    notes: str = "",
) -> None:
    coeff_error = float(np.linalg.norm(c - c_true))
    rel_error = rel_norm(q, q_true)
    covered = coeff_error <= radius + 1e-12
    sharpness = float(radius / (coeff_error + 1e-12))
    rows.append(CandidateResult(
        experiment=experiment,
        scenario=scenario,
        candidate=candidate,
        rel_error=rel_error,
        coeff_error=coeff_error,
        data_residual=float(data_res),
        physics_residual=float(pde_res),
        boundary_residual=float(bc_res),
        optimization_residual=float(opt_res),
        delta_bound=float(delta),
        stability_constant=float(Cstab),
        radius=float(radius),
        covered=bool(covered),
        radius_sharpness=sharpness,
        notes=notes,
    ))


def apply_noharm(rows: List[CandidateResult], experiment: str, scenario: str, eps_safe: float = 0.0) -> None:
    base = [r for r in rows if r.experiment == experiment and r.scenario == scenario and r.candidate == "baseline"]
    if not base:
        return
    base_radius = base[0].radius
    candidates = [r for r in rows if r.experiment == experiment and r.scenario == scenario and r.candidate != "baseline"]
    for r in candidates:
        accepted = r.radius <= base_radius + eps_safe
        r.accepted_by_noharm = accepted
        r.safe_choice = r.candidate if accepted else "baseline"


# -----------------------------
# Experiment 1: Poisson source
# -----------------------------
def experiment_poisson(fig_dir: str, rng: np.random.Generator) -> Tuple[List[CandidateResult], Dict[str, float]]:
    exp = "poisson_source"
    n, k, m = 120, 10, 35
    x, B = sine_basis_1d(n, k)
    K, h = fd_poisson_matrix(n)
    H = observation_matrix(n, m, rng, mode="spread")
    S = np.linalg.solve(K, B)  # state basis columns from q basis
    F = H @ S
    Cstab, sigma_min, svals = stable_inverse_constant(F)

    c_true = np.array([1.2, -0.7, 0.45, 0.25, -0.22, 0.12, 0.08, -0.06, 0.04, 0.02])
    q_true = B @ c_true
    u_true = solve_poisson_source(q_true, K)
    noise_level = 0.02 * np.linalg.norm(H @ u_true) / np.sqrt(m)
    noise = noise_level * rng.normal(size=m)
    y = H @ u_true + noise
    delta = float(np.linalg.norm(noise))

    lam = 1e-5
    c_base = ridge_solution(F, y, lam)
    q_base = B @ c_base
    u_base = solve_poisson_source(q_base, K)

    # Good learned advice: close to truth, supported by data.
    c_good = c_true + 0.035 * rng.normal(size=k)
    q_good = B @ c_good
    u_good = solve_poisson_source(q_good, K)

    # Bad learned advice: structured bias in identifiable modes.
    c_bad = c_true.copy()
    c_bad[[1, 3, 5]] += np.array([0.65, -0.45, 0.35])
    q_bad = B @ c_bad
    u_bad = solve_poisson_source(q_bad, K)

    # Unfinished PINN-like output: state and source are inconsistent.
    c_unfinished = c_true + 0.06 * rng.normal(size=k)
    q_unfinished = B @ c_unfinished
    u_unfinished = solve_poisson_source(q_unfinished, K) + 0.06 * np.sin(15 * np.pi * x)

    candidates = {
        "baseline": (c_base, q_base, u_base, "classical ridge baseline"),
        "learned_good": (c_good, q_good, u_good, "learned advice close to admissible truth"),
        "learned_bad_shift": (c_bad, q_bad, u_bad, "learned advice with structured distribution shift"),
        "unfinished_pinn": (c_unfinished, q_unfinished, u_unfinished, "state-source pair with optimization/collocation failure"),
    }

    rows: List[CandidateResult] = []
    for name, (c, q, u, note) in candidates.items():
        data_res = np.linalg.norm(H @ u - y)
        pde_res = np.linalg.norm(K @ u - q) / np.sqrt(n)
        bc_res = 0.0  # interior Dirichlet solver encodes zero boundary; perturbation also zero at endpoints.
        grad = F.T @ (F @ c - y) + lam * c
        opt_res = np.linalg.norm(grad)
        radius = compute_radius(Cstab, 1.0, data_res, pde_res, bc_res, delta, opt_res, alpha_pde=0.05, alpha_opt=0.01)
        add_candidate_row(rows, exp, "default", name, q, q_true, c, c_true, data_res, pde_res, bc_res, opt_res, delta, Cstab, radius, note)

    apply_noharm(rows, exp, "default", eps_safe=0.0)

    # Figures
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, q_true, label="true source", linewidth=2.4)
    ax.plot(x, q_base, label="baseline", linestyle="--")
    ax.plot(x, q_good, label="learned good")
    ax.plot(x, q_bad, label="learned shifted")
    ax.plot(x, q_unfinished, label="unfinished PINN-like", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("q(x)")
    ax.set_title("Poisson inverse source recovery")
    ax.legend(ncol=2)
    savefig(fig, fig_dir, "poisson_source_reconstructions")

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    labels = [r.candidate for r in rows]
    errors = [r.coeff_error for r in rows]
    radii = [r.radius for r in rows]
    idx = np.arange(len(labels))
    ax.bar(idx - 0.18, errors, width=0.36, label="true coefficient error")
    ax.bar(idx + 0.18, radii, width=0.36, label="certified radius")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("coefficient-space norm")
    ax.set_title("Poisson: error versus certified radius")
    ax.legend()
    savefig(fig, fig_dir, "poisson_error_vs_radius")

    stab = {
        "experiment": exp,
        "scenario": "default",
        "sigma_min": sigma_min,
        "C_stab": Cstab,
        "n_state": n,
        "n_basis": k,
        "n_obs": m,
        "condition_number_F": float(np.max(svals) / np.min(svals)),
    }
    return rows, stab


# -----------------------------
# Experiment 2: Inverse heat
# -----------------------------
def heat_propagator(K: np.ndarray, kappa: float, T: float) -> np.ndarray:
    vals, vecs = eigh(K)
    return (vecs * np.exp(-kappa * T * vals)) @ vecs.T


def experiment_inverse_heat(fig_dir: str, rng: np.random.Generator) -> Tuple[List[CandidateResult], List[Dict[str, float]]]:
    exp = "inverse_heat_initial_condition"
    n, k, m = 120, 8, 45
    x, B = sine_basis_1d(n, k)
    K, _ = fd_poisson_matrix(n)
    H = observation_matrix(n, m, rng, mode="spread")
    kappa = 0.004
    times = [0.02, 0.08, 0.16]
    c_true = np.array([1.0, -0.55, 0.35, 0.20, -0.10, 0.08, -0.05, 0.03])
    q_true = B @ c_true

    all_rows: List[CandidateResult] = []
    stabs: List[Dict[str, float]] = []

    for T in times:
        E = heat_propagator(K, kappa, T)
        F = H @ E @ B
        Cstab, sigma_min, svals = stable_inverse_constant(F, floor=1e-12)
        u_true_T = E @ q_true
        noise_level = 0.015 * np.linalg.norm(H @ u_true_T) / np.sqrt(m)
        noise = noise_level * rng.normal(size=m)
        y = H @ u_true_T + noise
        delta = float(np.linalg.norm(noise))
        lam = 1e-4
        c_base = ridge_solution(F, y, lam)
        q_base = B @ c_base
        u_base_T = E @ q_base

        c_good = c_true + 0.04 * rng.normal(size=k)
        q_good = B @ c_good
        u_good_T = E @ q_good

        c_bad = c_true.copy()
        c_bad[2:6] += np.array([0.65, -0.50, 0.35, -0.25])
        q_bad = B @ c_bad
        u_bad_T = E @ q_bad

        # Unstable learned output with unsupported high-frequency component outside admissible basis.
        high = 0.15 * np.sin(30 * np.pi * x)
        q_hallucinated = q_good + high
        # Project to basis for coefficient error reporting.
        c_hall = B.T @ q_hallucinated
        u_hall_T = E @ q_hallucinated

        candidates = {
            "baseline": (c_base, q_base, u_base_T, "regularized heat inversion baseline"),
            "learned_good": (c_good, q_good, u_good_T, "learned initial condition close to truth"),
            "learned_hallucinated_highfreq": (c_hall, q_hallucinated, u_hall_T, "unsupported high-frequency learned structure"),
            "learned_bad_shift": (c_bad, q_bad, u_bad_T, "shifted learned coefficients"),
        }
        scenario = f"T={T:.2f}"
        for name, (c, q, uT, note) in candidates.items():
            data_res = np.linalg.norm(H @ uT - y)
            pde_res = 0.0  # candidate state obtained by discrete heat propagator
            bc_res = 0.0
            grad = F.T @ (F @ c - y) + lam * c if c.shape[0] == k else np.zeros(k)
            opt_res = np.linalg.norm(grad)
            radius = compute_radius(Cstab, 1.0, data_res, pde_res, bc_res, delta, opt_res, alpha_opt=0.005)
            add_candidate_row(all_rows, exp, scenario, name, q, q_true, c, c_true, data_res, pde_res, bc_res, opt_res, delta, Cstab, radius, note)
        apply_noharm(all_rows, exp, scenario, eps_safe=0.0)
        stabs.append({
            "experiment": exp,
            "scenario": scenario,
            "sigma_min": sigma_min,
            "C_stab": Cstab,
            "n_state": n,
            "n_basis": k,
            "n_obs": m,
            "condition_number_F": float(np.max(svals) / np.min(svals)),
            "T": T,
            "kappa": kappa,
        })

    # Figure: Cstab grows with final time.
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot([s["T"] for s in stabs], [s["C_stab"] for s in stabs], marker="o")
    ax.set_xlabel("final observation time T")
    ax.set_ylabel("conditional stability constant")
    ax.set_yscale("log")
    ax.set_title("Inverse heat: stability constant grows with ill-posedness")
    savefig(fig, fig_dir, "heat_stability_constant_vs_time")

    # Figure: reconstructions at hardest time
    hard = "T=0.16"
    rows_hard = [r for r in all_rows if r.scenario == hard]
    # Recompute arrays for plotting hard time
    Ehard = heat_propagator(K, kappa, 0.16)
    Fhard = H @ Ehard @ B
    y_dummy = None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, q_true, label="true initial condition", linewidth=2.4)
    for r in rows_hard:
        # reconstruct q from stored candidate name by regenerating not available here; skip exact plot from rows
        pass
    # Instead create representative arrays again for hard time.
    savefig(fig, fig_dir, "heat_placeholder_true_initial_condition")
    plt.close('all')

    return all_rows, stabs


# -----------------------------
# Experiment 3: Limited-angle tomography
# -----------------------------
def dct2_basis(N: int, kx: int, ky: int) -> np.ndarray:
    xs = np.arange(N)
    ys = np.arange(N)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    basis = []
    for i in range(kx):
        for j in range(ky):
            phi = np.cos(np.pi * i * (X + 0.5) / N) * np.cos(np.pi * j * (Y + 0.5) / N)
            phi = phi - np.mean(phi) if (i, j) != (0, 0) else phi
            nrm = np.linalg.norm(phi)
            if nrm > 0:
                phi = phi / nrm
            basis.append(phi.reshape(-1))
    return np.column_stack(basis)


def radon_like(img: np.ndarray, angles: np.ndarray) -> np.ndarray:
    projections = []
    for a in angles:
        rot = ndimage.rotate(img, angle=float(a), reshape=False, order=1, mode="constant", cval=0.0)
        projections.append(np.sum(rot, axis=0))
    return np.concatenate(projections)


def build_tomo_operator(B: np.ndarray, N: int, angles: np.ndarray) -> np.ndarray:
    cols = []
    for j in range(B.shape[1]):
        img = B[:, j].reshape(N, N)
        cols.append(radon_like(img, angles))
    return np.column_stack(cols)


def experiment_tomography(fig_dir: str, rng: np.random.Generator) -> Tuple[List[CandidateResult], Dict[str, float]]:
    exp = "limited_angle_tomography"
    N = 28
    B = dct2_basis(N, 6, 6)
    k = B.shape[1]
    angles = np.linspace(-50, 50, 15)
    F = build_tomo_operator(B, N, angles)
    Cstab, sigma_min, svals = stable_inverse_constant(F, floor=1e-8)

    c_true = np.zeros(k)
    # structured low-frequency phantom
    c_true[0] = 6.0
    c_true[1] = -2.0
    c_true[6] = 1.8
    c_true[7] = 1.2
    c_true[14] = -1.0
    c_true[21] = 0.8
    img_true = (B @ c_true).reshape(N, N)
    # add two Gaussian blobs while keeping coefficient projection for admissible part
    xx = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(xx, xx, indexing="ij")
    blob = 2.0 * np.exp(-((X - 0.35) ** 2 + (Y + 0.2) ** 2) / 0.05) - 1.3 * np.exp(-((X + 0.25) ** 2 + (Y - 0.25) ** 2) / 0.03)
    img_true = img_true + blob
    c_true = B.T @ img_true.reshape(-1)
    img_true_adm = (B @ c_true).reshape(N, N)
    y_clean = F @ c_true
    noise_level = 0.01 * np.linalg.norm(y_clean) / np.sqrt(y_clean.size)
    noise = noise_level * rng.normal(size=y_clean.size)
    y = y_clean + noise
    delta = float(np.linalg.norm(noise))
    lam = 1e-3
    c_base = ridge_solution(F, y, lam)
    img_base = (B @ c_base).reshape(N, N)

    c_good = c_true + 0.03 * rng.normal(size=k)
    img_good = (B @ c_good).reshape(N, N)

    c_bad = c_true.copy()
    c_bad[3:12:2] += np.array([2.0, -1.8, 1.4, -1.1, 0.8])
    img_bad = (B @ c_bad).reshape(N, N)

    candidates = {
        "baseline": (c_base, img_base, "ridge reconstruction from limited angles"),
        "learned_good": (c_good, img_good, "learned reconstruction close to admissible phantom"),
        "learned_hallucinated": (c_bad, img_bad, "plausible but unsupported limited-angle hallucination"),
    }

    rows: List[CandidateResult] = []
    q_true_vec = img_true_adm.reshape(-1)
    for name, (c, img, note) in candidates.items():
        data_res = np.linalg.norm(F @ c - y)
        pde_res = 0.0
        bc_res = 0.0
        grad = F.T @ (F @ c - y) + lam * c
        opt_res = np.linalg.norm(grad)
        radius = compute_radius(Cstab, 1.0, data_res, pde_res, bc_res, delta, opt_res, alpha_opt=0.001)
        add_candidate_row(rows, exp, "default", name, img.reshape(-1), q_true_vec, c, c_true, data_res, pde_res, bc_res, opt_res, delta, Cstab, radius, note)
    apply_noharm(rows, exp, "default", eps_safe=0.0)

    # 2D reconstruction panel
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    imgs = [img_true_adm, img_base, img_good, img_bad]
    titles = ["truth", "baseline", "learned good", "learned hallucinated"]
    vmin = min(np.min(im) for im in imgs)
    vmax = max(np.max(im) for im in imgs)
    for ax, im, title in zip(axs, imgs, titles):
        imh = ax.imshow(im, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(imh, ax=axs, shrink=0.75)
    savefig(fig, fig_dir, "tomography_reconstruction_panel")

    # 3D surface plot for truth and safe choice
    safe = "baseline"
    for r in rows:
        if r.candidate != "baseline" and r.accepted_by_noharm:
            safe = r.candidate
            break
    safe_img = {"baseline": img_base, "learned_good": img_good, "learned_hallucinated": img_bad}.get(safe, img_base)
    Xgrid, Ygrid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    fig = plt.figure(figsize=(10, 4.2))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(Xgrid, Ygrid, img_true_adm, linewidth=0, antialiased=True, cmap="viridis")
    ax1.set_title("true admissible phantom")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(Xgrid, Ygrid, safe_img, linewidth=0, antialiased=True, cmap="viridis")
    ax2.set_title(f"safe output: {safe}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    savefig(fig, fig_dir, "tomography_3d_surface_truth_vs_safe")

    stab = {
        "experiment": exp,
        "scenario": "default",
        "sigma_min": sigma_min,
        "C_stab": Cstab,
        "n_pixels": N * N,
        "n_basis": k,
        "n_measurements": int(F.shape[0]),
        "condition_number_F": float(np.max(svals) / np.min(svals)),
        "n_angles": len(angles),
    }
    return rows, stab


# -----------------------------
# Experiment 4: 1D coefficient identification
# -----------------------------
def coeff_basis_1d(n: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    return sine_basis_1d(n, k)


def elliptic_matrix_from_a(a: np.ndarray) -> np.ndarray:
    """Finite-volume matrix for -(a u')'=f on interior grid with zero Dirichlet BC."""
    n = len(a)
    h = 1.0 / (n + 1)
    # face coefficients: harmonic/average between cells, boundary faces use adjacent cell
    af = np.zeros(n + 1)
    af[0] = a[0]
    af[-1] = a[-1]
    af[1:-1] = 0.5 * (a[:-1] + a[1:])
    main = (af[:-1] + af[1:]) / h**2
    off_lower = -af[1:-1] / h**2
    A = np.diag(main) + np.diag(off_lower, -1) + np.diag(off_lower, 1)
    return A


def solve_elliptic_coeff(c: np.ndarray, B: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    loga = B @ c
    a = np.exp(loga)
    A = elliptic_matrix_from_a(a)
    u = np.linalg.solve(A, f)
    return a, u


def experiment_geophysical_coeff(fig_dir: str, rng: np.random.Generator) -> Tuple[List[CandidateResult], Dict[str, float]]:
    exp = "geophysical_elliptic_coefficient"
    n, k, m = 90, 6, 30
    x, B = coeff_basis_1d(n, k)
    H = observation_matrix(n, m, rng, mode="spread")
    f = 1.0 + 0.5 * np.sin(2 * np.pi * x)
    c_true = np.array([0.15, -0.35, 0.22, 0.10, -0.08, 0.04])
    a_true, u_true = solve_elliptic_coeff(c_true, B, f)
    noise_level = 0.01 * np.linalg.norm(H @ u_true) / np.sqrt(m)
    noise = noise_level * rng.normal(size=m)
    y = H @ u_true + noise
    delta = float(np.linalg.norm(noise))

    def residual_for_c(c: np.ndarray, lam: float = 1e-3) -> np.ndarray:
        _, u = solve_elliptic_coeff(c, B, f)
        return np.concatenate([H @ u - y, np.sqrt(lam) * c])

    # Baseline: conservative least-squares from zero.
    res = least_squares(lambda cc: residual_for_c(cc, lam=5e-3), x0=np.zeros(k), max_nfev=300, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    c_base = res.x
    a_base, u_base = solve_elliptic_coeff(c_base, B, f)

    # Local stability constant from finite-difference Jacobian at truth.
    eps_fd = 1e-5
    J = np.zeros((m, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = eps_fd
        _, up = solve_elliptic_coeff(c_true + e, B, f)
        _, um = solve_elliptic_coeff(c_true - e, B, f)
        J[:, j] = H @ (up - um) / (2 * eps_fd)
    Cstab, sigma_min, svals = stable_inverse_constant(J, floor=1e-9)

    c_good = c_true + 0.025 * rng.normal(size=k)
    a_good, u_good = solve_elliptic_coeff(c_good, B, f)

    c_bad = c_true + np.array([0.40, -0.25, 0.20, -0.18, 0.10, -0.08])
    a_bad, u_bad = solve_elliptic_coeff(c_bad, B, f)

    candidates = {
        "baseline": (c_base, a_base, u_base, "PDE-constrained least-squares baseline"),
        "learned_good": (c_good, a_good, u_good, "learned coefficient close to truth"),
        "learned_bad_shift": (c_bad, a_bad, u_bad, "shifted coefficient field"),
    }

    rows: List[CandidateResult] = []
    for name, (c, a, u, note) in candidates.items():
        data_res = np.linalg.norm(H @ u - y)
        # PDE residual is zero because u solves the discrete PDE for a. We still record model residual via data.
        pde_res = np.linalg.norm(elliptic_matrix_from_a(a) @ u - f) / np.sqrt(n)
        bc_res = 0.0
        # Approximate optimization residual for nonlinear least squares
        r0 = residual_for_c(c, lam=5e-3)
        Jc = np.zeros((len(r0), k))
        for j in range(k):
            e = np.zeros(k)
            e[j] = eps_fd
            Jc[:, j] = (residual_for_c(c + e, lam=5e-3) - residual_for_c(c - e, lam=5e-3)) / (2 * eps_fd)
        opt_res = np.linalg.norm(Jc.T @ r0)
        radius = compute_radius(Cstab, 1.0, data_res, pde_res, bc_res, delta, opt_res, alpha_pde=0.01, alpha_opt=0.001)
        add_candidate_row(rows, exp, "default", name, a, a_true, c, c_true, data_res, pde_res, bc_res, opt_res, delta, Cstab, radius, note)
    apply_noharm(rows, exp, "default", eps_safe=0.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, a_true, label="true coefficient", linewidth=2.4)
    ax.plot(x, a_base, label="baseline", linestyle="--")
    ax.plot(x, a_good, label="learned good")
    ax.plot(x, a_bad, label="learned shifted")
    ax.set_xlabel("x")
    ax.set_ylabel("a(x)")
    ax.set_title("Geophysical-style elliptic coefficient identification")
    ax.legend()
    savefig(fig, fig_dir, "geophysical_coefficient_reconstructions")

    stab = {
        "experiment": exp,
        "scenario": "default",
        "sigma_min": sigma_min,
        "C_stab": Cstab,
        "n_state": n,
        "n_basis": k,
        "n_obs": m,
        "condition_number_J": float(np.max(svals) / np.min(svals)),
    }
    return rows, stab


# -----------------------------
# Experiment 5: Stochastic collocation residual sweep
# -----------------------------
def experiment_stochastic_residual(fig_dir: str, rng: np.random.Generator) -> pd.DataFrame:
    n = 2000
    x = np.linspace(0, 1, n)
    # A synthetic residual field with localized spikes and oscillations.
    residual = 0.3 * np.sin(6 * np.pi * x) + 1.5 * np.exp(-((x - 0.72) ** 2) / 0.002) + 0.8 * np.exp(-((x - 0.22) ** 2) / 0.0008)
    z = residual**2
    true_mean = float(np.mean(z))
    B = float(np.max(z))
    zeta = 0.05
    Ms = [20, 50, 100, 200, 500, 1000, 2000]
    reps = 250
    rows = []
    for M in Ms:
        cover = 0
        estimates = []
        bounds = []
        for r in range(reps):
            idx = rng.integers(0, n, size=M)
            emp = float(np.mean(z[idx]))
            hp = emp + B * np.sqrt(np.log(1.0 / zeta) / (2.0 * M))
            estimates.append(emp)
            bounds.append(hp)
            cover += int(true_mean <= hp + 1e-14)
        rows.append({
            "M": M,
            "repetitions": reps,
            "true_mean_squared_residual": true_mean,
            "mean_empirical_squared_residual": float(np.mean(estimates)),
            "mean_high_probability_upper_bound": float(np.mean(bounds)),
            "coverage_rate": cover / reps,
            "zeta": zeta,
            "envelope_B": B,
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(df["M"], df["mean_empirical_squared_residual"], marker="o", label="empirical residual")
    ax.plot(df["M"], df["mean_high_probability_upper_bound"], marker="o", label="high-probability bound")
    ax.axhline(true_mean, linestyle="--", label="true residual")
    ax.set_xscale("log")
    ax.set_xlabel("validation collocation points M")
    ax.set_ylabel("squared residual")
    ax.set_title("Stochastic residual certification improves with validation sampling")
    ax.legend()
    savefig(fig, fig_dir, "stochastic_residual_sweep")

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax.plot(x, residual, linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("residual field")
    ax.set_title("Synthetic physics residual field for collocation validation")
    savefig(fig, fig_dir, "stochastic_residual_field")
    return df


# -----------------------------
# Aggregate plots
# -----------------------------
def aggregate_plots(summary: pd.DataFrame, fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for exp, g in summary.groupby("experiment"):
        ax.scatter(g["coeff_error"], g["radius"], s=60, label=exp, alpha=0.85)
    maxv = max(summary["coeff_error"].max(), summary["radius"].max()) * 1.05
    ax.plot([0, maxv], [0, maxv], linestyle="--", color="black", linewidth=1.0, label="radius = error")
    ax.set_xlabel("true coefficient/parameter error")
    ax.set_ylabel("certified radius")
    ax.set_title("A posteriori certificate: error versus radius")
    ax.legend(fontsize=7)
    savefig(fig, fig_dir, "aggregate_error_vs_radius")

    # No-harm decisions
    dec = summary[summary["candidate"] != "baseline"].copy()
    if not dec.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        labels = [f"{r.experiment}\n{r.scenario}\n{r.candidate}" for _, r in dec.iterrows()]
        colors = ["tab:green" if bool(r.accepted_by_noharm) else "tab:red" for _, r in dec.iterrows()]
        ax.bar(np.arange(len(dec)), dec["radius"], color=colors, alpha=0.8)
        ax.set_xticks(np.arange(len(dec)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        ax.set_ylabel("certified radius")
        ax.set_title("No-harm decisions: green accepted, red rejected")
        savefig(fig, fig_dir, "aggregate_noharm_decisions")

    # Coverage by experiment
    cov = summary.groupby("experiment")["covered"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(cov["experiment"], cov["covered"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage rate")
    ax.set_title("Certificate coverage by experiment")
    ax.set_xticklabels(cov["experiment"], rotation=25, ha="right")
    savefig(fig, fig_dir, "aggregate_coverage_by_experiment")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results_noharm_piil", help="output directory")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    args = parser.parse_args()

    fig_dir, tab_dir = ensure_dirs(args.out)
    rng = np.random.default_rng(args.seed)

    all_rows: List[CandidateResult] = []
    stability_rows: List[Dict[str, float]] = []

    rows, stab = experiment_poisson(fig_dir, rng)
    all_rows.extend(rows)
    stability_rows.append(stab)

    rows, stabs = experiment_inverse_heat(fig_dir, rng)
    all_rows.extend(rows)
    stability_rows.extend(stabs)

    rows, stab = experiment_tomography(fig_dir, rng)
    all_rows.extend(rows)
    stability_rows.append(stab)

    rows, stab = experiment_geophysical_coeff(fig_dir, rng)
    all_rows.extend(rows)
    stability_rows.append(stab)

    stoch_df = experiment_stochastic_residual(fig_dir, rng)

    summary = pd.DataFrame([r.__dict__ for r in all_rows])
    summary.to_csv(os.path.join(tab_dir, "summary_all_experiments.csv"), index=False)

    noharm = summary[summary["candidate"] != "baseline"].copy()
    noharm.to_csv(os.path.join(tab_dir, "noharm_decisions.csv"), index=False)

    pd.DataFrame(stability_rows).to_csv(os.path.join(tab_dir, "stability_constants.csv"), index=False)
    stoch_df.to_csv(os.path.join(tab_dir, "stochastic_residual_sweep.csv"), index=False)

    aggregate_plots(summary, fig_dir)

    print("Done.")
    print(f"Figures saved to: {fig_dir}")
    print(f"Tables saved to:  {tab_dir}")
    print("Main table: summary_all_experiments.csv")


if __name__ == "__main__":
    main()
