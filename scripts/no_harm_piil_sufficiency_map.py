#!/usr/bin/env python3
"""
No-harm PIIL sufficiency-map validation with clean figure labels.

Purpose
-------
This script complements the main validation suite. The main validation suite
demonstrates the no-harm certificate on representative inverse problems. This
script maps the operational threshold:

    A learned physics-informed inverse reconstruction is sufficient to replace
    a baseline only when its residual-calibrated certificate is no worse than
    the baseline certificate.

In symbols, the learned reconstruction is selected only when

    R_learn <= R_base + eps_safe.

The script produces:
    1. A certificate-boundary scatter plot.
    2. Acceptance-rate heatmaps.
    3. Unsafe-acceptance and false-rejection diagnostic heatmaps.
    4. A data/noise sufficiency map.
    5. CSV tables for manuscript tables.
    6. A LaTeX table summarizing the reusable no-harm conditions.

The experiment is deliberately finite-dimensional and reproducible. It uses a
1D Poisson inverse source problem because this permits controlled variation of:
    - observation density,
    - noise level,
    - learned reconstruction quality,
    - state-source inconsistency,
    - inverse stability.

Run
---
python no_harm_piil_sufficiency_map.py --out results_noharm_map --seed 2026

Dependencies
------------
numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Plot configuration
# -----------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 140,
    "savefig.dpi": 250,
    "lines.linewidth": 1.7,
})


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class SweepRow:
    replicate: int
    n_state: int
    n_basis: int
    n_obs: int
    noise_fraction: float
    learned_coeff_scale: float
    mismatch_amplitude: float
    sigma_min: float
    C_stab: float
    condition_number: float
    delta_bound: float
    baseline_error: float
    learned_error: float
    baseline_data_residual: float
    learned_data_residual: float
    baseline_pde_residual: float
    learned_pde_residual: float
    baseline_opt_residual: float
    learned_opt_residual: float
    R_base: float
    R_learn: float
    certificate_ratio: float
    error_ratio: float
    accepted_by_noharm: bool
    learned_actually_better: bool
    unsafe_acceptance: bool
    false_rejection: bool
    safe_improvement: bool


# -----------------------------
# Utilities
# -----------------------------
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


def normalize_columns(B: np.ndarray) -> np.ndarray:
    B = B.copy()
    for j in range(B.shape[1]):
        nrm = np.linalg.norm(B[:, j])
        if nrm > 0:
            B[:, j] /= nrm
    return B


def sine_basis_1d(n: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, n + 2)[1:-1]
    B = np.column_stack([np.sin(np.pi * (j + 1) * x) for j in range(k)])
    return x, normalize_columns(B)


def fd_poisson_matrix(n: int) -> Tuple[np.ndarray, float]:
    h = 1.0 / (n + 1)
    main = 2.0 * np.ones(n) / h**2
    off = -1.0 * np.ones(n - 1) / h**2
    K = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    return K, h


def solve_poisson(q: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.linalg.solve(K, q)


def observation_matrix(n: int, m: int, rng: np.random.Generator, mode: str = "random") -> np.ndarray:
    if m > n:
        raise ValueError("m cannot exceed n.")
    if mode == "spread":
        idx = np.linspace(0, n - 1, m, dtype=int)
    elif mode == "random":
        idx = np.sort(rng.choice(n, size=m, replace=False))
    else:
        raise ValueError("mode must be 'spread' or 'random'.")
    H = np.zeros((m, n))
    H[np.arange(m), idx] = 1.0
    return H


def stable_inverse_constant(F: np.ndarray, floor: float = 1e-12) -> Tuple[float, float, np.ndarray]:
    svals = np.linalg.svd(F, compute_uv=False)
    sigma_min = float(np.min(svals))
    C_stab = 1.0 / max(sigma_min, floor)
    return C_stab, sigma_min, svals


def ridge_solution(F: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n = F.shape[1]
    return np.linalg.solve(F.T @ F + lam * np.eye(n), F.T @ y)


def compute_operational_radius(
    C_stab: float,
    data_residual: float,
    pde_residual: float,
    boundary_residual: float,
    delta_bound: float,
    opt_residual: float,
    p: float = 1.0,
    alpha_pde: float = 0.05,
    alpha_bc: float = 1.0,
    alpha_opt: float = 0.01,
) -> float:
    total = (
        data_residual
        + alpha_pde * pde_residual
        + alpha_bc * boundary_residual
        + delta_bound
        + alpha_opt * opt_residual
    )
    return float(C_stab * max(total, 0.0) ** p)


def relative_error(q: np.ndarray, q_true: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(q - q_true) / (np.linalg.norm(q_true) + eps))


def coeff_error(c: np.ndarray, c_true: np.ndarray) -> float:
    return float(np.linalg.norm(c - c_true))


# -----------------------------
# Single controlled trial
# -----------------------------
def run_single_poisson_trial(
    rng: np.random.Generator,
    replicate: int,
    n_state: int,
    n_basis: int,
    n_obs: int,
    noise_fraction: float,
    learned_coeff_scale: float,
    mismatch_amplitude: float,
    lam: float,
    eps_safe: float,
    alpha_pde: float,
    alpha_opt: float,
) -> SweepRow:
    x, B = sine_basis_1d(n_state, n_basis)
    K, _ = fd_poisson_matrix(n_state)
    H = observation_matrix(n_state, n_obs, rng, mode="random")

    S = np.linalg.solve(K, B)
    F = H @ S
    C_stab, sigma_min, svals = stable_inverse_constant(F)
    condition_number = float(np.max(svals) / max(np.min(svals), 1e-12))

    c_true = np.zeros(n_basis)
    template = np.array([1.2, -0.7, 0.45, 0.25, -0.22, 0.12, 0.08, -0.06, 0.04, 0.02])
    c_true[: min(n_basis, len(template))] = template[: min(n_basis, len(template))]

    q_true = B @ c_true
    u_true = solve_poisson(q_true, K)

    clean_y = H @ u_true
    if noise_fraction == 0:
        noise = np.zeros(n_obs)
    else:
        noise_scale = noise_fraction * np.linalg.norm(clean_y) / np.sqrt(n_obs)
        noise = noise_scale * rng.normal(size=n_obs)

    y = clean_y + noise
    delta_bound = float(np.linalg.norm(noise))

    # Conservative baseline
    c_base = ridge_solution(F, y, lam)
    q_base = B @ c_base
    u_base = solve_poisson(q_base, K)

    # Learned candidate: true coefficients plus perturbation.
    c_learn = c_true + learned_coeff_scale * rng.normal(size=n_basis)
    q_learn = B @ c_learn

    # State inconsistency models unfinished training, collocation failure, or mismatch.
    u_learn_consistent = solve_poisson(q_learn, K)
    mismatch = mismatch_amplitude * np.sin(15.0 * np.pi * x)
    u_learn = u_learn_consistent + mismatch

    # Residuals
    base_data = float(np.linalg.norm(H @ u_base - y))
    learn_data = float(np.linalg.norm(H @ u_learn - y))

    base_pde = float(np.linalg.norm(K @ u_base - q_base) / np.sqrt(n_state))
    learn_pde = float(np.linalg.norm(K @ u_learn - q_learn) / np.sqrt(n_state))

    base_grad = F.T @ (F @ c_base - y) + lam * c_base
    learn_grad = F.T @ (F @ c_learn - y) + lam * c_learn

    base_opt = float(np.linalg.norm(base_grad))
    learn_opt = float(np.linalg.norm(learn_grad))

    R_base = compute_operational_radius(
        C_stab=C_stab,
        data_residual=base_data,
        pde_residual=base_pde,
        boundary_residual=0.0,
        delta_bound=delta_bound,
        opt_residual=base_opt,
        alpha_pde=alpha_pde,
        alpha_opt=alpha_opt,
    )
    R_learn = compute_operational_radius(
        C_stab=C_stab,
        data_residual=learn_data,
        pde_residual=learn_pde,
        boundary_residual=0.0,
        delta_bound=delta_bound,
        opt_residual=learn_opt,
        alpha_pde=alpha_pde,
        alpha_opt=alpha_opt,
    )

    base_err = coeff_error(c_base, c_true)
    learn_err = coeff_error(c_learn, c_true)

    certificate_ratio = float(R_learn / (R_base + 1e-12))
    error_ratio = float(learn_err / (base_err + 1e-12))

    accepted = bool(R_learn <= R_base + eps_safe)
    learned_better = bool(learn_err <= base_err)

    unsafe_acceptance = bool(accepted and not learned_better)
    false_rejection = bool((not accepted) and learned_better)
    safe_improvement = bool(accepted and learned_better)

    return SweepRow(
        replicate=replicate,
        n_state=n_state,
        n_basis=n_basis,
        n_obs=n_obs,
        noise_fraction=noise_fraction,
        learned_coeff_scale=learned_coeff_scale,
        mismatch_amplitude=mismatch_amplitude,
        sigma_min=sigma_min,
        C_stab=C_stab,
        condition_number=condition_number,
        delta_bound=delta_bound,
        baseline_error=base_err,
        learned_error=learn_err,
        baseline_data_residual=base_data,
        learned_data_residual=learn_data,
        baseline_pde_residual=base_pde,
        learned_pde_residual=learn_pde,
        baseline_opt_residual=base_opt,
        learned_opt_residual=learn_opt,
        R_base=R_base,
        R_learn=R_learn,
        certificate_ratio=certificate_ratio,
        error_ratio=error_ratio,
        accepted_by_noharm=accepted,
        learned_actually_better=learned_better,
        unsafe_acceptance=unsafe_acceptance,
        false_rejection=false_rejection,
        safe_improvement=safe_improvement,
    )


# -----------------------------
# Sweep
# -----------------------------
def run_sufficiency_sweep(
    seed: int,
    reps: int,
    n_state: int,
    n_basis: int,
    lam: float,
    eps_safe: float,
    alpha_pde: float,
    alpha_opt: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_obs_values = [10, 15, 20, 25, 35, 50, 70]
    noise_values = [0.0, 0.01, 0.02, 0.05, 0.10]
    learned_scales = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
    mismatch_values = [0.0, 0.02, 0.05, 0.10]

    rows: List[SweepRow] = []
    rep_id = 0

    for n_obs in n_obs_values:
        for noise_fraction in noise_values:
            for learned_scale in learned_scales:
                for mismatch_amp in mismatch_values:
                    for _ in range(reps):
                        rep_id += 1
                        row = run_single_poisson_trial(
                            rng=rng,
                            replicate=rep_id,
                            n_state=n_state,
                            n_basis=n_basis,
                            n_obs=n_obs,
                            noise_fraction=noise_fraction,
                            learned_coeff_scale=learned_scale,
                            mismatch_amplitude=mismatch_amp,
                            lam=lam,
                            eps_safe=eps_safe,
                            alpha_pde=alpha_pde,
                            alpha_opt=alpha_opt,
                        )
                        rows.append(row)

    return pd.DataFrame([r.__dict__ for r in rows])


# -----------------------------
# Summaries
# -----------------------------
def summarize_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["n_obs", "noise_fraction", "learned_coeff_scale", "mismatch_amplitude"]
    summary = (
        df.groupby(group_cols)
        .agg(
            trials=("replicate", "count"),
            median_C_stab=("C_stab", "median"),
            median_condition_number=("condition_number", "median"),
            median_R_base=("R_base", "median"),
            median_R_learn=("R_learn", "median"),
            median_certificate_ratio=("certificate_ratio", "median"),
            median_error_ratio=("error_ratio", "median"),
            accept_rate=("accepted_by_noharm", "mean"),
            learned_better_rate=("learned_actually_better", "mean"),
            safe_improvement_rate=("safe_improvement", "mean"),
            unsafe_acceptance_rate=("unsafe_acceptance", "mean"),
            false_rejection_rate=("false_rejection", "mean"),
        )
        .reset_index()
    )

    summary["sufficiency_regime"] = np.where(
        summary["median_certificate_ratio"] <= 1.0,
        "certificate-sufficient",
        "fallback-required",
    )
    return summary


def write_boxed_conditions_tex(tab_dir: str) -> None:
    tex = r"""
\begin{table*}[!ht]
\centering
\caption{Reusable no-harm conditions for deciding when physics-informed inverse learning is sufficient to replace a baseline. The learned reconstruction is selected only when the certificate-dominance condition holds.}
\label{tab:noharm-reusable-conditions}
\resizebox{\textwidth}{!}{%
\begin{tabular}{p{3.2cm}p{5.6cm}p{6.4cm}}
\toprule
\textbf{Condition} & \textbf{Computable test} & \textbf{Meaning for physics-informed inverse learning} \\
\midrule
Admissible inverse stability &
A conditional stability estimate holds on the admissible class, with finite \(C_{\mathrm{stab}}\) and exponent \(p\in(0,1]\). &
Residuals can be interpreted as reliability evidence only on a class where the inverse problem is stable enough for certification. \\
Residual-calibrated uncertainty &
Compute \(R_\delta\), \(R_{\delta,\zeta}^{\mathrm{stoch}}\), or \(R_\delta^{\mathrm{op}}\) using data, physics, boundary, noise, and optimization information. &
The output is not only a reconstruction; it carries a radius that quantifies the available reliability evidence. \\
Certificate dominance &
\[
R_{\mathrm{learn}}\le R_{\mathrm{base}}+\varepsilon_{\mathrm{safe}}.
\]
&
The learned physics-informed reconstruction is sufficient to replace the baseline only in this region. \\
Fallback condition &
\[
R_{\mathrm{learn}}>R_{\mathrm{base}}+\varepsilon_{\mathrm{safe}}.
\]
&
The learned output may still look plausible, but it is not certified enough to overrule the baseline. The method returns the baseline. \\
Independent residual validation &
Use validation collocation points not used for training, and replace \(r_{\mathrm{pde}}\) by \(\widehat r_{\mathrm{pde},\zeta}\) when stochastic residuals are used. &
The certificate should be based on residual evidence available after training, not only on the training loss. \\
Optimization adequacy &
Require \(r_{\mathrm{opt}}(\widehat\theta)\) to be small or include \(\alpha_{\mathrm{opt}}r_{\mathrm{opt}}(\widehat\theta)\) in \(R_\delta^{\mathrm{op}}\). &
A learned solution with unfinished optimization should not be treated as fully certified even if some residual terms are small. \\
\bottomrule
\end{tabular}%
}
\end{table*}
"""
    with open(os.path.join(tab_dir, "boxed_noharm_conditions.tex"), "w", encoding="utf-8") as f:
        f.write(tex.strip() + "\n")


# -----------------------------
# Figures
# -----------------------------
def plot_certificate_boundary(df: pd.DataFrame, fig_dir: str) -> None:
    sample = df.copy()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=1)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = np.where(sample["accepted_by_noharm"], "tab:green", "tab:red")
    ax.scatter(
        sample["certificate_ratio"],
        sample["error_ratio"],
        c=colors,
        alpha=0.45,
        s=24,
        edgecolor="none",
    )
    ax.axvline(1.0, linestyle="--", color="black", linewidth=1.2)
    ax.axhline(1.0, linestyle=":", color="black", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("certificate ratio: learned radius / baseline radius")
    ax.set_ylabel("hindsight error ratio: learned error / baseline error")
    ax.set_title("No-harm selection boundary")
    ax.text(
        0.55,
        0.06,
        "selected and better",
        transform=ax.transAxes,
        fontsize=9,
        ha="center",
        va="center",
    )
    ax.text(
        0.78,
        0.88,
        "fallback region",
        transform=ax.transAxes,
        fontsize=9,
        ha="center",
        va="center",
    )
    ax.text(
        0.04,
        0.96,
        "green: selected\nred: fallback",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    savefig(fig, fig_dir, "certificate_boundary_scatter")


def pivot_metric(
    summary: pd.DataFrame,
    noise_fraction: float,
    mismatch_amplitude: float,
    metric: str,
) -> pd.DataFrame:
    sub = summary[
        (summary["noise_fraction"] == noise_fraction)
        & (summary["mismatch_amplitude"] == mismatch_amplitude)
    ].copy()
    return sub.pivot(
        index="learned_coeff_scale",
        columns="n_obs",
        values=metric,
    ).sort_index(ascending=False)


def heatmap(
    mat: pd.DataFrame,
    fig_dir: str,
    name: str,
    title: str,
    cbar_label: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.imshow(mat.values, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_xlabel("number of observations")
    ax.set_ylabel("learned coefficient perturbation scale")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    savefig(fig, fig_dir, name)


def plot_heatmaps(summary: pd.DataFrame, fig_dir: str) -> None:
    noise = 0.02
    mismatch = 0.05

    accept = pivot_metric(summary, noise, mismatch, "accept_rate")
    unsafe = pivot_metric(summary, noise, mismatch, "unsafe_acceptance_rate")
    false_rej = pivot_metric(summary, noise, mismatch, "false_rejection_rate")

    heatmap(
        accept,
        fig_dir,
        "acceptance_rate_heatmap",
        title="Selection rate under no-harm rule (noise=0.02, mismatch=0.05)",
        cbar_label="selection rate",
    )
    heatmap(
        unsafe,
        fig_dir,
        "unsafe_acceptance_heatmap",
        title="Unsafe selection diagnostic (noise=0.02, mismatch=0.05)",
        cbar_label="rate",
    )
    heatmap(
        false_rej,
        fig_dir,
        "false_rejection_heatmap",
        title="False rejection diagnostic (noise=0.02, mismatch=0.05)",
        cbar_label="rate",
    )


def plot_certificate_ratio_by_data_noise(summary: pd.DataFrame, fig_dir: str) -> None:
    # Fix a moderate learned perturbation and mismatch, then show median certificate ratio over data/noise.
    learned_scale = 0.05
    mismatch = 0.02

    sub = summary[
        (summary["learned_coeff_scale"] == learned_scale)
        & (summary["mismatch_amplitude"] == mismatch)
    ].copy()

    mat = sub.pivot(
        index="noise_fraction",
        columns="n_obs",
        values="median_certificate_ratio",
    ).sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_xlabel("number of observations")
    ax.set_ylabel("noise fraction")
    ax.set_title("Median certificate ratio: learned radius / baseline radius")

    # Mark the selection threshold. Cells below or equal to one are sufficient.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            label = f"{val:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("median certificate ratio")
    savefig(fig, fig_dir, "certificate_ratio_by_data_noise")


def plot_all(df: pd.DataFrame, summary: pd.DataFrame, fig_dir: str) -> None:
    plot_certificate_boundary(df, fig_dir)
    plot_heatmaps(summary, fig_dir)
    plot_certificate_ratio_by_data_noise(summary, fig_dir)


# -----------------------------
# Manuscript statement
# -----------------------------
def write_manuscript_statement(tab_dir: str) -> None:
    statement = r"""
Field-level validation statement:

The sufficiency sweep supports the no-harm PIIL principle. Physics-informed
inverse learning is sufficient to replace a baseline only in the certificate-
dominance region
\[
R_{\mathrm{learn}}\le R_{\mathrm{base}}+\varepsilon_{\mathrm{safe}}.
\]
Outside this region, the learned reconstruction may still be visually plausible
or accurate in hindsight, but it is not certified enough to overrule the
baseline. The practical threshold is therefore not small PINN loss, visual
quality, or synthetic hindsight error. The practical threshold is residual-
calibrated certificate dominance.
"""
    with open(os.path.join(tab_dir, "manuscript_sufficiency_statement.txt"), "w", encoding="utf-8") as f:
        f.write(statement.strip() + "\n")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results_noharm_map")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--n_state", type=int, default=120)
    parser.add_argument("--n_basis", type=int, default=10)
    parser.add_argument("--lambda_ridge", type=float, default=1e-5)
    parser.add_argument("--eps_safe", type=float, default=0.0)
    parser.add_argument("--alpha_pde", type=float, default=0.05)
    parser.add_argument("--alpha_opt", type=float, default=0.01)
    args = parser.parse_args()

    fig_dir, tab_dir = ensure_dirs(args.out)

    df = run_sufficiency_sweep(
        seed=args.seed,
        reps=args.reps,
        n_state=args.n_state,
        n_basis=args.n_basis,
        lam=args.lambda_ridge,
        eps_safe=args.eps_safe,
        alpha_pde=args.alpha_pde,
        alpha_opt=args.alpha_opt,
    )

    summary = summarize_by_regime(df)

    df.to_csv(os.path.join(tab_dir, "sufficiency_sweep_raw.csv"), index=False)
    summary.to_csv(os.path.join(tab_dir, "sufficiency_summary_by_regime.csv"), index=False)

    write_boxed_conditions_tex(tab_dir)
    write_manuscript_statement(tab_dir)
    plot_all(df, summary, fig_dir)

    print("Done.")
    print(f"Raw sweep table: {os.path.join(tab_dir, 'sufficiency_sweep_raw.csv')}")
    print(f"Summary table:   {os.path.join(tab_dir, 'sufficiency_summary_by_regime.csv')}")
    print(f"Figures:         {fig_dir}")
    print(f"LaTeX table:     {os.path.join(tab_dir, 'boxed_noharm_conditions.tex')}")


if __name__ == "__main__":
    main()