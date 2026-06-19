from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class NoHarmDecision:
    candidate: str
    baseline: str
    accepted: bool
    safe_choice: str
    candidate_radius: float
    baseline_radius: float
    certificate_ratio: float
    eps_safe: float


def noharm_select(
    candidate_radius: float,
    baseline_radius: float,
    candidate: str = "learned",
    baseline: str = "baseline",
    eps_safe: float = 0.0,
) -> NoHarmDecision:
    """Apply the no-harm replacement rule.

    A learned/assisted candidate replaces the baseline only when
    candidate_radius <= baseline_radius + eps_safe.
    """
    if baseline_radius < 0 or candidate_radius < 0:
        raise ValueError("certificate radii must be nonnegative.")
    accepted = bool(candidate_radius <= baseline_radius + eps_safe)
    return NoHarmDecision(
        candidate=candidate,
        baseline=baseline,
        accepted=accepted,
        safe_choice=candidate if accepted else baseline,
        candidate_radius=float(candidate_radius),
        baseline_radius=float(baseline_radius),
        certificate_ratio=float(candidate_radius / max(baseline_radius, 1e-14)),
        eps_safe=float(eps_safe),
    )


def rank_accepted_candidates(
    radii: Mapping[str, float],
    baseline: str = "baseline",
    eps_safe: float = 0.0,
) -> Tuple[str, List[NoHarmDecision]]:
    """Return the safest accepted candidate and all no-harm decisions.

    The baseline must be present. Among accepted candidates, the smallest
    certified radius is selected. If none improves the baseline, the baseline
    is returned.
    """
    if baseline not in radii:
        raise KeyError(f"baseline {baseline!r} not found in radii.")
    baseline_radius = float(radii[baseline])
    decisions: List[NoHarmDecision] = []
    safe_choice = baseline
    best_radius = baseline_radius
    for name, radius in radii.items():
        if name == baseline:
            continue
        decision = noharm_select(radius, baseline_radius, candidate=name, baseline=baseline, eps_safe=eps_safe)
        decisions.append(decision)
        if decision.accepted and radius < best_radius:
            best_radius = float(radius)
            safe_choice = name
    return safe_choice, decisions
