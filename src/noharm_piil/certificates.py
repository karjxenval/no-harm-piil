from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class OperationalCertificate:
    """Residual-calibrated certificate for a computed inverse/PDE candidate.

    The radius is intentionally operational: it combines data fit, physics
    residuals, boundary/constraint residuals, optimization residuals, and a
    noise/modeling allowance under a stability constant.
    """

    radius: float
    stability_constant: float
    data_residual: float
    physics_residual: float = 0.0
    boundary_residual: float = 0.0
    optimization_residual: float = 0.0
    delta_bound: float = 0.0
    exponent: float = 1.0
    alpha_physics: float = 1.0
    alpha_boundary: float = 1.0
    alpha_optimization: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_operational_radius(
    stability_constant: float,
    data_residual: float,
    physics_residual: float = 0.0,
    boundary_residual: float = 0.0,
    optimization_residual: float = 0.0,
    delta_bound: float = 0.0,
    exponent: float = 1.0,
    alpha_physics: float = 1.0,
    alpha_boundary: float = 1.0,
    alpha_optimization: float = 0.0,
) -> OperationalCertificate:
    """Compute the no-harm PIIL operational certificate radius.

    The learned candidate is not judged by visual plausibility or training loss
    alone. It receives a certified radius and is selected only if that radius
    is no worse than the baseline radius under the no-harm rule.
    """
    if stability_constant < 0:
        raise ValueError("stability_constant must be nonnegative.")
    if exponent <= 0:
        raise ValueError("exponent must be positive.")

    total = (
        data_residual
        + alpha_physics * physics_residual
        + alpha_boundary * boundary_residual
        + alpha_optimization * optimization_residual
        + delta_bound
    )
    radius = float(stability_constant * max(float(total), 0.0) ** exponent)
    return OperationalCertificate(
        radius=radius,
        stability_constant=float(stability_constant),
        data_residual=float(data_residual),
        physics_residual=float(physics_residual),
        boundary_residual=float(boundary_residual),
        optimization_residual=float(optimization_residual),
        delta_bound=float(delta_bound),
        exponent=float(exponent),
        alpha_physics=float(alpha_physics),
        alpha_boundary=float(alpha_boundary),
        alpha_optimization=float(alpha_optimization),
    )
