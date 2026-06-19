from noharm_piil.certificates import compute_operational_radius
from noharm_piil.decisions import noharm_select

base = compute_operational_radius(
    stability_constant=2.0,
    data_residual=0.04,
    physics_residual=0.02,
    delta_bound=0.01,
    alpha_physics=0.5,
)
learned = compute_operational_radius(
    stability_constant=2.0,
    data_residual=0.02,
    physics_residual=0.01,
    delta_bound=0.01,
    alpha_physics=0.5,
)

decision = noharm_select(learned.radius, base.radius, candidate="learned", baseline="baseline")
print(decision)
