from noharm_piil.certificates import compute_operational_radius


def test_operational_radius_combines_residuals():
    cert = compute_operational_radius(
        stability_constant=2.0,
        data_residual=1.0,
        physics_residual=3.0,
        boundary_residual=5.0,
        optimization_residual=7.0,
        delta_bound=11.0,
        alpha_physics=0.1,
        alpha_boundary=0.2,
        alpha_optimization=0.3,
    )
    expected = 2.0 * (1.0 + 0.1 * 3.0 + 0.2 * 5.0 + 0.3 * 7.0 + 11.0)
    assert abs(cert.radius - expected) < 1e-12


def test_operational_radius_rejects_bad_exponent():
    try:
        compute_operational_radius(1.0, 1.0, exponent=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
