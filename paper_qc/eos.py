"""EOS-80 density at pressure (UNESCO 1983 secant bulk modulus).

`paper_qc.lib.data` has the one-atmosphere equation of state and potential temperature. The
real-time density inversion test needs potential density referenced to the local mid-point
pressure, so this adds the pressure term. Temperatures in are ITS-90; EOS-80 is defined on
IPTS-68, hence the 1.00024 factor, as in paper_qc.lib.data.sigma_theta.
"""
import numpy as np

from paper_qc.lib.data import density_at_surface, potential_temperature


def secant_bulk_modulus(s, t, p_bar):
    """K(S, t, p) in bars; t on IPTS-68, p in bars (Fofonoff and Millard 1983)."""
    sr = np.sqrt(s)
    kw = 19652.21 + (148.4206 + (-2.327105 + (1.360477e-2 - 5.155288e-5 * t) * t) * t) * t
    aw = 3.239908 + (1.43713e-3 + (1.16092e-4 - 5.77905e-7 * t) * t) * t
    bw = 8.50935e-5 + (-6.12293e-6 + 5.2787e-8 * t) * t
    k0 = (kw + s * (54.6746 + (-0.603459 + (1.09987e-2 - 6.1670e-5 * t) * t) * t)
          + s * sr * (7.944e-2 + (1.6483e-2 - 5.3009e-4 * t) * t))
    a = aw + s * (2.2838e-3 + (-1.0981e-5 - 1.6078e-6 * t) * t) + 1.91075e-4 * s * sr
    b = bw + s * (-9.9348e-7 + (2.0816e-8 + 9.1697e-10 * t) * t)
    return k0 + (a + b * p_bar) * p_bar


def density(s, t68, p_dbar):
    """In-situ density (kg m-3); t on IPTS-68, p in dbar."""
    p_bar = np.asarray(p_dbar, dtype="float64") / 10.0
    return density_at_surface(s, t68) / (1.0 - p_bar / secant_bulk_modulus(s, t68, p_bar))


def potential_density(psal, temp, pres, pref):
    """Potential density (kg m-3) referenced to `pref` dbar; temp is ITS-90."""
    s = np.asarray(psal, dtype="float64")
    t68 = 1.00024 * np.asarray(temp, dtype="float64")
    p = np.asarray(pres, dtype="float64")
    theta = potential_temperature(s, t68, p, pref)
    return density(s, theta, pref)


def _self_test():
    checks = [  # UNESCO (1983) Technical Paper 44 check values, IPTS-68
        ("K(35, 25, 1000 bar)", float(secant_bulk_modulus(35.0, 25.0, 1000.0)), 27108.94504),
        ("rho(35, 25, 10000 dbar)", float(density(35.0, 25.0, 10000.0)), 1062.53817),
        ("rho(0, 5, 0)", float(density(0.0, 5.0, 0.0)), 999.96675),
    ]
    ok = True
    for name, got, want in checks:
        good = abs(got - want) < 1e-4
        ok &= good
        print(f"  {name:<24} {got:.5f}  expected {want:.5f}  {'ok' if good else 'FAIL'}")
    # referenced to the surface it must equal sigma-theta + 1000
    from paper_qc.lib.data import sigma_theta
    same = abs(float(potential_density(35.0, 3.0, 2000.0, 0.0)) - 1000.0
               - float(sigma_theta(np.array(35.0), np.array(3.0), np.array(2000.0)))) < 1e-9
    print(f"  potential density at pref=0 equals sigma-theta: {'ok' if same else 'FAIL'}")
    assert ok and same


if __name__ == "__main__":
    _self_test()
