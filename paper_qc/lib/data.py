"""
Loading the level table, the EOS-80 equation of state and oxygen solubility.

`load` reads the table written by `paper_qc.lib.build` (one row per retained level: pressure,
temperature, salinity, oxygen, position, time, float and cycle).

* `measured_ts_only` (default True) keeps levels where temperature and salinity were measured,
  dropping those where the synthetic-profile procedure interpolated them (TS_INTERP_FLAG).
* `o2_range` drops physically implausible oxygen. Surface saturation even in the coldest
  Southern Ocean water is 350-400 umol/kg, so the few levels above 400 are sensor artefacts that
  survived flagging; the lower bound is loose because small negatives are genuine near-zero
  measurements in oxygen-poor water.

The equation of state is EOS-80 (UNESCO 1983): the one-atmosphere density, the adiabatic lapse
rate and potential temperature. Oxygen solubility is Garcia and Gordon (1992).
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

DATA_PATH = os.path.join("data", "qc_levels.nc")
O2_RANGE = (-5.0, 400.0)

def load(path=None, measured_ts_only=True, o2_range=O2_RANGE, verbose=True):
    """The level table with year and day-of-year attached."""
    df = xr.open_dataset(path or DATA_PATH).to_dataframe().reset_index(drop=True)
    n0 = len(df)

    if measured_ts_only:
        df = df[df.TS_INTERP_FLAG == 0]
    n1 = len(df)

    if o2_range is not None:
        df = df[df.DOXY.between(*o2_range)]
    n2 = len(df)

    df = df.reset_index(drop=True)
    time = pd.to_datetime(df.TIME, unit="s")
    df["year"] = time.dt.year
    df["doy"] = time.dt.dayofyear

    if verbose:
        print(f"loaded {n0:,} levels -> {len(df):,} "
              f"({n0 - n1:,} interpolated T/S, {n1 - n2:,} outside {o2_range} umol/kg)")
        print(f"  {df.PLATFORM.nunique()} floats, {df.groupby(['PLATFORM', 'TIME']).ngroups:,} "
              f"profiles, {time.min():%Y-%m} to {time.max():%Y-%m}")
    return df

def _adiabatic_lapse_rate(s, t, p):
    """Adiabatic temperature gradient, deg C per dbar (UNESCO 1983, Bryden 1973)."""
    ds = s - 35.0
    return (3.5803e-5 + (8.5258e-6 + (-6.836e-8 + 6.6228e-10 * t) * t) * t
            + (1.8932e-6 + -4.2393e-8 * t) * ds
            + ((1.8741e-8 + (-6.7795e-10 + (8.733e-12 + -5.4481e-14 * t) * t) * t)
               + (-1.1351e-10 + 2.7759e-12 * t) * ds) * p
            + (-4.6206e-13 + (1.8676e-14 + -2.1687e-16 * t) * t) * p * p)

def potential_temperature(s, t, p, pr=0.0):
    """Potential temperature by the standard fourth-order Runge-Kutta integration."""
    h = pr - p
    xk = h * _adiabatic_lapse_rate(s, t, p)
    t = t + 0.5 * xk
    q = xk
    p = p + 0.5 * h
    xk = h * _adiabatic_lapse_rate(s, t, p)
    t = t + 0.29289322 * (xk - q)
    q = 0.58578644 * xk + 0.121320344 * q
    xk = h * _adiabatic_lapse_rate(s, t, p)
    t = t + 1.707106781 * (xk - q)
    q = 3.414213562 * xk - 4.121320344 * q
    p = p + 0.5 * h
    xk = h * _adiabatic_lapse_rate(s, t, p)
    return t + (xk - 2.0 * q) / 6.0

def density_at_surface(s, t):
    """Density (kg m-3) at one atmosphere: the EOS-80 international equation of state."""
    smow = (999.842594 + (6.793952e-2 + (-9.095290e-3 + (1.001685e-4
            + (-1.120083e-6 + 6.536332e-9 * t) * t) * t) * t) * t)
    b = (8.24493e-1 + (-4.0899e-3 + (7.6438e-5 + (-8.2467e-7 + 5.3875e-9 * t) * t) * t) * t)
    c = -5.72466e-3 + (1.0227e-4 + -1.6546e-6 * t) * t
    return smow + b * s + c * s * np.sqrt(s) + 4.8314e-4 * s * s

def sigma_theta(psal, temp, pres):
    """Potential density anomaly referenced to the surface (kg m-3).

    Temperatures are ITS-90 as Argo reports them; EOS-80 is defined on IPTS-68, hence the
    1.00024 conversion.
    """
    t68 = 1.00024 * np.asarray(temp, dtype="float64")
    s = np.asarray(psal, dtype="float64")
    p = np.asarray(pres, dtype="float64")
    theta = potential_temperature(s, t68, p, 0.0)
    return density_at_surface(s, theta) - 1000.0

def oxygen_solubility(psal, temp):
    """Oxygen solubility at one atmosphere, umol/kg (Garcia and Gordon 1992).

    The concentration seawater of this temperature and salinity would hold at equilibrium
    with the atmosphere. Used to carry a salinity error into optode oxygen (inject.py).
    """
    t = np.asarray(temp, dtype="float64")
    s = np.asarray(psal, dtype="float64")
    ts = np.log((298.15 - t) / (273.15 + t))
    a = [5.80871, 3.20291, 4.17887, 5.10006, -9.86643e-2, 3.80369]
    b = [-7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3]
    c0 = -2.75915e-7
    ln_c = (a[0] + a[1] * ts + a[2] * ts ** 2 + a[3] * ts ** 3 + a[4] * ts ** 4
            + a[5] * ts ** 5
            + s * (b[0] + b[1] * ts + b[2] * ts ** 2 + b[3] * ts ** 3)
            + c0 * s ** 2)
    return np.exp(ln_c)
