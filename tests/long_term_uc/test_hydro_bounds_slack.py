import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pytest

from src.include.hydro_bounds_preprocessor import relax_single_hydro_reservoir_bounds_lp


def test_hydro_bounds_slack_initial_soc_conflict():
    """
    Test that an initial SoC (e.g. 0.80) conflicting with an overly strict
    first target Vmax (e.g. 0.70) is minimally relaxed to permit feasibility.
    """
    n_ts = 24
    energy_capa = 10000.0  # MWh
    p_nom = 100.0          # MW (can discharge at most 100 MWh in 1 hour = 0.01 p.u.)
    soc_init = 0.80 * energy_capa  # starts at 80%
    inflow = np.zeros(n_ts)

    # Impose Vmax = 0.70 at t = 0
    soc_min_pu = np.zeros(n_ts)
    soc_max_pu = np.ones(n_ts)
    soc_max_pu[0] = 0.70

    relaxed_min, relaxed_max = relax_single_hydro_reservoir_bounds_lp(
        unit_name="test_hydro",
        soc_min_pu=soc_min_pu,
        soc_max_pu=soc_max_pu,
        energy_capa=energy_capa,
        p_nom=p_nom,
        soc_init=soc_init,
        inflow=inflow,
        min_bandwidth_pu=0.02
    )

    # In 1 hour, max discharge is p_nom = 100 MWh -> SoC at t=0 cannot be lower than 8000 - 100 = 7900 MWh (0.79 p.u.)
    # Therefore, relaxed Vmax at t=0 must be raised to at least 0.79
    assert relaxed_max[0] >= 0.79 - 1e-5
    # For unconstrained hours, Vmax remains 1.0
    assert relaxed_max[1] == pytest.approx(1.0, abs=1e-5)


def test_hydro_bounds_slack_out_of_scale_scandinavia():
    """
    Test that out-of-scale bounds (> 1.0, e.g. Scandinavia with max ~ 4.79)
    are properly normalized and relaxed rather than clipped to 100% constant.
    """
    n_ts = 168
    energy_capa = 40000.0
    p_nom = 1000.0
    soc_init = 0.80 * energy_capa
    inflow = np.full(n_ts, 10.0)

    # Raw ERAA-like numbers for Scandinavia: week 1 min=1.199, max=4.397
    soc_min_pu = np.zeros(n_ts)
    soc_max_pu = np.ones(n_ts)
    soc_min_pu[0] = 1.199
    soc_max_pu[0] = 4.397

    relaxed_min, relaxed_max = relax_single_hydro_reservoir_bounds_lp(
        unit_name="scandinavia_reservoir",
        soc_min_pu=soc_min_pu,
        soc_max_pu=soc_max_pu,
        energy_capa=energy_capa,
        p_nom=p_nom,
        soc_init=soc_init,
        inflow=inflow,
        min_bandwidth_pu=0.05
    )

    # Normalized by scale factor 5: min ~ 0.24, max ~ 0.88
    # Since soc_init is 0.80, it lies well within [0.24, 0.88]!
    assert relaxed_min[0] <= 0.80 + 1e-5
    assert relaxed_max[0] >= 0.80 - 1e-5
    # Bounds must be within [0, 1]
    assert np.all(relaxed_min >= 0.0)
    assert np.all(relaxed_max <= 1.0)


def test_hydro_bounds_slack_already_feasible_unchanged():
    """
    Test that when bounds are easily reachable and feasible,
    the preprocessor preserves them without distortion.
    """
    n_ts = 24
    energy_capa = 10000.0
    p_nom = 500.0
    soc_init = 0.50 * energy_capa
    inflow = np.full(n_ts, 50.0)

    # Feasible corridor around 50%
    soc_min_pu = np.zeros(n_ts)
    soc_max_pu = np.ones(n_ts)
    soc_min_pu[0] = 0.40
    soc_max_pu[0] = 0.60

    relaxed_min, relaxed_max = relax_single_hydro_reservoir_bounds_lp(
        unit_name="test_feasible",
        soc_min_pu=soc_min_pu,
        soc_max_pu=soc_max_pu,
        energy_capa=energy_capa,
        p_nom=p_nom,
        soc_init=soc_init,
        inflow=inflow,
        min_bandwidth_pu=0.05
    )

    assert relaxed_min[0] == pytest.approx(0.40, abs=1e-5)
    assert relaxed_max[0] == pytest.approx(0.60, abs=1e-5)


def test_hydro_bounds_slack_bandwidth_enforcement():
    """
    Test that when Vmin and Vmax are extremely close, the preprocessor
    expands them to guarantee a minimum operational bandwidth for dispatch.
    """
    n_ts = 24
    energy_capa = 10000.0
    p_nom = 500.0
    soc_init = 0.50 * energy_capa
    inflow = np.full(n_ts, 50.0)

    # Pinpoint bounds Vmin = 0.50, Vmax = 0.50
    soc_min_pu = np.zeros(n_ts)
    soc_max_pu = np.ones(n_ts)
    soc_min_pu[0] = 0.50
    soc_max_pu[0] = 0.50

    min_bandwidth = 0.06
    relaxed_min, relaxed_max = relax_single_hydro_reservoir_bounds_lp(
        unit_name="test_pinpoint",
        soc_min_pu=soc_min_pu,
        soc_max_pu=soc_max_pu,
        energy_capa=energy_capa,
        p_nom=p_nom,
        soc_init=soc_init,
        inflow=inflow,
        min_bandwidth_pu=min_bandwidth
    )

    width = relaxed_max[0] - relaxed_min[0]
    assert width >= min_bandwidth - 1e-5
