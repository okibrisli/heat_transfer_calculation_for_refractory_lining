"""
film_models.py
Surface film coefficient models for the wall scripts.

Every film is a callable: film(Ts_c) returns (h_total, h_conv, h_rad) in W/m2K.

INSIDE film modes (ambient = hot gas temperature)
  manual       {"mode": "manual", "h": 150.0}
  radiation    {"mode": "radiation", "eps_eff": 0.5, "h_conv": 10.0}

OUTSIDE film modes (ambient = outside air temperature)
  manual       {"mode": "manual", "h": 8.132}              fixed value, e.g. from SimuTherm
  astm_c680    Churchill and Chu, McAdams, Churchill mixing (ASTM C680)
  iso12241     ISO 12241 vertical wall equations (orientation V only)
  simple_air   simplified air correlations (1.42 dT/L, 1.31 dT^(1/3) and so on)
  calibrated   convection fitted to your own SimuTherm readings
  juerges      Juerges wind correlation 5.8 + 3.95 v (smooth surface, v up to 5 m/s)

Common keys for the outside modes (except manual):
  orientation  V = vertical, R = ceiling (heat flow up), F = floor (heat flow down)
  emissivity   surface emissivity for the radiation part
  wind_m_s     wind speed, 0 for still air
  height_m     wall height (V)
  length_m     A/P for R and F, also the length along the wind
"""

import numpy as np

SIGMA = 5.6697e-8
G = 9.81

# SimuTherm readings for the "calibrated" mode:
# (orientation, surface C, ambient C, emissivity, h_total W/m2K). Add rows freely.
SIMUTHERM_POINTS = [
    ("V", 49.7, 30.0, 0.5, 8.132),
    ("V", 56.3, 40.0, 0.5, 8.046),
    ("R", 50.3, 30.0, 0.5, 7.876),
    ("F", 56.6, 30.0, 0.5, 5.981),
]
DEFAULT_EXPONENT = 0.25


# =============================================================================
# BUILDING BLOCKS
# =============================================================================
def h_radiation(Ts_c, Ta_c, eps):
    Ts, Ta = Ts_c + 273.15, Ta_c + 273.15
    return eps * SIGMA * (Ts ** 3 + Ts ** 2 * Ta + Ts * Ta ** 2 + Ta ** 3)


def air_properties(Tf_k):
    """Dry air at 1 atm. Replace this function to use other property fits."""
    mu = 1.716e-5 * (Tf_k / 273.15) ** 1.5 * (273.15 + 110.4) / (Tf_k + 110.4)
    kf = 0.0241 * (Tf_k / 273.15) ** 1.5 * (273.15 + 194.0) / (Tf_k + 194.0)
    cp = (1.9327e-10 * Tf_k ** 4 - 7.9999e-7 * Tf_k ** 3 + 1.1407e-3 * Tf_k ** 2
          - 0.4489 * Tf_k + 1057.5)
    rho = 101325.0 / (287.05 * Tf_k)
    return {"nu": mu / rho, "k": kf, "Pr": mu * cp / kf, "beta": 1.0 / Tf_k}


def _rayleigh(Ts_c, Ta_c, L):
    Tf = 0.5 * (Ts_c + Ta_c) + 273.15
    p = air_properties(Tf)
    return G * p["beta"] * abs(Ts_c - Ta_c) * L ** 3 * p["Pr"] / p["nu"] ** 2, p


def _length(cfg):
    ori = cfg.get("orientation", "V")
    return float(cfg.get("height_m", 1.0)) if ori == "V" else float(cfg.get("length_m", 1.0))


def _h_forced(Ts_c, Ta_c, wind, length_m):
    if wind <= 0:
        return 0.0
    Tf = 0.5 * (Ts_c + Ta_c) + 273.15
    p = air_properties(Tf)
    Re = wind * length_m / p["nu"]
    if Re < 5e5:
        Nu = 0.6774 * Re ** 0.5 * p["Pr"] ** (1 / 3) / (1 + (0.0468 / p["Pr"]) ** (2 / 3)) ** 0.25
    else:
        Nu = (0.037 * Re ** 0.8 - 871.0) * p["Pr"] ** (1 / 3)
    return Nu * p["k"] / length_m


# =============================================================================
# NATURAL CONVECTION MODELS: each returns h_conv in W/m2K
# =============================================================================
def natural_astm(Ts_c, Ta_c, cfg):
    ori, L = cfg.get("orientation", "V"), _length(cfg)
    dT = abs(Ts_c - Ta_c)
    if dT < 1e-9:
        return 0.0
    Ra, p = _rayleigh(Ts_c, Ta_c, L)
    if ori == "V":
        Nu = (0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / p["Pr"]) ** (9 / 16)) ** (8 / 27)) ** 2
    elif ori == "R":
        Nu = 0.54 * Ra ** 0.25 if Ra < 1e7 else 0.15 * Ra ** (1 / 3)
    elif ori == "F":
        Nu = 0.27 * Ra ** 0.25
    else:
        raise ValueError("orientation must be V, R or F")
    return Nu * p["k"] / L


def natural_iso12241(Ts_c, Ta_c, cfg):
    if cfg.get("orientation", "V") != "V":
        raise ValueError("iso12241 is implemented for orientation V only. "
                         "Use astm_c680 or simple_air for R and F.")
    H = float(cfg.get("height_m", 1.0))
    dT = abs(Ts_c - Ta_c)
    if dT < 1e-9:
        return 0.0
    if H ** 3 * dT <= 10.0:
        return 1.32 * (dT / H) ** 0.25
    return 1.74 * dT ** (1.0 / 3.0)


def natural_simple_air(Ts_c, Ta_c, cfg):
    ori, L = cfg.get("orientation", "V"), _length(cfg)
    dT = abs(Ts_c - Ta_c)
    if dT < 1e-9:
        return 0.0
    Ra, _ = _rayleigh(Ts_c, Ta_c, L)
    if ori == "V":
        return 1.42 * (dT / L) ** 0.25 if Ra < 1e9 else 1.31 * dT ** (1 / 3)
    if ori == "R":
        return 1.32 * (dT / L) ** 0.25 if Ra < 1e7 else 1.52 * dT ** (1 / 3)
    if ori == "F":
        return 0.59 * (dT / L) ** 0.25
    raise ValueError("orientation must be V, R or F")


def _fit_calibration(points):
    fits = {}
    for ori in sorted({p[0] for p in points}):
        rows = [(ts - ta, h - h_radiation(ts, ta, e)) for o, ts, ta, e, h in points if o == ori]
        dT = np.array([r[0] for r in rows])
        hc = np.array([r[1] for r in rows])
        if len(rows) >= 2 and np.ptp(dT) > 1.0:
            n, lnC = np.polyfit(np.log(dT), np.log(hc), 1)
            fits[ori] = (float(np.exp(lnC)), float(n))
        else:
            fits[ori] = (float(np.mean(hc / dT ** DEFAULT_EXPONENT)), DEFAULT_EXPONENT)
    return fits


def make_natural_calibrated(cfg):
    fits = _fit_calibration(cfg.get("points", SIMUTHERM_POINTS))
    ori = cfg.get("orientation", "V")
    if ori not in fits:
        raise ValueError(f"No calibration points for orientation {ori}.")
    C, n = fits[ori]
    return lambda Ts_c, Ta_c, c: C * abs(Ts_c - Ta_c) ** n


NATURAL_MODELS = {
    "astm_c680": natural_astm,
    "iso12241": natural_iso12241,
    "simple_air": natural_simple_air,
}

LABELS = {
    "astm_c680": "ASTM C680",
    "iso12241": "ISO 12241",
    "simple_air": "Simplified air correlations",
    "calibrated": "Calibrated on SimuTherm readings",
    "juerges": "Juerges wind correlation",
}


# =============================================================================
# FILM FACTORY
# =============================================================================
class Film:
    def __init__(self, label, func):
        self.label = label
        self._func = func

    def __call__(self, Ts_c):
        return self._func(Ts_c)


def make_film(cfg, T_ambient_c):
    mode = cfg["mode"]

    if mode == "manual":
        h = float(cfg["h"])
        if h <= 0:
            raise ValueError("Manual film coefficient must be positive.")
        return Film("Manual input", lambda Ts_c: (h, h, 0.0))

    if mode == "radiation":
        eps = float(cfg.get("eps_eff", 0.5))
        hc = float(cfg.get("h_conv", 10.0))

        def func_rad(Ts_c):
            hr = h_radiation(Ts_c, T_ambient_c, eps)
            return hr + hc, hc, hr

        return Film("Radiation plus gas convection", func_rad)

    if mode in ("astm_c680", "iso12241", "simple_air", "calibrated", "juerges"):
        ori = cfg.get("orientation", "V")
        eps = float(cfg.get("emissivity", 0.5))
        wind = float(cfg.get("wind_m_s", 0.0))
        length = float(cfg.get("length_m", 1.0))
        j = 3.0 if ori == "V" else 3.5

        if mode == "juerges":
            if wind > 5.0:
                raise ValueError("Juerges correlation is used here for wind up to 5 m/s.")

            def func_j(Ts_c):
                hc = 5.8 + 3.95 * wind
                hr = h_radiation(Ts_c, T_ambient_c, eps)
                return hc + hr, hc, hr

            return Film(f"{LABELS[mode]}, orientation {ori}", func_j)

        natural = make_natural_calibrated(cfg) if mode == "calibrated" else NATURAL_MODELS[mode]

        def func(Ts_c):
            hn = natural(Ts_c, T_ambient_c, cfg)
            hf = _h_forced(Ts_c, T_ambient_c, wind, length)
            hc = (hn ** j + hf ** j) ** (1.0 / j)
            hr = h_radiation(Ts_c, T_ambient_c, eps)
            return hc + hr, hc, hr

        return Film(f"{LABELS[mode]}, orientation {ori}", func)

    raise ValueError(f"Unknown film mode: {mode}")