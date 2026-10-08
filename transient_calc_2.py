"""
Multilayer wall transient calculation (solver only).
Writes: wds_transient_data.csv and wds_transient_meta.json
Film coefficients come from film_models.py (same modes as the steady script).
The films are recalculated at every Picard iteration from the current surface
temperatures, so they follow the wall while it heats up.
"""

import csv
import json
import numpy as np
import heatrapy as htp
from film_models import make_film

# =============================================================================
# USER SETTINGS
# =============================================================================
MATERIALS_PATH = (r"C:\Users\orhan.kibrisli\PycharmProjects\heat_transfer\.venv"
                  r"\Lib\site-packages\heatrapy\database\\")

# Order: hot face first, cold face last. Names must exist in the heatrapy database.
LAYERS = [
    {"name": "WDS", "thickness_mm": 150.0, "dx_mm": 10.0, "max_temp_c": 950.0},
    {"name": "Al",  "thickness_mm": 50.0,  "dx_mm": 5.0,  "max_temp_c": 600.0},
    {"name": "Al",  "thickness_mm": 50.0,  "dx_mm": 5.0,  "max_temp_c": 600.0},
]

DT = 50.0
T_INITIAL_C = 30.0
T_GAS_HOT_C = 2000.0
T_GAS_COLD_C = 30.0

# ---- Inside (hot) film: choose one -----------------------------------------
INSIDE_FILM = {"mode": "manual", "h": 150.0}
# INSIDE_FILM = {"mode": "radiation", "eps_eff": 0.5, "h_conv": 10.0}

# ---- Outside (cold) film: choose one mode ----------------------------------
# modes: astm_c680, iso12241, simple_air, calibrated, juerges, manual
# orientation: V = vertical, R = ceiling (heat flow up), F = floor (heat flow down)
OUTSIDE_FILM = {"mode": "astm_c680", "orientation": "V", "emissivity": 0.5,
                "wind_m_s": 0.0, "height_m": 1.0, "length_m": 1.0}
# OUTSIDE_FILM = {"mode": "iso12241", "orientation": "V", "emissivity": 0.5, "wind_m_s": 0.0, "height_m": 1.0}
# OUTSIDE_FILM = {"mode": "simple_air", "orientation": "F", "emissivity": 0.5, "wind_m_s": 0.0, "length_m": 1.0}
# OUTSIDE_FILM = {"mode": "calibrated", "orientation": "V", "emissivity": 0.5}
# OUTSIDE_FILM = {"mode": "juerges", "orientation": "V", "emissivity": 0.5, "wind_m_s": 2.0}
# OUTSIDE_FILM = {"mode": "manual", "h": 12.09}

TRANSIENT_TOTAL_TIME_S = 50000.0
TRANSIENT_OUTPUT_INTERVAL_S = 60.0
PICARD_TOLERANCE_K = 1.0e-6
PICARD_MAX_ITERATIONS = 100

TRANSIENT_CSV = "wds_transient_data.csv"
TRANSIENT_META = "wds_transient_meta.json"

# =============================================================================
# MESH AND MATERIAL LOADING
# =============================================================================
if DT <= 0:
    raise ValueError("Time step must be positive.")

film_hot = make_film(INSIDE_FILM, T_GAS_HOT_C)
film_cold = make_film(OUTSIDE_FILM, T_GAS_COLD_C)

MATERIAL_NAME = " + ".join(l["name"] for l in LAYERS)
T_INITIAL = T_INITIAL_C + 273.15
T_GAS_HOT = T_GAS_HOT_C + 273.15
T_GAS_COLD = T_GAS_COLD_C + 273.15

seg_dx, seg_idx_by_layer, materials, layer_bounds_mm = [], [], [], []
n_seg = 0
x_start = 0.0
for lay in LAYERS:
    if lay["thickness_mm"] <= 0 or lay["dx_mm"] <= 0:
        raise ValueError(f"Layer {lay['name']} needs positive thickness and dx.")
    n = max(1, int(round(lay["thickness_mm"] / lay["dx_mm"])))
    d = lay["thickness_mm"] / 1000.0 / n
    seg_dx += [d] * n
    seg_idx_by_layer.append(np.arange(n_seg, n_seg + n))
    n_seg += n
    loader = htp.SingleObject1D(
        T_INITIAL, materials=(lay["name"],), borders=(1, n),
        materials_order=(0,), dx=d, dt=DT, boundaries=(0, 0),
        materials_path=MATERIALS_PATH, file_name=None, draw=[]
    )
    materials.append(loader.object.materials[0])
    layer_bounds_mm.append((x_start, x_start + lay["thickness_mm"]))
    x_start += lay["thickness_mm"]

seg_dx = np.array(seg_dx)
n_nodes = n_seg + 1
x_m = np.concatenate(([0.0], np.cumsum(seg_dx)))
x_mm = x_m * 1000.0
L_MM = float(x_mm[-1])

# =============================================================================
# FINITE VOLUME SOLVER
# =============================================================================


def vec(f, T):
    return np.array([f(float(v)) for v in T], dtype=float)


def hmean(a, b):
    return 2.0 * a * b / (a + b)


def seg_props(T):
    """Segment conductance G (W/m2K) and node heat capacity per area (J/m2K)."""
    G = np.empty(n_seg)
    Cn = np.zeros(n_nodes)
    for mat, idx in zip(materials, seg_idx_by_layer):
        TL, TR = T[idx], T[idx + 1]
        kL, kR = vec(mat.k0, TL), vec(mat.k0, TR)
        rcL = vec(mat.rho0, TL) * vec(mat.cp0, TL)
        rcR = vec(mat.rho0, TR) * vec(mat.cp0, TR)
        if np.any(kL <= 0) or np.any(kR <= 0) or np.any(rcL <= 0) or np.any(rcR <= 0):
            raise ValueError("Material database returned a nonpositive property.")
        G[idx] = hmean(kL, kR) / seg_dx[idx]
        Cn[idx] += 0.5 * rcL * seg_dx[idx]
        Cn[idx + 1] += 0.5 * rcR * seg_dx[idx]
    return G, Cn


def films(T):
    """Inside and outside film coefficients for the current surface temperatures."""
    return film_hot(T[0] - 273.15)[0], film_cold(T[-1] - 273.15)[0]


def solve_step(T_old, dt):
    T = T_old.copy()
    for _ in range(PICARD_MAX_ITERATIONS):
        G, Cn = seg_props(T)
        h_hot, h_cold = films(T)
        C = Cn / dt
        diag = C.copy()
        diag[:-1] += G
        diag[1:] += G
        diag[0] += h_hot
        diag[-1] += h_cold
        A = np.diag(diag) - np.diag(G, 1) - np.diag(G, -1)
        b = C * T_old
        b[0] += h_hot * T_GAS_HOT
        b[-1] += h_cold * T_GAS_COLD
        candidate = np.linalg.solve(A, b)
        if np.max(np.abs(candidate - T)) < PICARD_TOLERANCE_K:
            return candidate
        T = candidate
    raise RuntimeError("Picard iteration failed. Reduce DT.")


def fluxes(T):
    G, _ = seg_props(T)
    h_hot, h_cold = films(T)
    qwall = G * (T[:-1] - T[1:])
    qhot = h_hot * (T_GAS_HOT - T[0])
    qcold = h_cold * (T[-1] - T_GAS_COLD)
    return qhot, qcold, float(np.mean(qwall)), h_hot, h_cold


# =============================================================================
# TRANSIENT RUN
# =============================================================================
print(f"Layers: {MATERIAL_NAME}")
print(f"Mesh: {n_nodes} nodes, total thickness {L_MM:.2f} mm")
print(f"Inside film : {film_hot.label}")
print(f"Outside film: {film_cold.label}")
print("Running transient calculation")

T = np.full(n_nodes, T_INITIAL)
t = 0.0
next_save = 0.0
Times, Profiles, Qhot, Qcold, Qwall, Hhot, Hcold = [], [], [], [], [], [], []


def save_state(T, t):
    qh, qc, qw, hh, hc = fluxes(T)
    Times.append(t)
    Profiles.append((T - 273.15).copy())
    Qhot.append(qh); Qcold.append(qc); Qwall.append(qw)
    Hhot.append(hh); Hcold.append(hc)


while t < TRANSIENT_TOTAL_TIME_S - 1.0e-10:
    if t >= next_save - 1.0e-10:
        save_state(T, t)
        next_save += TRANSIENT_OUTPUT_INTERVAL_S
    else:
        dtime = min(DT, next_save - t, TRANSIENT_TOTAL_TIME_S - t)
        T = solve_step(T, dtime)
        t += dtime

if abs(Times[-1] - TRANSIENT_TOTAL_TIME_S) > 1.0e-8:
    save_state(T, TRANSIENT_TOTAL_TIME_S)

Times, Profiles = np.asarray(Times), np.asarray(Profiles)
Qhot, Qcold, Qwall = np.asarray(Qhot), np.asarray(Qcold), np.asarray(Qwall)
Hhot, Hcold = np.asarray(Hhot), np.asarray(Hcold)

with open(TRANSIENT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["time_s", "q_hot_W_m2", "q_cold_W_m2", "q_wall_W_m2", "h_hot_W_m2K", "h_cold_W_m2K"]
               + [f"T_x_{x:.3f}_mm_C" for x in x_mm])
    for i in range(len(Times)):
        w.writerow([Times[i], Qhot[i], Qcold[i], Qwall[i], Hhot[i], Hcold[i], *Profiles[i]])

meta = {
    "material_name": MATERIAL_NAME,
    "layers": [
        {"name": l["name"], "start_mm": b[0], "end_mm": b[1], "max_temp_c": l["max_temp_c"]}
        for l, b in zip(LAYERS, layer_bounds_mm)
    ],
    "l_mm": L_MM,
    "dt_s": DT,
    "t_initial_c": T_INITIAL_C,
    "t_gas_hot_c": T_GAS_HOT_C,
    "t_gas_cold_c": T_GAS_COLD_C,
    "film_model_hot": film_hot.label,
    "film_model_cold": film_cold.label,
    "h_hot_start": float(Hhot[0]),
    "h_hot_end": float(Hhot[-1]),
    "h_cold_start": float(Hcold[0]),
    "h_cold_end": float(Hcold[-1]),
    "orientation": OUTSIDE_FILM.get("orientation", ""),
    "emissivity": OUTSIDE_FILM.get("emissivity", None),
    "wind": OUTSIDE_FILM.get("wind_m_s", None),
    "total_time_s": TRANSIENT_TOTAL_TIME_S,
    "output_interval_s": TRANSIENT_OUTPUT_INTERVAL_S,
}
with open(TRANSIENT_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Outside film: {Hcold[0]:.2f} W/m2K at start, {Hcold[-1]:.2f} W/m2K at end")
print(f"Saved {TRANSIENT_CSV} and {TRANSIENT_META}")
