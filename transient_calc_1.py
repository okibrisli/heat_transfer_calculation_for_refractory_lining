"""
Multilayer wall transient calculation (solver only).
Writes: wds_transient_data.csv and wds_transient_meta.json
"""

import csv
import json
import numpy as np
import heatrapy as htp

# =============================================================================
# USER SETTINGS
# =============================================================================
MATERIALS_PATH = (r"C:\Users\orhan.kibrisli\PycharmProjects\heat_transfer\.venv"
                  r"\Lib\site-packages\heatrapy\database\\")

# Order: hot face first, cold face last. Names must exist in the heatrapy database.
LAYERS = [
    {"name": "WDS", "thickness_mm": 150.0, "dx_mm": 10.0, "max_temp_c": 950.0},
    {"name": "Al", "thickness_mm": 50.0,  "dx_mm": 5.0,  "max_temp_c": 600.0},
    {"name": "Al", "thickness_mm": 50.0,  "dx_mm": 5.0,  "max_temp_c": 600.0},
]

DT = 50.0
T_INITIAL_C = 30.0
T_GAS_HOT_C = 2000.0
T_GAS_COLD_C = 30.0
H_HOT = 150.0
H_COLD = 12.09

TRANSIENT_TOTAL_TIME_S = 50000.0
TRANSIENT_OUTPUT_INTERVAL_S = 60.0
PICARD_TOLERANCE_K = 1.0e-6
PICARD_MAX_ITERATIONS = 100

TRANSIENT_CSV = "wds_transient_data.csv"
TRANSIENT_META = "wds_transient_meta.json"

# =============================================================================
# MESH AND MATERIAL LOADING
# =============================================================================
if DT <= 0 or H_HOT <= 0 or H_COLD <= 0:
    raise ValueError("Time step and film coefficients must be positive.")

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


def solve_step(T_old, dt):
    T = T_old.copy()
    for _ in range(PICARD_MAX_ITERATIONS):
        G, Cn = seg_props(T)
        C = Cn / dt
        diag = C.copy()
        diag[:-1] += G
        diag[1:] += G
        diag[0] += H_HOT
        diag[-1] += H_COLD
        A = np.diag(diag) - np.diag(G, 1) - np.diag(G, -1)
        b = C * T_old
        b[0] += H_HOT * T_GAS_HOT
        b[-1] += H_COLD * T_GAS_COLD
        candidate = np.linalg.solve(A, b)
        if np.max(np.abs(candidate - T)) < PICARD_TOLERANCE_K:
            return candidate
        T = candidate
    raise RuntimeError("Picard iteration failed. Reduce DT.")


def fluxes(T):
    G, _ = seg_props(T)
    qwall = G * (T[:-1] - T[1:])
    qhot = H_HOT * (T_GAS_HOT - T[0])
    qcold = H_COLD * (T[-1] - T_GAS_COLD)
    return qhot, qcold, float(np.mean(qwall))

# =============================================================================
# TRANSIENT RUN
# =============================================================================
print(f"Layers: {MATERIAL_NAME}")
print(f"Mesh: {n_nodes} nodes, total thickness {L_MM:.2f} mm")
print("Running transient calculation")

T = np.full(n_nodes, T_INITIAL)
t = 0.0
next_save = 0.0
Times, Profiles, Qhot, Qcold, Qwall = [], [], [], [], []

while t < TRANSIENT_TOTAL_TIME_S - 1.0e-10:
    if t >= next_save - 1.0e-10:
        qh, qc, qw = fluxes(T)
        Times.append(t)
        Profiles.append((T - 273.15).copy())
        Qhot.append(qh); Qcold.append(qc); Qwall.append(qw)
        next_save += TRANSIENT_OUTPUT_INTERVAL_S
    else:
        dtime = min(DT, next_save - t, TRANSIENT_TOTAL_TIME_S - t)
        T = solve_step(T, dtime)
        t += dtime

if abs(Times[-1] - TRANSIENT_TOTAL_TIME_S) > 1.0e-8:
    qh, qc, qw = fluxes(T)
    Times.append(TRANSIENT_TOTAL_TIME_S)
    Profiles.append((T - 273.15).copy())
    Qhot.append(qh); Qcold.append(qc); Qwall.append(qw)

Times, Profiles = np.asarray(Times), np.asarray(Profiles)
Qhot, Qcold, Qwall = np.asarray(Qhot), np.asarray(Qcold), np.asarray(Qwall)

with open(TRANSIENT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["time_s", "q_hot_W_m2", "q_cold_W_m2", "q_wall_W_m2"]
               + [f"T_x_{x:.3f}_mm_C" for x in x_mm])
    for i in range(len(Times)):
        w.writerow([Times[i], Qhot[i], Qcold[i], Qwall[i], *Profiles[i]])

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
    "h_hot": H_HOT,
    "h_cold": H_COLD,
    "total_time_s": TRANSIENT_TOTAL_TIME_S,
    "output_interval_s": TRANSIENT_OUTPUT_INTERVAL_S,
}
with open(TRANSIENT_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Saved {TRANSIENT_CSV} and {TRANSIENT_META}")