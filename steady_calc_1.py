"""
Multilayer wall steady state calculation (solver only).
Writes: wds_steady_data.csv and wds_steady_meta.json
"""

import csv
import json
import numpy as np
import heatrapy as htp

# =============================================================================
# PROJECT INFORMATION (shown at the top of the report)
# =============================================================================
PROJECT = {
    "customer": "Linde",
    "plant": "Drehrohrofen LSM",
    "detail": "Sidewall without heater",
    "name": "Orhan Kibrisli",
}

# =============================================================================
# USER SETTINGS
# =============================================================================
MATERIALS_PATH = (r"C:\Users\orhan.kibrisli\PycharmProjects\heat_transfer\.venv"
                  r"\Lib\site-packages\heatrapy\database\\")

# Order: hot face first, cold face last. Names must exist in the heatrapy database.
LAYERS = [
    {"name": "WDS", "thickness_mm": 150.0, "dx_mm": 10.0, "max_temp_c": 950.0},
]

T_GAS_HOT_C = 900.0
T_GAS_COLD_C = 30.0
H_HOT = 150.0
H_COLD = 5.981

TOLERANCE_K = 1.0e-6
MAX_ITERATIONS = 500
RELAXATION = 0.7

STEADY_CSV = "wds_steady_data.csv"
STEADY_META = "wds_steady_meta.json"

# =============================================================================
# MESH AND MATERIAL LOADING
# =============================================================================
if H_HOT <= 0 or H_COLD <= 0:
    raise ValueError("Film coefficients must be positive.")

T_GAS_HOT = T_GAS_HOT_C + 273.15
T_GAS_COLD = T_GAS_COLD_C + 273.15
T_REF = 0.5 * (T_GAS_HOT + T_GAS_COLD)

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
        T_REF, materials=(lay["name"],), borders=(1, n),
        materials_order=(0,), dx=d, dt=1.0, boundaries=(0, 0),
        materials_path=MATERIALS_PATH, file_name=None, draw=[]
    )
    materials.append(loader.object.materials[0])
    layer_bounds_mm.append((x_start, x_start + lay["thickness_mm"]))
    x_start += lay["thickness_mm"]

seg_dx = np.array(seg_dx)
n_nodes = n_seg + 1
x_mm = np.concatenate(([0.0], np.cumsum(seg_dx))) * 1000.0
L_MM = float(x_mm[-1])

# =============================================================================
# STEADY STATE SOLVER
# =============================================================================


def vec(f, T):
    return np.array([f(float(v)) for v in T], dtype=float)


def hmean(a, b):
    return 2.0 * a * b / (a + b)


def seg_conductance(T):
    G = np.empty(n_seg)
    for mat, idx in zip(materials, seg_idx_by_layer):
        kL, kR = vec(mat.k0, T[idx]), vec(mat.k0, T[idx + 1])
        if np.any(kL <= 0) or np.any(kR <= 0):
            raise ValueError("Material database returned a nonpositive conductivity.")
        G[idx] = hmean(kL, kR) / seg_dx[idx]
    return G


def solve_steady():
    T = np.linspace(T_GAS_HOT, T_GAS_COLD, n_nodes)
    for it in range(1, MAX_ITERATIONS + 1):
        G = seg_conductance(T)
        diag = np.zeros(n_nodes)
        diag[:-1] += G
        diag[1:] += G
        diag[0] += H_HOT
        diag[-1] += H_COLD
        A = np.diag(diag) - np.diag(G, 1) - np.diag(G, -1)
        b = np.zeros(n_nodes)
        b[0] = H_HOT * T_GAS_HOT
        b[-1] = H_COLD * T_GAS_COLD
        candidate = np.linalg.solve(A, b)
        delta = float(np.max(np.abs(candidate - T)))
        if delta < TOLERANCE_K:
            return candidate, it, True, delta
        T = T + RELAXATION * (candidate - T)
    return T, MAX_ITERATIONS, False, delta


print("Running steady state calculation")
T, iterations, converged, last_delta = solve_steady()
G = seg_conductance(T)
TC = T - 273.15

q_hot = H_HOT * (T_GAS_HOT - T[0])
q_cold = H_COLD * (T[-1] - T_GAS_COLD)
q_seg = G * (T[:-1] - T[1:])
imbalance_pct = 100.0 * abs(q_hot - q_cold) / max(abs(q_hot), 1e-12)

# =============================================================================
# LAYER RESULTS
# =============================================================================
layer_results = []
for lay, (xa, xb), idx in zip(LAYERS, layer_bounds_mm, seg_idx_by_layer):
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    t_hot_side, t_cold_side = float(TC[i0]), float(TC[i1])
    thickness_m = (xb - xa) / 1000.0
    t_mean = float(np.sum(seg_dx[idx] * 0.5 * (TC[idx] + TC[idx + 1])) / thickness_m)
    q_layer = float(np.mean(q_seg[idx]))
    dT = t_hot_side - t_cold_side
    k_eff = q_layer * thickness_m / dT if abs(dT) > 1e-12 else float("nan")
    layer_results.append({
        "name": lay["name"], "thickness_mm": lay["thickness_mm"],
        "start_mm": xa, "end_mm": xb, "max_temp_c": lay["max_temp_c"],
        "k_eff": k_eff, "t_mean_c": t_mean,
        "t_hot_side_c": t_hot_side, "t_cold_side_c": t_cold_side,
    })

layer_of_node = []
for i in range(n_nodes):
    j = 0
    for k, (xa, xb) in enumerate(layer_bounds_mm):
        if x_mm[i] >= xa - 1e-9:
            j = k
    layer_of_node.append(LAYERS[j]["name"])

with open(STEADY_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["x_mm", "T_C", "layer"])
    for xv, tv, nm in zip(x_mm, TC, layer_of_node):
        w.writerow([f"{xv:.6f}", f"{tv:.6f}", nm])

meta = {
    "project": PROJECT,
    "layers": layer_results,
    "l_mm": L_MM,
    "t_gas_hot_c": T_GAS_HOT_C,
    "t_gas_cold_c": T_GAS_COLD_C,
    "t_surface_hot_c": float(TC[0]),
    "t_surface_cold_c": float(TC[-1]),
    "h_hot": H_HOT,
    "h_cold": H_COLD,
    "q_hot": float(q_hot),
    "q_cold": float(q_cold),
    "imbalance_pct": float(imbalance_pct),
    "converged": bool(converged),
    "iterations": int(iterations),
    "last_delta_k": float(last_delta),
}
with open(STEADY_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Converged: {converged} after {iterations} iterations")
print(f"Surface temperatures: {TC[0]:.1f} C hot side, {TC[-1]:.1f} C cold side")
print(f"Heat loss: {q_hot:.1f} W/m2 (imbalance {imbalance_pct:.2e} %)")
print(f"Saved {STEADY_CSV} and {STEADY_META}")