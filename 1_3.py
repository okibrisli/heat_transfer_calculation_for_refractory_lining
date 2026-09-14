"""
Heat conduction test using heatrapy with the custom WDS (Porextherm WDS Ultra)
material - steady-state check + transient step-response run.
================================================================================

ALL values you might want to change are grouped in the "USER SETTINGS" block
right below, using mm for length and degrees C for temperature (the units on
your material datasheet / SIMU-THERM), so you don't have to mentally convert
anything. The script converts everything to SI units (meters, Kelvin)
internally, since that's what heatrapy expects, then converts results back
to mm/degC for the printed output and saved CSVs.
"""

import os
import csv
import numpy as np
import heatrapy as htp

# =============================================================================
# USER SETTINGS - edit these, nothing else (lengths in mm, temperatures in C)
# =============================================================================

# --- Material ---------------------------------------------------------------
MATERIAL_NAME = "WDS"
# Folder name (must match exactly) inside MATERIALS_PATH containing
# k0.txt, ka.txt, cp0.txt, cpa.txt, rho0.txt, rhoa.txt, tadd.txt, tadi.txt,
# lheat0.txt, lheata.txt - the files we built from the Porextherm WDS Ultra
# datasheet (k(T), Cp(T), density, in Kelvin / W/m.K / J/kg.K).

MATERIALS_PATH = r"C:\Users\orhan.kibrisli\PycharmProjects\heat_transfer\.venv\Lib\site-packages\heatrapy\database\\"
# Folder that CONTAINS the "WDS" material folder (i.e. WDS must be a
# subfolder of this path). If you installed the WDS folder directly inside
# heatrapy's own built-in database folder, this should point there. If you
# put it somewhere else (e.g. your project folder), change this path
# accordingly - it must be the PARENT of the WDS folder, not the WDS folder
# itself.

MATERIAL_MAX_TEMP_C = 950.0
# Manufacturer-rated maximum continuous use temperature (deg C) from the
# Porextherm datasheet. The script will just WARN you if T_HOT_C exceeds
# this - it will not stop the calculation, since briefly testing beyond the
# rated limit can still be informative, but you should know if you're doing
# it.

# --- Geometry (millimeters) ----------------------------------------------
L_MM = 20.0        # slab thickness [mm]
DX_MM = 2.0         # spatial step [mm] -> smaller = finer mesh, slower run

# --- Time step (seconds - no alternate unit needed here) ------------------
DT = 50             # time step [s] -> smaller = more accurate, slower run

# --- Boundary and initial conditions (degrees Celsius) ---------------------
T_INITIAL_C = 30.0    # uniform starting temperature of the whole slab [C]
T_HOT_C = 900.0       # hot-face temperature held from t=0 onward [C]
T_COLD_C = 30.0       # cold-face temperature held from t=0 onward [C]

# --- Solver ---------------------------------------------------------------
SOLVER = "implicit_k(x)"
# WDS's conductivity varies more than 2x across its usable range (0.018 to
# 0.040 W/m.K), so we use the solver that lets k vary with local temperature
# and position, rather than 'implicit_general' (constant k) used in the
# earlier Al tests. This is more physically realistic for this material.

# --- Steady-state convergence settings ----------------------------------
STEADY_CHUNK_TIME = 5000.0      # seconds simulated per convergence check
STEADY_MAX_TIME = 2000000.0     # safety cap on total simulated time
STEADY_TOLERANCE = 1e-4         # K, max allowed change between checks
                                 # (kept in K here since it's a tiny numerical
                                 # convergence threshold, not a physical
                                 # quantity you need to think in C for)

# --- Transient sampling settings -----------------------------------------
# Sample times are given as fractions of the characteristic diffusion time
# (L^2 / alpha), computed automatically below, so early/mid/late transient
# behaviour is captured regardless of the material's actual diffusivity.
SAMPLE_FRACTIONS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5]

# Positions (as fraction of L) where "virtual thermocouples" record
# temperature over time in the transient run.
MONITOR_FRACTIONS = {"x=L/4": 0.25, "x=L/2": 0.5, "x=3L/4": 0.75}

# --- Output files ---------------------------------------------------------
STEADY_OUTPUT_CSV = "wds_steady_state_profile.csv"
TRANSIENT_MONITOR_CSV = "wds_transient_monitor_points.csv"
TRANSIENT_PROFILES_CSV = "wds_transient_profiles.csv"

# =============================================================================
# UNIT CONVERSION - mm -> m, degC -> K. Everything below this line works in
# SI units internally (what heatrapy expects), then converts back to mm/degC
# only when printing or writing CSVs.
# =============================================================================
L = L_MM / 1000.0
DX = DX_MM / 1000.0
T_INITIAL = T_INITIAL_C + 273.15
T_HOT = T_HOT_C + 273.15
T_COLD = T_COLD_C + 273.15

# =============================================================================
# END OF USER SETTINGS - logic below reads only the variables set above
# =============================================================================

if T_HOT_C > MATERIAL_MAX_TEMP_C:
    print(f"WARNING: T_HOT_C ({T_HOT_C:.1f} C) exceeds WDS's rated max "
          f"continuous use temperature ({MATERIAL_MAX_TEMP_C:.1f} C). "
          f"Results beyond this point are outside the manufacturer's "
          f"validated range.")

n_points = int(round(L / DX))

wall = htp.SingleObject1D(
    T_INITIAL,
    materials=(MATERIAL_NAME,),
    borders=(1, n_points),
    materials_order=(0,),
    dx=DX,
    dt=DT,
    boundaries=(T_HOT, T_COLD),
    materials_path=MATERIALS_PATH,
    file_name=None,
    draw=[],
)

n_nodes = len(wall.object.temperature)
x_positions_mm = [i * DX_MM for i in range(n_nodes)]
actual_length_mm = (n_nodes - 1) * DX_MM
print(f"Material: {MATERIAL_NAME}")
print(f"Domain: {n_nodes} nodes, length = {actual_length_mm:.2f} mm "
      f"(target was {L_MM:.1f} mm)")

# ---------------------------------------------------------------------
# PART 1: steady state (run transient to convergence)
# ---------------------------------------------------------------------
previous_profile = None
total_time = 0.0
converged = False

while total_time < STEADY_MAX_TIME:
    wall.compute(STEADY_CHUNK_TIME, write_interval=10**9, solver=SOLVER,
                 verbose=False)
    total_time += STEADY_CHUNK_TIME
    current_profile = [pt[0] for pt in wall.object.temperature]

    if previous_profile is not None:
        max_delta = max(abs(a - b) for a, b in
                         zip(current_profile, previous_profile))
        if max_delta < STEADY_TOLERANCE:
            print(f"Steady state reached after {total_time:.0f} s "
                  f"(max change {max_delta:.2e} K)")
            converged = True
            break

    previous_profile = current_profile

if not converged:
    print(f"WARNING: did not converge within {STEADY_MAX_TIME:.0f} s")

steady_profile_K = [pt[0] for pt in wall.object.temperature]
steady_profile_C = [T - 273.15 for T in steady_profile_K]

# Validity check for spatially-varying k: in steady state with no internal
# heat generation, the heat flux q = -k(T)*dT/dx must be CONSTANT along the
# whole wall (energy conservation), even though k itself varies with
# position/temperature. This is the correct check to use here instead of
# assuming a linear temperature profile (which only holds for constant k).
# NOTE: dT/dx here uses DX in METERS (SI), since k is in W/(m.K) - mixing in
# mm here would silently give a flux 1000x too large.
fluxes = []
for i in range(n_nodes - 1):
    k_local = 0.5 * (wall.object.k[i] + wall.object.k[i + 1])
    q = -k_local * (steady_profile_K[i + 1] - steady_profile_K[i]) / DX
    fluxes.append(q)

flux_mean = float(np.mean(fluxes))
flux_spread = max(fluxes) - min(fluxes)
print(f"Steady-state heat flux: mean = {flux_mean:.2f} W/m^2, "
      f"spread across wall = {flux_spread:.4f} W/m^2 "
      f"(should be near zero if truly converged)")

with open(STEADY_OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x_mm", "T_C"])
    for x_mm, T_C in zip(x_positions_mm, steady_profile_C):
        writer.writerow([x_mm, T_C])
print(f"Saved {STEADY_OUTPUT_CSV} (x in mm, T in degC)")

# ---------------------------------------------------------------------
# PART 2: transient run, resetting to the initial condition
# ---------------------------------------------------------------------
wall_t = htp.SingleObject1D(
    T_INITIAL,
    materials=(MATERIAL_NAME,),
    borders=(1, n_points),
    materials_order=(0,),
    dx=DX,
    dt=DT,
    boundaries=(T_HOT, T_COLD),
    materials_path=MATERIALS_PATH,
    file_name=None,
    draw=[],
)

mid = n_nodes // 2
k_ref = wall_t.object.k[mid]
rho_ref = wall_t.object.rho[mid]
cp_ref = wall_t.object.Cp[mid]
alpha_ref = k_ref / (rho_ref * cp_ref)
char_time = L**2 / alpha_ref   # uses L in meters - alpha is in m^2/s
print(f"\nReference properties @ initial T: k={k_ref:.4f} W/(m.K), "
      f"rho={rho_ref:.1f} kg/m^3, Cp={cp_ref:.1f} J/(kg.K)")
print(f"Characteristic diffusion time L^2/alpha = {char_time:.1f} s")

sample_times = [f * char_time for f in SAMPLE_FRACTIONS]
monitor_indices = {
    name: min(range(n_nodes), key=lambda i: abs(x_positions_mm[i] - frac * L_MM))
    for name, frac in MONITOR_FRACTIONS.items()
}

profile_snapshots = []   # (t, [temperatures in degC])
monitor_series = []      # (t, T_L4_C, T_L2_C, T_3L4_C)
t_current = 0.0

for t_target in sample_times:
    chunk = t_target - t_current
    if chunk > 0:
        wall_t.compute(chunk, write_interval=10**9, solver=SOLVER,
                       verbose=False)
        t_current = t_target

    profile_K = [pt[0] for pt in wall_t.object.temperature]
    profile_C = [T - 273.15 for T in profile_K]
    profile_snapshots.append((t_current, profile_C))

    row = [t_current] + [profile_C[monitor_indices[name]]
                          for name in MONITOR_FRACTIONS]
    monitor_series.append(row)
    print(f"t={t_current:8.1f} s  (t/tau={t_current/char_time:.2f})")

with open(TRANSIENT_MONITOR_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s"] + [f"{name}_C" for name in MONITOR_FRACTIONS])
    writer.writerows(monitor_series)
print(f"Saved {TRANSIENT_MONITOR_CSV} (temperatures in degC)")

with open(TRANSIENT_PROFILES_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["x_mm"] + [f"t={t:.1f}s" for t, _ in profile_snapshots]
    writer.writerow(header)
    for i, x_mm in enumerate(x_positions_mm):
        row = [x_mm] + [profile_C[i] for _, profile_C in profile_snapshots]
        writer.writerow(row)
print(f"Saved {TRANSIENT_PROFILES_CSV} (x in mm, T in degC)")

print("\nDone. Compare wds_steady_state_profile.csv and "
      "wds_transient_monitor_points.csv against the equivalent SIMU-THERM "
      "run using the same L_MM, T_INITIAL_C, T_HOT_C, T_COLD_C values above.")