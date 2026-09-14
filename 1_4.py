"""
1D transient and steady state conduction through WDS Ultra with convective
Robin boundary conditions. The gas temperatures are not imposed as wall
surface temperatures.

The script uses heatrapy's custom WDS material database for k(T), rho(T), and
Cp(T), but uses a finite volume solver here because heatrapy's SingleObject1D
public boundary interface accepts only fixed temperature or insulated faces.

Model convention, x increases from the hot face to the cold face:
q_hot = h_hot * (T_gas_hot - T_surface_hot)
q_cold = h_cold * (T_surface_cold - T_gas_cold)

All temperatures in the user settings are degrees C. SI units and K are used
internally. Temperature differences in C and K have the same numerical value.
"""

import csv
import numpy as np
import heatrapy as htp

# =============================================================================
# USER SETTINGS
# =============================================================================

# Material database
MATERIAL_NAME = "WDS"
MATERIALS_PATH = (
    r"C:\Users\orhan.kibrisli\PycharmProjects\heat_transfer\.venv"
    r"\Lib\site-packages\heatrapy\database\\"
)
MATERIAL_MAX_TEMP_C = 950.0

# Wall geometry
L_MM = 20.0
DX_MM = 2.0

# Time integration
DT = 50.0
PICARD_TOLERANCE = 1.0e-6
PICARD_MAX_ITERATIONS = 100

# Initial temperature and gas side boundary conditions
T_INITIAL_C = 30.0
T_GAS_HOT_C = 900.0
T_GAS_COLD_C = 30.0
H_HOT = 150.0
H_COLD = 12.09

# Steady state convergence
STEADY_CHUNK_TIME = 5000.0
STEADY_MAX_TIME = 2000000.0
STEADY_TOLERANCE = 1.0e-4

# Transient output
SAMPLE_FRACTIONS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5]
MONITOR_FRACTIONS = {"x=L/4": 0.25, "x=L/2": 0.5, "x=3L/4": 0.75}

# Output files
STEADY_OUTPUT_CSV = "wds_robin_steady_state_profile.csv"
TRANSIENT_MONITOR_CSV = "wds_robin_transient_monitor_points.csv"
TRANSIENT_PROFILES_CSV = "wds_robin_transient_profiles.csv"

# =============================================================================
# UNIT CONVERSION AND MODEL SETUP
# =============================================================================

L = L_MM / 1000.0
DX_REQUESTED = DX_MM / 1000.0
T_INITIAL = T_INITIAL_C + 273.15
T_GAS_HOT = T_GAS_HOT_C + 273.15
T_GAS_COLD = T_GAS_COLD_C + 273.15

if T_GAS_HOT_C > MATERIAL_MAX_TEMP_C:
    print(
        f"WARNING: hot gas temperature {T_GAS_HOT_C:.1f} C exceeds the "
        f"WDS rated maximum continuous use temperature of "
        f"{MATERIAL_MAX_TEMP_C:.1f} C."
    )

if L <= 0.0 or DX_REQUESTED <= 0.0 or DT <= 0.0:
    raise ValueError("L_MM, DX_MM, and DT must be positive.")
if H_HOT <= 0.0 or H_COLD <= 0.0:
    raise ValueError("H_HOT and H_COLD must be positive for Robin boundaries.")

# Nodes are actual material surfaces at x=0 and x=L. End nodes have half the
# control volume of internal nodes, which gives an energy consistent FVM mesh.
n_intervals = int(round(L / DX_REQUESTED))
if n_intervals < 1:
    raise ValueError("Use at least one spatial interval.")
DX = L / n_intervals
n_nodes = n_intervals + 1
x_m = np.linspace(0.0, L, n_nodes)
x_mm = x_m * 1000.0
control_volume_width = np.full(n_nodes, DX)
control_volume_width[0] = DX / 2.0
control_volume_width[-1] = DX / 2.0

# Use heatrapy only to load and evaluate the same custom material functions
# used in the original script. Insulated dummy boundaries avoid imposing an
# unwanted fixed surface temperature in this loader object.
material_loader = htp.SingleObject1D(
    T_INITIAL,
    materials=(MATERIAL_NAME,),
    borders=(1, n_intervals),
    materials_order=(0,),
    dx=DX,
    dt=DT,
    boundaries=(0, 0),
    materials_path=MATERIALS_PATH,
    file_name=None,
    draw=[],
)
material = material_loader.object.materials[0]


def properties(temperature_k):
    """Evaluate heatrapy database properties at each nodal temperature."""
    k = np.array([material.k0(float(T)) for T in temperature_k], dtype=float)
    rho = np.array([material.rho0(float(T)) for T in temperature_k], dtype=float)
    cp = np.array([material.cp0(float(T)) for T in temperature_k], dtype=float)
    if np.any(k <= 0.0) or np.any(rho <= 0.0) or np.any(cp <= 0.0):
        raise ValueError("Material database returned a nonpositive k, rho, or Cp.")
    return k, rho, cp


def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b)


def advance_one_step(T_old, dt):
    """Backward Euler FVM step with Picard iterations for k(T), rho(T), Cp(T)."""
    T_new = T_old.copy()

    for iteration in range(1, PICARD_MAX_ITERATIONS + 1):
        k, rho, cp = properties(T_new)
        k_face = harmonic_mean(k[:-1], k[1:])
        G_face = k_face / DX
        thermal_capacity = rho * cp * control_volume_width / dt

        A = np.zeros((n_nodes, n_nodes), dtype=float)
        b = thermal_capacity * T_old

        # Hot surface control volume. The Robin term is h_hot*(Tgas hot - Ts).
        A[0, 0] = thermal_capacity[0] + H_HOT + G_face[0]
        A[0, 1] = -G_face[0]
        b[0] += H_HOT * T_GAS_HOT

        # Internal full control volumes.
        for i in range(1, n_nodes - 1):
            G_left = G_face[i - 1]
            G_right = G_face[i]
            A[i, i - 1] = -G_left
            A[i, i] = thermal_capacity[i] + G_left + G_right
            A[i, i + 1] = -G_right

        # Cold surface control volume. The Robin term is h_cold*(Ts - Tgas cold).
        A[-1, -2] = -G_face[-1]
        A[-1, -1] = thermal_capacity[-1] + G_face[-1] + H_COLD
        b[-1] += H_COLD * T_GAS_COLD

        T_candidate = np.linalg.solve(A, b)
        if np.max(np.abs(T_candidate - T_new)) < PICARD_TOLERANCE:
            return T_candidate
        T_new = T_candidate

    raise RuntimeError(
        f"Picard iteration failed after {PICARD_MAX_ITERATIONS} iterations. "
        "Reduce DT or increase PICARD_MAX_ITERATIONS."
    )


def advance(T_start, duration):
    """Advance exactly duration seconds, using a shorter last step if needed."""
    T = T_start.copy()
    elapsed = 0.0
    while elapsed < duration - 1.0e-12:
        dt_step = min(DT, duration - elapsed)
        T = advance_one_step(T, dt_step)
        elapsed += dt_step
    return T


def flux_report(T):
    """Return film and conductive heat fluxes, positive hot to cold."""
    k, _, _ = properties(T)
    k_face = harmonic_mean(k[:-1], k[1:])
    q_conduction = -k_face * np.diff(T) / DX
    q_hot_film = H_HOT * (T_GAS_HOT - T[0])
    q_cold_film = H_COLD * (T[-1] - T_GAS_COLD)
    return q_hot_film, q_cold_film, q_conduction


print(f"Material: {MATERIAL_NAME}")
print(
    f"Mesh: {n_nodes} nodes, {n_intervals} intervals, dx = {DX * 1000.0:.4f} mm, "
    f"length = {L * 1000.0:.4f} mm"
)
print(
    f"Robin boundaries: hot gas = {T_GAS_HOT_C:.1f} C, h_hot = {H_HOT:.3f} W/(m2.K); "
    f"cold gas = {T_GAS_COLD_C:.1f} C, h_cold = {H_COLD:.3f} W/(m2.K)"
)

# =============================================================================
# PART 1: STEADY STATE
# =============================================================================

T_steady = np.full(n_nodes, T_INITIAL, dtype=float)
previous_profile = None
total_time = 0.0
converged = False

while total_time < STEADY_MAX_TIME:
    duration = min(STEADY_CHUNK_TIME, STEADY_MAX_TIME - total_time)
    T_steady = advance(T_steady, duration)
    total_time += duration

    if previous_profile is not None:
        max_delta = float(np.max(np.abs(T_steady - previous_profile)))
        if max_delta < STEADY_TOLERANCE:
            print(
                f"Steady state reached after {total_time:.0f} s "
                f"with maximum profile change {max_delta:.2e} K."
            )
            converged = True
            break
    previous_profile = T_steady.copy()

if not converged:
    print(f"WARNING: steady state did not converge within {STEADY_MAX_TIME:.0f} s.")

q_hot, q_cold, q_wall = flux_report(T_steady)
q_wall_mean = float(np.mean(q_wall))
q_wall_spread = float(np.max(q_wall) - np.min(q_wall))

print("\nSteady state results")
print(f"Hot wall surface temperature:  {T_steady[0] - 273.15:.2f} C")
print(f"Cold wall surface temperature: {T_steady[-1] - 273.15:.2f} C")
print(f"Hot film heat flux:            {q_hot:.2f} W/m2")
print(f"Wall heat flux, mean:          {q_wall_mean:.2f} W/m2")
print(f"Cold film heat flux:           {q_cold:.2f} W/m2")
print(f"Wall flux spread:              {q_wall_spread:.5f} W/m2")
print(
    f"Energy mismatch hot to cold:   {abs(q_hot - q_cold):.4f} W/m2 "
    f"({abs(q_hot - q_cold) / max(abs(q_hot), 1.0) * 100.0:.4f} %)"
)

with open(STEADY_OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x_mm", "T_C", "k_W_mK"])
    k_steady, _, _ = properties(T_steady)
    for x, T, k_value in zip(x_mm, T_steady - 273.15, k_steady):
        writer.writerow([x, T, k_value])
print(f"Saved {STEADY_OUTPUT_CSV}")

# =============================================================================
# PART 2: TRANSIENT RESPONSE FROM THE INITIAL CONDITION
# =============================================================================

k_ref, rho_ref, cp_ref = properties(np.full(n_nodes, T_INITIAL))
mid = n_nodes // 2
alpha_ref = k_ref[mid] / (rho_ref[mid] * cp_ref[mid])
char_time = L ** 2 / alpha_ref
print(
    f"\nReference properties at initial temperature: k = {k_ref[mid]:.5f} W/(m.K), "
    f"rho = {rho_ref[mid]:.2f} kg/m3, Cp = {cp_ref[mid]:.2f} J/(kg.K)"
)
print(f"Characteristic diffusion time L2/alpha = {char_time:.1f} s")

sample_times = [fraction * char_time for fraction in SAMPLE_FRACTIONS]
monitor_indices = {
    name: int(np.argmin(np.abs(x_m - fraction * L)))
    for name, fraction in MONITOR_FRACTIONS.items()
}

T_transient = np.full(n_nodes, T_INITIAL, dtype=float)
t_current = 0.0
profile_snapshots = []
monitor_series = []

for target_time in sample_times:
    T_transient = advance(T_transient, target_time - t_current)
    t_current = target_time
    T_C = T_transient - 273.15
    profile_snapshots.append((t_current, T_C.copy()))
    monitor_series.append(
        [t_current, T_C[0], T_C[-1]]
        + [T_C[monitor_indices[name]] for name in MONITOR_FRACTIONS]
    )
    print(f"Transient output at t = {t_current:.1f} s, t/tau = {t_current / char_time:.2f}")

with open(TRANSIENT_MONITOR_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["time_s", "T_hot_surface_C", "T_cold_surface_C"]
        + [f"{name}_C" for name in MONITOR_FRACTIONS]
    )
    writer.writerows(monitor_series)
print(f"Saved {TRANSIENT_MONITOR_CSV}")

with open(TRANSIENT_PROFILES_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x_mm"] + [f"t={t:.1f}s" for t, _ in profile_snapshots])
    for i, x in enumerate(x_mm):
        writer.writerow([x] + [profile[i] for _, profile in profile_snapshots])
print(f"Saved {TRANSIENT_PROFILES_CSV}")

print(
    "\nDone. Compare the gas temperatures, h values, hot and cold wall surface "
    "temperatures, and heat flux with the corresponding SIMU THERM case."
)
