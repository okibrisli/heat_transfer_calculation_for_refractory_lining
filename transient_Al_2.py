"""
Transient heat conduction using heatrapy (single-layer slab, step boundary change)
====================================================================================

This extends the steady-state script to a genuine TRANSIENT case: at t=0 the
slab is uniformly at an initial (ambient) temperature, and the hot face is
suddenly raised to T_hot while the cold face is held at T_cold - similar to a
kiln being lit from cold. We then track how the temperature field evolves
over time, both at fixed monitoring points (like thermocouples embedded in
the wall) and as full spatial snapshots.

Two independent checks are used to validate the heatrapy result:

1. A closed-form analytical solution (Fourier sine series) for 1D transient
   conduction in a finite slab with fixed, DIFFERENT temperatures on each
   face and a uniform initial temperature. This assumes CONSTANT material
   properties (k, rho, Cp independent of temperature), so we use the
   'implicit_general' solver in heatrapy to match that assumption exactly.

2. The long-time limit of the transient run must converge to the same
   linear steady-state profile validated in the previous steady-state script.

NOTE ON THE DOMAIN-LENGTH FIX: the previous script had an off-by-one that
made the real domain 0.202 m instead of the intended 0.20 m. Here
`borders=(1, n_points)` is used (not `n_points + 1`), which gives exactly
`n_points + 1` nodes spanning `dx * n_points == L`.
"""

import csv
import numpy as np
from scipy.integrate import quad
import heatrapy as htp

# ---------------------------------------------------------------------
# 1. Pick a built-in material (same fallback logic as the steady-state script)
# ---------------------------------------------------------------------
candidate_materials = [
    "Al2O3", "alumina", "MgO", "SiO2", "steel", "Fe", "iron",
    "brick", "concrete", "Al", "Cu",
]

chosen_material = None
for name in candidate_materials:
    try:
        test_obj = htp.SingleObject1D(
            293, materials=(name,), borders=(1, 2), materials_order=(0,),
            dx=0.01, dt=0.1, boundaries=(0, 0), draw=[],
        )
        chosen_material = name
        del test_obj
        break
    except Exception:
        continue

if chosen_material is None:
    raise RuntimeError(
        "No candidate material found. List what's actually installed with:\n"
        "  import heatrapy, os\n"
        "  print(os.listdir(os.path.dirname(heatrapy.__file__) + '/database'))"
    )

print(f"Using built-in material: '{chosen_material}'")

# ---------------------------------------------------------------------
# 2. Geometry, initial condition, and boundary conditions
# ---------------------------------------------------------------------
L = 0.20            # slab thickness [m]
dx = 0.002          # spatial step [m]
dt = 0.05           # time step [s]

T_initial = 293.0   # uniform initial temperature [K] (cold start)
T_hot = 1273.0      # hot face temperature imposed at t=0+ [K]
T_cold = 323.0      # cold face temperature imposed at t=0+ [K]

n_points = int(round(L / dx))   # FIXED: gives exactly n_points+1 nodes over L

wall = htp.SingleObject1D(
    T_initial,
    materials=(chosen_material,),
    borders=(1, n_points),
    materials_order=(0,),
    dx=dx,
    dt=dt,
    boundaries=(T_hot, T_cold),
    file_name=None,       # do NOT let heatrapy write every step (huge files)
    draw=[],
)

n_nodes = len(wall.object.temperature)
x_positions = [i * dx for i in range(n_nodes)]
print(f"Domain: {n_nodes} nodes, length = {(n_nodes - 1) * dx:.4f} m "
      f"(target was {L} m)")

# Reference material properties at the initial temperature, for the
# analytical Fourier solution (assumes constant properties)
mid = n_nodes // 2
k_ref = wall.object.k[mid]
rho_ref = wall.object.rho[mid]
cp_ref = wall.object.Cp[mid]
alpha_ref = k_ref / (rho_ref * cp_ref)   # thermal diffusivity [m^2/s]

print(f"Reference properties @ initial T: k={k_ref:.3f} W/(m.K), "
      f"rho={rho_ref:.1f} kg/m^3, Cp={cp_ref:.1f} J/(kg.K), "
      f"alpha={alpha_ref:.3e} m^2/s")

char_time = L**2 / alpha_ref
print(f"Characteristic diffusion time L^2/alpha = {char_time:.1f} s")

# ---------------------------------------------------------------------
# 3. Analytical transient solution (Fourier sine series), constant properties
# ---------------------------------------------------------------------
def T_steady(x):
    return T_hot + (T_cold - T_hot) * x / L

n_terms = 200

def b_n_integrand(x, n):
    return (T_initial - T_steady(x)) * np.sin(n * np.pi * x / L)

b_n = []
for n in range(1, n_terms + 1):
    val, _ = quad(b_n_integrand, 0, L, args=(n,))
    b_n.append(2.0 / L * val)

def T_analytical(x, t):
    total = T_steady(x)
    for n in range(1, n_terms + 1):
        lam = n * np.pi / L
        total += b_n[n - 1] * np.sin(lam * x) * np.exp(-alpha_ref * lam**2 * t)
    return total

# ---------------------------------------------------------------------
# 4. Run the transient simulation, sampling at chosen times
# ---------------------------------------------------------------------
# Sample times chosen relative to the characteristic diffusion time so the
# run captures early transient, mid-transient, and near-steady behaviour.
sample_fractions = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5]
sample_times = [f * char_time for f in sample_fractions]

monitor_fracs = {"x=L/4": 0.25, "x=L/2": 0.5, "x=3L/4": 0.75}
monitor_indices = {
    name: min(range(n_nodes), key=lambda i: abs(x_positions[i] - frac * L))
    for name, frac in monitor_fracs.items()
}

profile_snapshots = []   # list of (t, [temperatures])
monitor_series = []      # list of (t, T_L4, T_L2, T_3L4)

t_current = 0.0
for t_target in sample_times:
    chunk = t_target - t_current
    if chunk > 0:
        wall.compute(chunk, write_interval=10**9, solver="implicit_general",
                     verbose=False)
        t_current = t_target

    profile = [pt[0] for pt in wall.object.temperature]
    profile_snapshots.append((t_current, profile))

    row = (t_current,
           profile[monitor_indices["x=L/4"]],
           profile[monitor_indices["x=L/2"]],
           profile[monitor_indices["x=3L/4"]])
    monitor_series.append(row)

    max_err = max(
        abs(profile[i] - T_analytical(x_positions[i], t_current))
        for i in range(n_nodes)
    )
    print(f"t={t_current:8.1f} s  (t/tau={t_current/char_time:.2f})  "
          f"max deviation vs analytical = {max_err:.3f} K")

# ---------------------------------------------------------------------
# 5. Save results for comparison against SIMU-THERM's transient run
# ---------------------------------------------------------------------
with open("old/transient_monitor_points.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "T_x_L4_K", "T_x_L2_K", "T_x_3L4_K"])
    writer.writerows(monitor_series)

with open("old/transient_profiles.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["x_m"] + [f"t={t:.1f}s" for t, _ in profile_snapshots]
    writer.writerow(header)
    for i, x in enumerate(x_positions):
        row = [x] + [profile[i] for _, profile in profile_snapshots]
        writer.writerow(row)

print("\nSaved transient_monitor_points.csv (thermocouple-style time series)")
print("Saved transient_profiles.csv (full spatial profile at each sample time)")
print("\nFor SIMU-THERM comparison, replicate: L=0.20 m, T_initial=293 K, "
      "T_hot=1273 K, T_cold=323 K, and the same material properties "
      f"(k={k_ref:.3f}, rho={rho_ref:.1f}, Cp={cp_ref:.1f}), "
      "then compare temperature at the same times and monitor positions.")
