"""
Steady-state heat conduction check using heatrapy (single-layer slab)
======================================================================

heatrapy has NO dedicated steady-state solver - it is a transient
(finite-difference) solver only. To get a "steady-state" result we run
the transient simulation with fixed (Dirichlet) hot-face / cold-face
temperatures until the temperature profile stops changing in time.
Once converged, the result IS the steady-state solution, and can be
compared directly against a SIMU-THERM steady-state run with the same
geometry, material, and boundary temperatures.

This first script intentionally keeps things simple:
  - ONE homogeneous layer (no multi-layer stack yet)
  - constant material properties (k, rho, Cp) -> solver = 'implicit_general'
  - fixed temperature boundary conditions on both faces
  - no internal heat generation

Once you've validated this against SIMU-THERM, we expand to:
  - multiple layers (SystemObjects1D, one object per refractory course)
  - your own custom material data (via materials_path)
  - k(T)-dependent conduction (solver = 'implicit_k(x)')
  - true transient runs (varying boundary conditions in time)
"""

import os
import csv
import heatrapy as htp

# ---------------------------------------------------------------------
# 1. Pick a built-in material as close to a dense refractory as possible
# ---------------------------------------------------------------------
# heatrapy's built-in database is small and was originally built for
# caloric-refrigeration research, so there is no guarantee a true
# refractory oxide (Al2O3, MgO, SiO2, firebrick, ...) is included.
# This tries several candidate names, from "most refractory-like" to
# "safe fallback", and uses whichever one actually loads on your
# installation. Check the printed message to see which one was used.

candidate_materials = [
    "Al2O3", "alumina", "MgO", "SiO2", "steel", "Fe", "iron",
    "brick", "concrete", "Al", "Cu",
]

chosen_material = None
for name in candidate_materials:
    try:
        test_obj = htp.SingleObject1D(
            293,
            materials=(name,),
            borders=(1, 3),
            materials_order=(0,),
            dx=0.01,
            dt=0.1,
            boundaries=(0, 0),
            draw=[],
        )
        chosen_material = name
        del test_obj
        break
    except Exception:
        continue

if chosen_material is None:
    raise RuntimeError(
        "None of the candidate materials were found in heatrapy's "
        "database on this installation. Run "
        "`import heatrapy, os; print(os.listdir(os.path.dirname(heatrapy.__file__) + '/database'))` "
        "to see what is actually available, then edit "
        "`candidate_materials` above."
    )

print(f"Using built-in material: '{chosen_material}'")

# ---------------------------------------------------------------------
# 2. Geometry and boundary conditions
# ---------------------------------------------------------------------
# Single slab, e.g. representing one refractory course.
L = 0.20          # slab thickness [m]  (e.g. 200 mm refractory layer)
dx = 0.002        # spatial step [m]  -> 100 nodes across the slab
dt = 0.05         # time step [s]     (small for numerical stability)

T_hot = 1273.0    # hot-face temperature [K]  (~1000 C)
T_cold = 323.0    # cold-face temperature [K] (~50 C)

n_points = int(L / dx)

wall = htp.SingleObject1D(
    293,                                # initial (irrelevant once we reach steady state)
    materials=(chosen_material,),
    borders=(1, n_points + 1),
    materials_order=(0,),
    dx=dx,
    dt=dt,
    boundaries=(T_hot, T_cold),         # Dirichlet BCs on both faces
    file_name="old/wall_temperature.csv",
    draw=[],                            # disable live plotting for batch runs
)

# ---------------------------------------------------------------------
# 3. Run transient simulation until steady state is reached
# ---------------------------------------------------------------------
# "implicit_general" assumes constant k over the slab, matching the
# simplest SIMU-THERM steady-state case for one homogeneous layer.

chunk_time = 200.0     # seconds simulated per convergence check
max_time = 20000.0     # safety cap on total simulated time
tolerance = 1e-4       # K, max allowed change between checks

previous_profile = None
total_time = 0.0

while total_time < max_time:
    wall.compute(chunk_time, write_interval=1, solver="implicit_general", verbose=False)
    total_time += chunk_time

    current_profile = [pt[0] for pt in wall.object.temperature]

    if previous_profile is not None:
        max_delta = max(
            abs(a - b) for a, b in zip(current_profile, previous_profile)
        )
        if max_delta < tolerance:
            print(f"Converged after {total_time:.0f} s "
                  f"(max change {max_delta:.2e} K)")
            break

    previous_profile = current_profile
else:
    print(f"WARNING: did not converge within {max_time:.0f} s, "
          f"max change was {max_delta:.2e} K")

# ---------------------------------------------------------------------
# 4. Extract results and compare against the analytical steady state
# ---------------------------------------------------------------------
final_profile = [pt[0] for pt in wall.object.temperature]
x_positions = [i * dx for i in range(len(final_profile))]

# Analytical steady-state profile for a homogeneous slab is LINEAR
# (independent of density/Cp - those only affect transient speed, not
# the final profile), so this is the same check SIMU-THERM performs.
analytical_profile = [
    T_hot + (T_cold - T_hot) * (x / L) for x in x_positions
]

max_error = max(
    abs(a - b) for a, b in zip(final_profile, analytical_profile)
)

# Numerical heat flux from the converged profile (forward difference
# at the hot face), using the k value the solver actually used there.
k_used = wall.object.k[1]
q_numerical = -k_used * (final_profile[1] - final_profile[0]) / dx
q_analytical = k_used * (T_hot - T_cold) / L

print(f"Material: {chosen_material}")
print(f"k used by solver: {k_used:.4f} W/(m.K)")
print(f"Max deviation from linear analytical profile: {max_error:.4f} K")
print(f"Heat flux (numerical):  {q_numerical:.2f} W/m^2")
print(f"Heat flux (analytical): {q_analytical:.2f} W/m^2")

# ---------------------------------------------------------------------
# 5. Save profile for comparison against SIMU-THERM
# ---------------------------------------------------------------------
with open("old/steady_state_profile.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x_m", "T_heatrapy_K", "T_analytical_K"])
    for x, t_num, t_ana in zip(x_positions, final_profile, analytical_profile):
        writer.writerow([x, t_num, t_ana])

print("Saved profile to steady_state_profile.csv")
print("Compare T_hot, T_cold, L, k, and q above directly against your "
      "SIMU-THERM steady-state run for the same slab.")
