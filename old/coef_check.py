SIGMA = 5.670374419e-8
CAL = {"V": (2.207645, 0.25), "R": (2.065726, 0.25), "F": (1.048352, 0.25)}

# =============================================================================
# INPUT
# =============================================================================
ORIENTATION = "V"        # V = vertical, R = ceiling, F = floor
T_SURFACE_C = 800       # outer surface temperature
T_AMBIENT_C = 40.0       # ambient temperature outside
EMISSIVITY = 0.5

# =============================================================================
# CALCULATION
# =============================================================================
Ts, Ta = T_SURFACE_C + 273.15, T_AMBIENT_C + 273.15
C, n = CAL[ORIENTATION]

h_conv = C * abs(T_SURFACE_C - T_AMBIENT_C) ** n
h_rad = EMISSIVITY * SIGMA * (Ts**2 + Ta**2) * (Ts + Ta)
h_total = h_conv + h_rad

print(f"Convection : {h_conv:.3f} W/m2K")
print(f"Radiation  : {h_rad:.3f} W/m2K")
print(f"Total h    : {h_total:.3f} W/m2K")