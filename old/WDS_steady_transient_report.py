"""
Steady state WDS wall calculation with Robin boundary conditions.

Files created:
  wds_robin_steady_state_profile.csv
  wds_robin_steady_state_report.html
  wds_robin_steady_state_profile.svg
  wds_robin_steady_state_interactive.html

Use the INTERACTIVE HTML file for mouse readout. The SVG file is a portable
XML vector export and intentionally has no JavaScript dependency.
"""

import csv
import html
import numpy as np
import heatrapy as htp

# =============================================================================
# USER SETTINGS
# =============================================================================
MATERIAL_NAME = "WDS"
MATERIALS_PATH = (
    r"C:\Users\orhan.kibrisli\PycharmProjects\heat_transfer\.venv"
    r"\Lib\site-packages\heatrapy\database\\"
)
MATERIAL_MAX_TEMP_C = 950.0

L_MM = 20.0
DX_MM = 2.0
DT = 50.0

T_INITIAL_C = 30.0
T_GAS_HOT_C = 900.0
T_GAS_COLD_C = 30.0
H_HOT = 150.0
H_COLD = 12.09

PICARD_TOLERANCE = 1.0e-6
PICARD_MAX_ITERATIONS = 100
STEADY_CHUNK_TIME = 5000.0
STEADY_MAX_TIME = 2000000.0
STEADY_TOLERANCE = 1.0e-4

STEADY_OUTPUT_CSV = "wds_robin_steady_state_profile.csv"
HTML_REPORT_FILE = "wds_robin_steady_state_report.html"
SVG_GRAPH_FILE = "wds_robin_steady_state_profile.svg"
INTERACTIVE_GRAPH_FILE = "wds_robin_steady_state_interactive.html"

# =============================================================================
# SOLVER
# =============================================================================
L = L_MM / 1000.0
T_INITIAL = T_INITIAL_C + 273.15
T_GAS_HOT = T_GAS_HOT_C + 273.15
T_GAS_COLD = T_GAS_COLD_C + 273.15
n_intervals = int(round(L / (DX_MM / 1000.0)))
if n_intervals < 1 or DT <= 0.0 or H_HOT <= 0.0 or H_COLD <= 0.0:
    raise ValueError("Check wall thickness, mesh step, time step, and film coefficients.")
DX = L / n_intervals
n_nodes = n_intervals + 1
x_m = np.linspace(0.0, L, n_nodes)
x_mm = x_m * 1000.0
cv_width = np.full(n_nodes, DX)
cv_width[0] = DX / 2.0
cv_width[-1] = DX / 2.0

loader = htp.SingleObject1D(
    T_INITIAL, materials=(MATERIAL_NAME,), borders=(1, n_intervals),
    materials_order=(0,), dx=DX, dt=DT, boundaries=(0, 0),
    materials_path=MATERIALS_PATH, file_name=None, draw=[]
)
material = loader.object.materials[0]


def properties(T):
    k = np.array([material.k0(float(v)) for v in T], dtype=float)
    rho = np.array([material.rho0(float(v)) for v in T], dtype=float)
    cp = np.array([material.cp0(float(v)) for v in T], dtype=float)
    if np.any(k <= 0.0) or np.any(rho <= 0.0) or np.any(cp <= 0.0):
        raise ValueError("Material database returned a nonpositive property.")
    return k, rho, cp


def hmean(a, b):
    return 2.0 * a * b / (a + b)


def step(T_old, dt):
    """Fully implicit FVM step with Picard iteration for k(T), rho(T), Cp(T)."""
    T = T_old.copy()
    for _ in range(PICARD_MAX_ITERATIONS):
        k, rho, cp = properties(T)
        g = hmean(k[:-1], k[1:]) / DX
        c = rho * cp * cv_width / dt
        A = np.zeros((n_nodes, n_nodes))
        b = c * T_old

        A[0, 0], A[0, 1] = c[0] + H_HOT + g[0], -g[0]
        b[0] += H_HOT * T_GAS_HOT
        for i in range(1, n_nodes - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = -g[i - 1], c[i] + g[i - 1] + g[i], -g[i]
        A[-1, -2], A[-1, -1] = -g[-1], c[-1] + g[-1] + H_COLD
        b[-1] += H_COLD * T_GAS_COLD

        candidate = np.linalg.solve(A, b)
        if np.max(np.abs(candidate - T)) < PICARD_TOLERANCE:
            return candidate
        T = candidate
    raise RuntimeError("Picard iteration did not converge. Reduce DT.")


def advance(T_start, seconds):
    T, elapsed = T_start.copy(), 0.0
    while elapsed < seconds - 1.0e-12:
        dt_step = min(DT, seconds - elapsed)
        T = step(T, dt_step)
        elapsed += dt_step
    return T


print(f"Material: {MATERIAL_NAME}")
print(f"Mesh: {n_nodes} nodes, {n_intervals} intervals, dx = {DX * 1000.0:.4f} mm")
T = np.full(n_nodes, T_INITIAL)
previous = None
total_time = 0.0
max_delta = float("nan")
converged = False
while total_time < STEADY_MAX_TIME:
    duration = min(STEADY_CHUNK_TIME, STEADY_MAX_TIME - total_time)
    T = advance(T, duration)
    total_time += duration
    if previous is not None:
        max_delta = float(np.max(np.abs(T - previous)))
        if max_delta < STEADY_TOLERANCE:
            converged = True
            break
    previous = T.copy()

k_nodes, _, _ = properties(T)
k_face = hmean(k_nodes[:-1], k_nodes[1:])
q_wall = -k_face * np.diff(T) / DX
q_mean = float(np.mean(q_wall))
q_hot = H_HOT * (T_GAS_HOT - T[0])
q_cold = H_COLD * (T[-1] - T_GAS_COLD)
q_spread = float(np.max(q_wall) - np.min(q_wall))
T_C = T - 273.15
T_hot, T_cold = float(T_C[0]), float(T_C[-1])
T_mean = float(np.trapezoid(T_C, x_m) / L)
k_mean = float(material.k0(T_mean + 273.15))
k_effective = q_mean * L / (T_hot - T_cold)
R_hot, R_wall, R_cold = 1.0 / H_HOT, (T_hot - T_cold) / q_mean, 1.0 / H_COLD
R_total = (T_GAS_HOT_C - T_GAS_COLD_C) / q_mean

print(f"Steady state {'reached' if converged else 'not reached'} after {total_time:.0f} s")
print(f"Hot wall surface temperature:  {T_hot:.2f} C")
print(f"Cold wall surface temperature: {T_cold:.2f} C")
print(f"Heat loss:                     {q_mean:.2f} W/m2")

with open(STEADY_OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["x_mm", "T_C", "k_W_mK"])
    writer.writerows(zip(x_mm, T_C, k_nodes))
print(f"Saved {STEADY_OUTPUT_CSV}")

# =============================================================================
# PRINTABLE HTML REPORT
# =============================================================================
def fmt(v, d=2):
    return f"{v:.{d}f}"

status = "Converged" if converged else "Maximum calculation time reached"
report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>WDS steady state thermal report</title><style>
body{{font-family:Arial,Helvetica,sans-serif;color:#14202b;margin:36px;max-width:1000px}}h1{{font-size:25px;margin:0 0 6px}}h2{{font-size:17px;margin:27px 0 8px;border-bottom:2px solid #1f4e79;padding-bottom:4px}}p{{margin:5px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:26px}}table{{border-collapse:collapse;width:100%;margin:8px 0 16px}}th{{background:#dce6f1;text-align:left}}th,td{{border:1px solid #7f8c8d;padding:7px 9px}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}.note{{background:#f3f7fb;border-left:4px solid #1f4e79;padding:11px 13px}}.ok{{color:#116329;font-weight:700}}@media print{{body{{margin:15mm}}.printnote{{display:none}}}}
</style></head><body>
<h1>Steady state wall heat loss report</h1><p>Planar WDS Ultra insulation wall with nonlinear material properties and convective surface films.</p><p class="printnote">Use the browser print command to save as PDF.</p>
<h2>Calculation status</h2><p class="ok">{status}</p><p>Simulated time: {fmt(total_time,0)} s. Final profile change: {fmt(max_delta,8)} K.</p>
<div class="grid"><div><h2>Boundary conditions</h2><table><tr><th>Parameter</th><th class="num">Inside</th><th class="num">Outside</th><th>Unit</th></tr><tr><td>Ambient temperature, input</td><td class="num">{fmt(T_GAS_HOT_C,1)}</td><td class="num">{fmt(T_GAS_COLD_C,1)}</td><td>°C</td></tr><tr><td>Surface temperature, calculated</td><td class="num">{fmt(T_hot,1)}</td><td class="num">{fmt(T_cold,1)}</td><td>°C</td></tr><tr><td>Heat transfer coefficient</td><td class="num">{fmt(H_HOT,2)}</td><td class="num">{fmt(H_COLD,2)}</td><td>W/m²K</td></tr><tr><td>Boundary condition</td><td class="num">Robin</td><td class="num">Robin</td><td></td></tr><tr><td>Heat loss per area</td><td class="num">{fmt(q_hot,2)}</td><td class="num">{fmt(q_cold,2)}</td><td>W/m²</td></tr></table></div>
<div><h2>Wall layer</h2><table><tr><th>Material</th><th class="num">Thickness</th><th class="num">Mean temperature</th><th class="num">k at mean T</th></tr><tr><td>{html.escape(MATERIAL_NAME)} Ultra</td><td class="num">{fmt(L_MM,2)} mm</td><td class="num">{fmt(T_mean,1)} °C</td><td class="num">{fmt(k_mean,4)} W/mK</td></tr></table><h2>Numerical setup</h2><table><tr><th>Item</th><th class="num">Value</th></tr><tr><td>Spatial intervals</td><td class="num">{n_intervals}</td></tr><tr><td>Spatial step</td><td class="num">{fmt(DX*1000,4)} mm</td></tr><tr><td>Time step</td><td class="num">{fmt(DT,2)} s</td></tr><tr><td>Solver</td><td class="num">Implicit FVM with Picard iteration</td></tr></table></div></div>
<h2>Resistance and energy balance</h2><table><tr><th>Quantity</th><th class="num">Value</th><th>Unit</th></tr><tr><td>Inside film resistance</td><td class="num">{fmt(R_hot,6)}</td><td>m²K/W</td></tr><tr><td>WDS conduction resistance</td><td class="num">{fmt(R_wall,6)}</td><td>m²K/W</td></tr><tr><td>Outside film resistance</td><td class="num">{fmt(R_cold,6)}</td><td>m²K/W</td></tr><tr><td>Total resistance</td><td class="num">{fmt(R_total,6)}</td><td>m²K/W</td></tr><tr><td>Effective WDS conductivity</td><td class="num">{fmt(k_effective,5)}</td><td>W/mK</td></tr><tr><td>Mean wall heat flux</td><td class="num">{fmt(q_mean,2)}</td><td>W/m²</td></tr><tr><td>Wall flux spread</td><td class="num">{fmt(q_spread,7)}</td><td>W/m²</td></tr><tr><td>Film flux mismatch</td><td class="num">{fmt(abs(q_hot-q_cold),7)}</td><td>W/m²</td></tr></table>
<div class="note"><strong>Boundary interpretation:</strong> the ambient temperatures are gas temperatures. The two WDS surface temperatures are calculated from the conduction solution and convective film resistances.</div></body></html>"""
with open(HTML_REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Saved {HTML_REPORT_FILE}")

# =============================================================================
# GRAPH LAYOUT AND SVG CONTENT
# =============================================================================
W, H = 1420, 820
left, right, top, bottom = 165, 1110, 135, 660
plot_w, plot_h = right-left, bottom-top
T_min = min(T_GAS_COLD_C, float(np.min(T_C))) - 20.0
T_max = max(MATERIAL_MAX_TEMP_C, T_GAS_HOT_C, float(np.max(T_C))) + 20.0
sx = lambda x: left + x/L_MM*plot_w
sy = lambda temp: bottom - (temp-T_min)/(T_max-T_min)*plot_h
points = " ".join(f"{sx(x):.3f},{sy(temp):.3f}" for x, temp in zip(x_mm, T_C))

y_ticks = range(int(np.ceil(T_min/100)*100), int(np.floor(T_max/100)*100)+1, 100)
x_ticks = np.linspace(0, L_MM, 5)
grid_y = "\n".join(f'<line class="grid" x1="{left}" y1="{sy(y):.3f}" x2="{right}" y2="{sy(y):.3f}"/><text class="axis" x="{left-15}" y="{sy(y)+5:.3f}" text-anchor="end">{y}</text>' for y in y_ticks)
grid_x = "\n".join(f'<line class="grid" x1="{sx(x):.3f}" y1="{top}" x2="{sx(x):.3f}" y2="{bottom}"/><text class="axis" x="{sx(x):.3f}" y="{bottom+32}" text-anchor="middle">{x:.1f}</text>' for x in x_ticks)
node_circles = "\n".join(f'<circle class="node" cx="{sx(x):.3f}" cy="{sy(temp):.3f}" r="6"><title>x = {x:.3f} mm, T = {temp:.3f} °C, k = {k:.5f} W/mK</title></circle>' for x, temp, k in zip(x_mm,T_C,k_nodes))

# Surface labels are deliberately outside the graph. Their leader lines start
# at the exact calculated surface temperatures.
svg_body = f"""
<style>
text{{font-family:Arial,Helvetica,sans-serif;fill:#101820}}.title{{font-size:24px;font-weight:bold}}.subtitle{{font-size:15px}}.axis{{font-size:13px}}.label{{font-size:16px;font-weight:bold}}.grid{{stroke:#d6dce4;stroke-width:1}}.border{{stroke:#101820;stroke-width:2;fill:white}}.profile{{fill:none;stroke:#c9242b;stroke-width:3;stroke-linejoin:round;stroke-linecap:round}}.limit{{stroke:#16803c;stroke-width:3}}.film{{stroke:#65717d;stroke-width:2;stroke-dasharray:7 5}}.callout{{stroke:#65717d;stroke-width:1.5}}.node{{fill:#c9242b;fill-opacity:0;cursor:crosshair}}
</style>
<rect width="100%" height="100%" fill="white"/>
<text class="title" x="{left}" y="40">Steady state temperature profile</text>
<text class="subtitle" x="{left}" y="67">{html.escape(MATERIAL_NAME)} Ultra, thickness {L_MM:.2f} mm, heat loss {q_mean:.2f} W/m²</text>
<text class="label" x="{left}" y="104">Temperature, °C</text>
<rect class="border" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>
{grid_y}
{grid_x}
<line class="limit" x1="{left}" y1="{sy(MATERIAL_MAX_TEMP_C):.3f}" x2="{right}" y2="{sy(MATERIAL_MAX_TEMP_C):.3f}"/>
<text class="axis" x="{right+16}" y="{sy(MATERIAL_MAX_TEMP_C)+5:.3f}">WDS maximum use temperature: {MATERIAL_MAX_TEMP_C:.0f} °C</text>
<line class="film" x1="{left-50}" y1="{sy(T_GAS_HOT_C):.3f}" x2="{left}" y2="{sy(T_hot):.3f}"/>
<line class="film" x1="{right}" y1="{sy(T_cold):.3f}" x2="{right+58}" y2="{sy(T_GAS_COLD_C):.3f}"/>
<polyline class="profile" points="{points}"/>
{node_circles}
<line class="callout" x1="{left}" y1="{sy(T_GAS_HOT_C):.3f}" x2="{left-32}" y2="{sy(T_GAS_HOT_C):.3f}"/>
<text class="axis" x="{left-40}" y="{sy(T_GAS_HOT_C)-9:.3f}" text-anchor="end">Hot gas: {T_GAS_HOT_C:.0f} °C</text>
<line class="callout" x1="{left}" y1="{sy(T_hot):.3f}" x2="{left-32}" y2="{sy(T_hot)+25:.3f}"/>
<text class="axis" x="{left-40}" y="{sy(T_hot)+31:.3f}" text-anchor="end">Hot surface: {T_hot:.1f} °C</text>
<line class="callout" x1="{right}" y1="{sy(T_cold):.3f}" x2="{right+42}" y2="{sy(T_cold)-24:.3f}"/>
<text class="axis" x="{right+50}" y="{sy(T_cold)-29:.3f}">Cold surface: {T_cold:.1f} °C</text>
<line class="callout" x1="{right+58}" y1="{sy(T_GAS_COLD_C):.3f}" x2="{right+82}" y2="{sy(T_GAS_COLD_C):.3f}"/>
<text class="axis" x="{right+90}" y="{sy(T_GAS_COLD_C)+5:.3f}">Cold gas: {T_GAS_COLD_C:.0f} °C</text>
<text class="label" x="{(left+right)/2:.3f}" y="{bottom+70}" text-anchor="middle">Wall depth from inside to outside, mm</text>
"""

static_svg = f'<?xml version="1.0" encoding="UTF-8"?><svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{svg_body}</svg>'
with open(SVG_GRAPH_FILE, "w", encoding="utf-8") as f:
    f.write(static_svg)
print(f"Saved {SVG_GRAPH_FILE}")

# =============================================================================
# INTERACTIVE HTML GRAPH
# =============================================================================
js_x = ",".join(f"{v:.12g}" for v in x_mm)
js_T = ",".join(f"{v:.12g}" for v in T_C)
js_k = ",".join(f"{v:.12g}" for v in k_nodes)
interactive_html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Interactive WDS steady state profile</title><style>body{{margin:0;background:#f4f6f8}}#frame{{width:1420px;margin:18px auto;background:white;box-shadow:0 1px 6px #9aa4ad}}#plot{{display:block}}#cross{{stroke:#1f4e79;stroke-width:1;stroke-dasharray:4 4;visibility:hidden}}#tipbox{{fill:#fff;stroke:#1f4e79;stroke-width:1;visibility:hidden}}#tip{{font-family:Arial,Helvetica,sans-serif;font-size:14px;fill:#101820;visibility:hidden}}</style></head><body><div id="frame"><svg id="plot" xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{svg_body}<line id="cross" x1="0" y1="{top}" x2="0" y2="{bottom}"/><rect id="tipbox" x="0" y="0" width="250" height="56" rx="4"/><text id="tip" x="0" y="0"></text><rect id="overlay" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="transparent" style="cursor:crosshair"/></svg></div><script>
const xData=[{js_x}], tData=[{js_T}], kData=[{js_k}];
const plot=document.getElementById('plot'), overlay=document.getElementById('overlay');
const cross=document.getElementById('cross'), tip=document.getElementById('tip'), tipbox=document.getElementById('tipbox');
function interp(values,x){{for(let i=0;i<xData.length-1;i++){{if(x<=xData[i+1]){{const r=(x-xData[i])/(xData[i+1]-xData[i]);return values[i]+r*(values[i+1]-values[i]);}}}}return values[values.length-1];}}
overlay.addEventListener('mousemove',function(e){{const r=plot.getBoundingClientRect();const px=(e.clientX-r.left)*{W}/r.width;const clipped=Math.max({left},Math.min({right},px));const x=(clipped-{left})/{plot_w}*{L_MM};const temp=interp(tData,x),k=interp(kData,x);cross.setAttribute('x1',clipped);cross.setAttribute('x2',clipped);cross.style.visibility='visible';const bx=Math.min(clipped+14,{right}-264),by={top}+15;tipbox.setAttribute('x',bx);tipbox.setAttribute('y',by);tipbox.style.visibility='visible';tip.setAttribute('x',bx+10);tip.setAttribute('y',by+21);tip.style.visibility='visible';while(tip.firstChild)tip.removeChild(tip.firstChild);tip.appendChild(document.createTextNode('x = '+x.toFixed(3)+' mm, T = '+temp.toFixed(2)+' °C'));const s=document.createElementNS('http://www.w3.org/2000/svg','tspan');s.setAttribute('x',bx+10);s.setAttribute('dy',19);s.textContent='k = '+k.toFixed(5)+' W/mK';tip.appendChild(s);}});
overlay.addEventListener('mouseleave',function(){{cross.style.visibility='hidden';tip.style.visibility='hidden';tipbox.style.visibility='hidden';}});
</script></body></html>"""
with open(INTERACTIVE_GRAPH_FILE, "w", encoding="utf-8") as f:
    f.write(interactive_html)
print(f"Saved {INTERACTIVE_GRAPH_FILE}")
print("Done. Open wds_robin_steady_state_interactive.html in a web browser for mouse readout.")
