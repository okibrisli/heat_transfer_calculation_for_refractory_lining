"""
Transient 1D WDS wall model with Robin boundary conditions.

Outputs:
  wds_robin_transient_data.csv
  wds_robin_transient_report.html
  wds_robin_transient_interactive.html

Open wds_robin_transient_interactive.html in Chrome, Edge, or Firefox.
It contains a profile time slider, temperature histories, heat fluxes, and
stored energy. The report HTML is formatted for printing to PDF.
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
DT = 20.0
TOTAL_TIME_S = 12000.0
OUTPUT_INTERVAL_S = 60.0

T_INITIAL_C = 30.0
T_GAS_HOT_C = 900.0
T_GAS_COLD_C = 30.0
H_HOT = 150.0
H_COLD = 12.09

PICARD_TOLERANCE = 1.0e-6
PICARD_MAX_ITERATIONS = 100
COLD_FACE_LIMITS_C = [50.0, 60.0, 80.0, 100.0]

CSV_FILE = "wds_robin_transient_data.csv"
REPORT_FILE = "wds_robin_transient_report.html"
DASHBOARD_FILE = "wds_robin_transient_interactive.html"

# =============================================================================
# MODEL SETUP
# =============================================================================
L = L_MM / 1000.0
T_INITIAL = T_INITIAL_C + 273.15
T_GAS_HOT = T_GAS_HOT_C + 273.15
T_GAS_COLD = T_GAS_COLD_C + 273.15
n_intervals = int(round(L / (DX_MM / 1000.0)))
if n_intervals < 1 or DT <= 0 or OUTPUT_INTERVAL_S <= 0 or TOTAL_TIME_S <= 0:
    raise ValueError("Check geometry, time step, total time, and output interval.")
if H_HOT <= 0 or H_COLD <= 0:
    raise ValueError("Heat transfer coefficients must be positive.")
if T_GAS_HOT_C > MATERIAL_MAX_TEMP_C:
    print(f"WARNING: hot gas temperature exceeds the WDS maximum use temperature of {MATERIAL_MAX_TEMP_C:.1f} C.")

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
    if np.any(k <= 0) or np.any(rho <= 0) or np.any(cp <= 0):
        raise ValueError("Material database returned a nonpositive property.")
    return k, rho, cp


def hmean(a, b):
    return 2.0 * a * b / (a + b)


def step(T_old, dt):
    """Backward Euler finite volume step with Picard iteration."""
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


def fluxes(T):
    k, _, _ = properties(T)
    q_wall = -hmean(k[:-1], k[1:]) * np.diff(T) / DX
    q_hot = H_HOT * (T_GAS_HOT - T[0])
    q_cold = H_COLD * (T[-1] - T_GAS_COLD)
    return q_hot, q_cold, float(np.mean(q_wall)), k


# Enthalpy lookup for variable volumetric heat capacity. E_stored is per m2
# relative to the initial uniform wall state.
T_lookup = np.linspace(min(T_INITIAL, T_GAS_COLD), max(T_GAS_HOT, T_INITIAL), 5001)
_, rho_lookup, cp_lookup = properties(T_lookup)
vol_cp = rho_lookup * cp_lookup
enthalpy_lookup = np.concatenate(([0.0], np.cumsum(0.5 * (vol_cp[1:] + vol_cp[:-1]) * np.diff(T_lookup))))
h_initial = float(np.interp(T_INITIAL, T_lookup, enthalpy_lookup))


def stored_energy(T):
    h = np.interp(T, T_lookup, enthalpy_lookup) - h_initial
    return float(np.sum(h * cv_width))


def crossing_time(times, values, limit):
    above = np.where(values >= limit)[0]
    if len(above) == 0:
        return None
    i = int(above[0])
    if i == 0:
        return float(times[0])
    t0, t1 = times[i - 1], times[i]
    v0, v1 = values[i - 1], values[i]
    return float(t0 + (limit - v0) * (t1 - t0) / (v1 - v0))


# =============================================================================
# TRANSIENT CALCULATION
# =============================================================================
print(f"Material: {MATERIAL_NAME}")
print(f"Mesh: {n_nodes} nodes, {n_intervals} intervals, dx = {DX * 1000.0:.4f} mm")
print(f"Transient duration: {TOTAL_TIME_S:.1f} s, integration step: {DT:.2f} s")

T = np.full(n_nodes, T_INITIAL)
t = 0.0
next_output = 0.0
times, profiles, q_hot_series, q_cold_series, q_wall_series, energy_series = [], [], [], [], [], []

while t < TOTAL_TIME_S - 1.0e-12:
    if t >= next_output - 1.0e-10:
        qh, qc, qw, _ = fluxes(T)
        times.append(t)
        profiles.append((T - 273.15).copy())
        q_hot_series.append(qh)
        q_cold_series.append(qc)
        q_wall_series.append(qw)
        energy_series.append(stored_energy(T))
        next_output += OUTPUT_INTERVAL_S
    dt_step = min(DT, TOTAL_TIME_S - t, max(next_output - t, 1.0e-12))
    T = step(T, dt_step)
    t += dt_step

# Always save the exact final time point.
qh, qc, qw, _ = fluxes(T)
if not times or abs(times[-1] - t) > 1.0e-8:
    times.append(t)
    profiles.append((T - 273.15).copy())
    q_hot_series.append(qh)
    q_cold_series.append(qc)
    q_wall_series.append(qw)
    energy_series.append(stored_energy(T))

Times = np.asarray(times)
Profiles = np.asarray(profiles)
Qhot = np.asarray(q_hot_series)
Qcold = np.asarray(q_cold_series)
Qwall = np.asarray(q_wall_series)
Energy = np.asarray(energy_series)

print(f"Calculated {len(Times)} saved transient states.")
print(f"Final hot surface: {Profiles[-1, 0]:.2f} C")
print(f"Final cold surface: {Profiles[-1, -1]:.2f} C")
print(f"Final heat loss: {Qwall[-1]:.2f} W/m2")

# =============================================================================
# CSV OUTPUT
# =============================================================================
with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["time_s", "q_hot_W_m2", "q_cold_W_m2", "q_wall_mean_W_m2", "stored_energy_J_m2"]
    header += [f"T_x_{x:.3f}_mm_C" for x in x_mm]
    writer.writerow(header)
    for i in range(len(Times)):
        writer.writerow([Times[i], Qhot[i], Qcold[i], Qwall[i], Energy[i], *Profiles[i]])
print(f"Saved {CSV_FILE}")

# =============================================================================
# REPORT METRICS
# =============================================================================
monitor_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
monitor_indices = [int(np.argmin(np.abs(x_m - f * L))) for f in monitor_fractions]
monitor_labels = [f"{x_mm[i]:.1f} mm" for i in monitor_indices]
cold_history = Profiles[:, -1]
threshold_rows = []
for limit in COLD_FACE_LIMITS_C:
    time_cross = crossing_time(Times, cold_history, limit)
    threshold_rows.append((limit, "Not reached" if time_cross is None else f"{time_cross:.1f} s, {time_cross / 60.0:.2f} min"))

# 90, 95 and 99 percent local cold face response relative to its final value.
response_rows = []
for fraction in [0.90, 0.95, 0.99]:
    target = T_INITIAL_C + fraction * (cold_history[-1] - T_INITIAL_C)
    time_cross = crossing_time(Times, cold_history, target)
    response_rows.append((fraction * 100.0, target, "Not reached" if time_cross is None else f"{time_cross:.1f} s"))

# =============================================================================
# PRINTABLE HTML REPORT
# =============================================================================
def fmt(v, d=2):
    return f"{v:.{d}f}"

threshold_html = "".join(f"<tr><td>{fmt(limit,1)} °C</td><td>{value}</td></tr>" for limit, value in threshold_rows)
response_html = "".join(f"<tr><td>{fmt(frac,0)} %</td><td>{fmt(temp,2)} °C</td><td>{value}</td></tr>" for frac, temp, value in response_rows)
report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>WDS transient thermal report</title><style>
body{{font-family:Arial,Helvetica,sans-serif;color:#14202b;margin:36px;max-width:1000px}}h1{{font-size:25px;margin:0 0 6px}}h2{{font-size:17px;margin:27px 0 8px;border-bottom:2px solid #1f4e79;padding-bottom:4px}}p{{margin:5px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:26px}}table{{border-collapse:collapse;width:100%;margin:8px 0 16px}}th{{background:#dce6f1;text-align:left}}th,td{{border:1px solid #7f8c8d;padding:7px 9px}}td.num{{text-align:right}}.note{{background:#f3f7fb;border-left:4px solid #1f4e79;padding:11px 13px}}@media print{{body{{margin:15mm}}.printnote{{display:none}}}}
</style></head><body>
<h1>Transient wall heat transfer report</h1><p>One dimensional planar {html.escape(MATERIAL_NAME)} Ultra insulation wall with nonlinear properties and Robin boundary conditions.</p><p class="printnote">Open the interactive HTML dashboard separately for profile animation and chart inspection. Use the browser print command to save this report as PDF.</p>
<h2>Model definition</h2><table><tr><th>Parameter</th><th class="num">Value</th><th>Unit</th></tr><tr><td>Wall thickness</td><td class="num">{fmt(L_MM,2)}</td><td>mm</td></tr><tr><td>Spatial intervals</td><td class="num">{n_intervals}</td><td></td></tr><tr><td>Spatial step</td><td class="num">{fmt(DX*1000,4)}</td><td>mm</td></tr><tr><td>Integration time step</td><td class="num">{fmt(DT,2)}</td><td>s</td></tr><tr><td>Simulated duration</td><td class="num">{fmt(Times[-1],1)}</td><td>s</td></tr><tr><td>Solver</td><td class="num">Implicit finite volume, Picard iteration</td><td></td></tr></table>
<div class="grid"><div><h2>Boundary conditions</h2><table><tr><th>Parameter</th><th class="num">Inside</th><th class="num">Outside</th><th>Unit</th></tr><tr><td>Gas temperature</td><td class="num">{fmt(T_GAS_HOT_C,1)}</td><td class="num">{fmt(T_GAS_COLD_C,1)}</td><td>°C</td></tr><tr><td>Film coefficient</td><td class="num">{fmt(H_HOT,2)}</td><td class="num">{fmt(H_COLD,2)}</td><td>W/m²K</td></tr><tr><td>Boundary type</td><td class="num">Robin</td><td class="num">Robin</td><td></td></tr></table></div><div><h2>Final transient state</h2><table><tr><th>Quantity</th><th class="num">Value</th><th>Unit</th></tr><tr><td>Hot surface temperature</td><td class="num">{fmt(Profiles[-1,0],2)}</td><td>°C</td></tr><tr><td>Cold surface temperature</td><td class="num">{fmt(Profiles[-1,-1],2)}</td><td>°C</td></tr><tr><td>Hot side flux</td><td class="num">{fmt(Qhot[-1],2)}</td><td>W/m²</td></tr><tr><td>Cold side flux</td><td class="num">{fmt(Qcold[-1],2)}</td><td>W/m²</td></tr><tr><td>Stored energy</td><td class="num">{fmt(Energy[-1],1)}</td><td>J/m²</td></tr></table></div></div>
<h2>Cold face temperature limits</h2><table><tr><th>Cold face limit</th><th>First crossing time</th></tr>{threshold_html}</table>
<h2>Cold face stabilization</h2><table><tr><th>Final response fraction</th><th>Temperature target</th><th>First crossing time</th></tr>{response_html}</table>
<div class="note"><strong>Transient energy balance:</strong> during heat up, the hot side heat flux is larger than the cold side flux. Their difference is stored as sensible thermal energy in the WDS wall. As steady state is approached, the two film fluxes converge.</div>
</body></html>"""
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Saved {REPORT_FILE}")

# =============================================================================
# INTERACTIVE HTML DASHBOARD
# =============================================================================
js_times = ",".join(f"{v:.12g}" for v in Times)
js_profiles = ",".join("[" + ",".join(f"{v:.12g}" for v in row) + "]" for row in Profiles)
js_qhot = ",".join(f"{v:.12g}" for v in Qhot)
js_qcold = ",".join(f"{v:.12g}" for v in Qcold)
js_qwall = ",".join(f"{v:.12g}" for v in Qwall)
js_energy = ",".join(f"{v:.12g}" for v in Energy)
js_x = ",".join(f"{v:.12g}" for v in x_mm)
js_monitor_indices = ",".join(str(i) for i in monitor_indices)
js_monitor_labels = ",".join('"' + label + '"' for label in monitor_labels)

dashboard = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>WDS transient dashboard</title><style>
body{{font-family:Arial,Helvetica,sans-serif;background:#f4f6f8;color:#14202b;margin:0}}main{{width:1180px;margin:20px auto 35px;background:#fff;padding:26px;box-shadow:0 1px 7px #aeb7c0}}h1{{margin:0 0 5px;font-size:25px}}h2{{font-size:18px;margin:28px 0 8px;border-bottom:2px solid #1f4e79;padding-bottom:4px}}canvas{{border:1px solid #65717d;display:block;background:#fff}}#profile{{cursor:crosshair}}.controls{{display:flex;align-items:center;gap:12px;margin:10px 0}}input[type=range]{{width:720px}}button{{padding:6px 12px}}#state{{font-weight:bold;min-width:230px}}.note{{background:#f3f7fb;border-left:4px solid #1f4e79;padding:10px 12px;margin-top:13px}}table{{border-collapse:collapse;margin-top:8px}}th,td{{border:1px solid #87929c;padding:6px 10px}}th{{background:#dce6f1;text-align:left}}td.num{{text-align:right}}
</style></head><body><main>
<h1>Transient temperature and heat flow dashboard</h1><p>{html.escape(MATERIAL_NAME)} Ultra, {L_MM:.2f} mm wall, hot gas {T_GAS_HOT_C:.0f} °C, cold ambient {T_GAS_COLD_C:.0f} °C.</p>
<h2>Temperature profile</h2><canvas id="profile" width="1120" height="500"></canvas><div class="controls"><button id="play">Play</button><button id="reset">Reset</button><input id="slider" type="range" min="0" max="{len(Times)-1}" value="0" step="1"><span id="state"></span></div><div class="note">Use the slider or Play button. Move the pointer across the profile plot for interpolated position, temperature, and conductivity.</div>
<h2>Temperature history at virtual thermocouples</h2><canvas id="history" width="1120" height="490"></canvas>
<h2>Heat flux and stored energy</h2><canvas id="flux" width="1120" height="490"></canvas>
</main><script>
const times=[{js_times}], profiles=[{js_profiles}], x=[{js_x}], qHot=[{js_qhot}], qCold=[{js_qcold}], qWall=[{js_qwall}], energy=[{js_energy}];
const monitorIdx=[{js_monitor_indices}], monitorLabels=[{js_monitor_labels}];
const Lmm={L_MM}, TgasHot={T_GAS_HOT_C}, TgasCold={T_GAS_COLD_C}, Tmax={MATERIAL_MAX_TEMP_C};
const colors=['#c9242b','#1f77b4','#7b3294','#e68100','#16803c'];
function range(a){{return [Math.min(...a),Math.max(...a)]}}
function frame(ctx,w,h,title,xlabel,ylabel,ymin,ymax){{ctx.clearRect(0,0,w,h);const p={{l:85,r:35,t:42,b:62,w:w-120,h:h-104}};ctx.strokeStyle='#101820';ctx.lineWidth=1.2;ctx.strokeRect(p.l,p.t,p.w,p.h);ctx.font='13px Arial';ctx.fillStyle='#101820';ctx.fillText(title,p.l,p.t-16);ctx.save();ctx.translate(20,p.t+p.h/2);ctx.rotate(-Math.PI/2);ctx.fillText(ylabel,0,0);ctx.restore();ctx.fillText(xlabel,p.l+p.w/2-70,h-18);for(let j=0;j<=5;j++){{const yy=p.t+p.h-j*p.h/5,val=ymin+j*(ymax-ymin)/5;ctx.strokeStyle='#d6dce4';ctx.beginPath();ctx.moveTo(p.l,yy);ctx.lineTo(p.l+p.w,yy);ctx.stroke();ctx.fillStyle='#101820';ctx.textAlign='right';ctx.fillText(val.toFixed(0),p.l-9,yy+4)}}ctx.textAlign='left';return p}}
function poly(ctx,p,xv,yv,xmin,xmax,ymin,ymax,color,width=2){{ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();for(let i=0;i<xv.length;i++){{const px=p.l+(xv[i]-xmin)/(xmax-xmin)*p.w,py=p.t+p.h-(yv[i]-ymin)/(ymax-ymin)*p.h;i?ctx.lineTo(px,py):ctx.moveTo(px,py)}}ctx.stroke()}}
let current=0, timer=null;
function drawProfile(){{const c=document.getElementById('profile'),ctx=c.getContext('2d'),w=c.width,h=c.height;const p=frame(ctx,w,h,'Temperature profile at selected time','Wall depth, mm','Temperature, °C',0,1000);const row=profiles[current];poly(ctx,p,x,row,0,Lmm,0,1000,'#c9242b',3);ctx.strokeStyle='#16803c';ctx.lineWidth=2;let y=p.t+p.h-(Tmax/1000)*p.h;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(p.l+p.w,y);ctx.stroke();ctx.fillStyle='#16803c';ctx.fillText('Maximum use temperature: '+Tmax.toFixed(0)+' °C',p.l+p.w-225,y-7);ctx.fillStyle='#101820';ctx.fillText('Hot gas: '+TgasHot.toFixed(0)+' °C',p.l+5,p.t+18);ctx.fillText('Cold gas: '+TgasCold.toFixed(0)+' °C',p.l+p.w-95,p.t+p.h-10);document.getElementById('state').textContent='t = '+times[current].toFixed(1)+' s  ('+(times[current]/60).toFixed(2)+' min)';}}
function drawHistory(){{const c=document.getElementById('history'),ctx=c.getContext('2d'),p=frame(ctx,c.width,c.height,'Temperature history','Time, s','Temperature, °C',0,1000);for(let j=0;j<monitorIdx.length;j++){{const vals=profiles.map(row=>row[monitorIdx[j]]);poly(ctx,p,times,vals,0,times[times.length-1],0,1000,colors[j],2);ctx.fillStyle=colors[j];ctx.fillRect(p.l+15+j*150,p.t+14,12,12);ctx.fillStyle='#101820';ctx.fillText(monitorLabels[j],p.l+31+j*150,p.t+24)}}}}
function drawFlux(){{const c=document.getElementById('flux'),ctx=c.getContext('2d');const maxq=Math.ceil(Math.max(...qHot,...qCold,...qWall)/100)*100;const p=frame(ctx,c.width,c.height,'Heat flux and stored energy','Time, s','Heat flux, W/m²',0,maxq);poly(ctx,p,times,qHot,0,times[times.length-1],0,maxq,'#c9242b',2);poly(ctx,p,times,qCold,0,times[times.length-1],0,maxq,'#1f77b4',2);poly(ctx,p,times,qWall,0,times[times.length-1],0,maxq,'#16803c',2);ctx.fillStyle='#c9242b';ctx.fillText('Hot film flux',p.l+15,p.t+20);ctx.fillStyle='#1f77b4';ctx.fillText('Cold film flux',p.l+145,p.t+20);ctx.fillStyle='#16803c';ctx.fillText('Mean wall flux',p.l+280,p.t+20);const emin=Math.min(...energy),emax=Math.max(...energy),scale=emax>emin?(emax-emin):1;ctx.strokeStyle='#7b3294';ctx.setLineDash([6,4]);ctx.lineWidth=2;ctx.beginPath();for(let i=0;i<times.length;i++){{const px=p.l+times[i]/times[times.length-1]*p.w,py=p.t+p.h-(energy[i]-emin)/scale*p.h;i?ctx.lineTo(px,py):ctx.moveTo(px,py)}}ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#7b3294';ctx.fillText('Stored energy, right scale: '+emax.toFixed(0)+' J/m²',p.l+435,p.t+20)}}
function update(){{drawProfile()}}document.getElementById('slider').addEventListener('input',e=>{{current=Number(e.target.value);update()}});document.getElementById('reset').onclick=()=>{{current=0;document.getElementById('slider').value=0;update()}};document.getElementById('play').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;document.getElementById('play').textContent='Play';return}}document.getElementById('play').textContent='Pause';timer=setInterval(()=>{{current=(current+1)%times.length;document.getElementById('slider').value=current;update()}},90)}};
document.getElementById('profile').addEventListener('mousemove',e=>{{const c=e.currentTarget,r=c.getBoundingClientRect(),px=(e.clientX-r.left)*c.width/r.width;const p={{l:85,r:35,t:42,b:62,w:c.width-120,h:c.height-104}};if(px<p.l||px>p.l+p.w)return;const xx=(px-p.l)/p.w*Lmm;let i=0;while(i<x.length-2&&x[i+1]<xx)i++;const rr=(xx-x[i])/(x[i+1]-x[i]),temp=profiles[current][i]+rr*(profiles[current][i+1]-profiles[current][i]);const k0={0.018};const tip='x = '+xx.toFixed(3)+' mm, T = '+temp.toFixed(2)+' °C';c.title=tip;}});
drawProfile();drawHistory();drawFlux();
</script></body></html>"""
with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
    f.write(dashboard)
print(f"Saved {DASHBOARD_FILE}")
print("Done. Open the interactive HTML dashboard in a web browser.")
