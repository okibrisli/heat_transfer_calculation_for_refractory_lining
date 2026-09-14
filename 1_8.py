"""
Multilayer 1D wall conduction model.
Calculates steady state and transient response with Robin boundaries.
Creates CSV files, HTML reports, and an interactive transient profile animation.
Layer order is inside hot face to outside cold face.
"""

import csv
import html
import numpy as np
import heatrapy as htp

# =============================================================================
# USER SETTINGS
# =============================================================================
LAYERS = [
    {"name": "WDS", "thickness_mm": 20.0, "max_temp_C": 950.0},
    {"name": "Al", "thickness_mm": 20.0, "max_temp_C": 660.0},
    # {"name": "DenseRefractory", "thickness_mm": 114.0, "max_temp_C": 1450.0},
]

MATERIALS_PATH = r"C:\Users\orhan.kibrisli\PycharmProjects\heat_transfer\.venv\Lib\site-packages\heatrapy\database\\"
DX_MM = 2.0
DT = 20.0

T_INITIAL_C = 30.0
T_GAS_HOT_C = 900.0
T_GAS_COLD_C = 30.0
H_HOT = 150.0
H_COLD = 12.09

STEADY_CHUNK_S = 5000.0
STEADY_MAX_S = 2000000.0
STEADY_TOL_K = 1.0e-4
TRANSIENT_TOTAL_S = 12000.0
SAVE_INTERVAL_S = 60.0
PICARD_TOL_K = 1.0e-6
PICARD_MAX = 100
COLD_FACE_LIMITS_C = [50.0, 60.0, 80.0, 100.0]

STEADY_CSV = "multilayer_steady_state_profile.csv"
TRANSIENT_CSV = "multilayer_transient_data.csv"
STEADY_REPORT = "multilayer_steady_state_report.html"
TRANSIENT_REPORT = "multilayer_transient_report.html"
TRANSIENT_DASHBOARD = "multilayer_transient_interactive.html"

# =============================================================================
# VALIDATION AND MESH
# =============================================================================
if not LAYERS:
    raise ValueError("LAYERS must contain at least one layer.")
if DX_MM <= 0.0 or DT <= 0.0 or H_HOT <= 0.0 or H_COLD <= 0.0:
    raise ValueError("DX_MM, DT, H_HOT, and H_COLD must be positive.")

for layer in LAYERS:
    if "name" not in layer or "thickness_mm" not in layer or "max_temp_C" not in layer:
        raise ValueError("Every layer needs name, thickness_mm, and max_temp_C.")
    cells = layer["thickness_mm"] / DX_MM
    if layer["thickness_mm"] <= 0.0 or abs(cells - round(cells)) > 1.0e-8:
        raise ValueError(
            f"Layer {layer['name']} thickness must be positive and an integer multiple of DX_MM."
        )

T0 = T_INITIAL_C + 273.15
TgH = T_GAS_HOT_C + 273.15
TgC = T_GAS_COLD_C + 273.15
layer_names = [layer["name"] for layer in LAYERS]
layer_counts = [int(round(layer["thickness_mm"] / DX_MM)) for layer in LAYERS]
n_cells = sum(layer_counts)
dx = DX_MM / 1000.0
L_mm = n_cells * DX_MM
L = L_mm / 1000.0
x_centers_mm = (np.arange(n_cells) + 0.5) * DX_MM
layer_id = np.concatenate([np.full(count, i, dtype=int) for i, count in enumerate(layer_counts)])
interfaces_mm = np.cumsum([layer["thickness_mm"] for layer in LAYERS])[:-1]
PLOT_MAX_TEMP_C = max(layer["max_temp_C"] for layer in LAYERS)

# =============================================================================
# MATERIAL PROPERTY LOADERS
# =============================================================================
material_objects = {}
for name in dict.fromkeys(layer_names):
    loader = htp.SingleObject1D(
        T0,
        materials=(name,),
        borders=(1, 2),
        materials_order=(0,),
        dx=dx,
        dt=DT,
        boundaries=(0, 0),
        materials_path=MATERIALS_PATH,
        file_name=None,
        draw=[],
    )
    material_objects[name] = loader.object.materials[0]


def properties(T_k):
    k = np.empty(n_cells)
    rho = np.empty(n_cells)
    cp = np.empty(n_cells)

    for layer_number, name in enumerate(layer_names):
        indices = np.where(layer_id == layer_number)[0]
        material = material_objects[name]
        k[indices] = [material.k0(float(T_k[i])) for i in indices]
        rho[indices] = [material.rho0(float(T_k[i])) for i in indices]
        cp[indices] = [material.cp0(float(T_k[i])) for i in indices]

    if np.any(k <= 0.0) or np.any(rho <= 0.0) or np.any(cp <= 0.0):
        raise ValueError("Material database returned a nonpositive property.")

    return k, rho, cp


def interface_conductance(k):
    return 1.0 / (dx / (2.0 * k[:-1]) + dx / (2.0 * k[1:]))


def solve_step(T_old, dt_step):
    """Fully implicit cell centered finite volume step with Picard iteration."""
    T_new = T_old.copy()

    for _ in range(PICARD_MAX):
        k, rho, cp = properties(T_new)
        G = interface_conductance(k)
        G_hot = 1.0 / (1.0 / H_HOT + dx / (2.0 * k[0]))
        G_cold = 1.0 / (1.0 / H_COLD + dx / (2.0 * k[-1]))
        capacity = rho * cp * dx / dt_step

        A = np.zeros((n_cells, n_cells))
        b = capacity * T_old

        A[0, 0] = capacity[0] + G_hot + G[0]
        A[0, 1] = -G[0]
        b[0] += G_hot * TgH

        for i in range(1, n_cells - 1):
            A[i, i - 1] = -G[i - 1]
            A[i, i] = capacity[i] + G[i - 1] + G[i]
            A[i, i + 1] = -G[i]

        A[-1, -2] = -G[-1]
        A[-1, -1] = capacity[-1] + G[-1] + G_cold
        b[-1] += G_cold * TgC

        candidate = np.linalg.solve(A, b)
        if np.max(np.abs(candidate - T_new)) < PICARD_TOL_K:
            return candidate
        T_new = candidate

    raise RuntimeError("Picard iteration did not converge. Reduce DT.")


def advance(T_start, duration):
    T = T_start.copy()
    elapsed = 0.0
    while elapsed < duration - 1.0e-12:
        dt_step = min(DT, duration - elapsed)
        T = solve_step(T, dt_step)
        elapsed += dt_step
    return T


def calculate_results(T_k):
    k, _, _ = properties(T_k)
    G = interface_conductance(k)
    G_hot = 1.0 / (1.0 / H_HOT + dx / (2.0 * k[0]))
    G_cold = 1.0 / (1.0 / H_COLD + dx / (2.0 * k[-1]))
    q_hot = G_hot * (TgH - T_k[0])
    q_cold = G_cold * (T_k[-1] - TgC)
    q_wall = G * (T_k[:-1] - T_k[1:])
    T_surface_hot = TgH - q_hot / H_HOT
    T_surface_cold = TgC + q_cold / H_COLD
    return q_hot, q_cold, float(np.mean(q_wall)), k, T_surface_hot, T_surface_cold


def fmt(value, decimals=2):
    return f"{value:.{decimals}f}"


def crossing_time(times, values, threshold):
    hits = np.where(values >= threshold)[0]
    if len(hits) == 0:
        return None
    i = int(hits[0])
    if i == 0:
        return float(times[0])
    return float(times[i - 1] + (threshold - values[i - 1]) * (times[i] - times[i - 1]) / (values[i] - values[i - 1]))

# =============================================================================
# STEADY STATE
# =============================================================================
print(f"Layers: {' | '.join(layer_names)}")
print(f"Total thickness: {L_mm:.2f} mm, cells: {n_cells}, dx: {DX_MM:.2f} mm")

T_ss = np.full(n_cells, T0)
previous = None
t_steady = 0.0
converged = False
max_delta = float("nan")

while t_steady < STEADY_MAX_S:
    chunk = min(STEADY_CHUNK_S, STEADY_MAX_S - t_steady)
    T_ss = advance(T_ss, chunk)
    t_steady += chunk
    if previous is not None:
        max_delta = float(np.max(np.abs(T_ss - previous)))
        if max_delta < STEADY_TOL_K:
            converged = True
            break
    previous = T_ss.copy()

q_hot_ss, q_cold_ss, q_wall_ss, k_ss, Ts_hot_ss, Ts_cold_ss = calculate_results(T_ss)
T_ss_C = T_ss - 273.15

with open(STEADY_CSV, "w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    writer.writerow(["x_mm", "material", "T_C", "k_W_mK"])
    for x_value, layer_number, temp, k_value in zip(x_centers_mm, layer_id, T_ss_C, k_ss):
        writer.writerow([x_value, layer_names[layer_number], temp, k_value])

layer_rows = "".join(
    f"<tr><td>{html.escape(layer['name'])}</td><td>{fmt(layer['thickness_mm'])}</td><td>{fmt(layer['max_temp_C'])}</td></tr>"
    for layer in LAYERS
)
steady_report = f"""<!doctype html><html><head><meta charset='utf-8'><title>Multilayer steady state report</title><style>body{{font-family:Arial;margin:35px;max-width:950px}}table{{border-collapse:collapse;width:100%;margin:10px 0}}th,td{{border:1px solid #777;padding:7px}}th{{background:#dce6f1;text-align:left}}</style></head><body><h1>Multilayer steady state report</h1><p>{'Converged' if converged else 'Maximum calculation time reached'}</p><h2>Layers inside to outside</h2><table><tr><th>Material</th><th>Thickness, mm</th><th>Maximum use temperature, °C</th></tr>{layer_rows}</table><h2>Boundary results</h2><table><tr><th>Quantity</th><th>Inside</th><th>Outside</th></tr><tr><td>Gas temperature</td><td>{fmt(T_GAS_HOT_C,1)} °C</td><td>{fmt(T_GAS_COLD_C,1)} °C</td></tr><tr><td>Surface temperature</td><td>{fmt(Ts_hot_ss-273.15,1)} °C</td><td>{fmt(Ts_cold_ss-273.15,1)} °C</td></tr><tr><td>Heat transfer coefficient</td><td>{fmt(H_HOT,2)} W/m²K</td><td>{fmt(H_COLD,2)} W/m²K</td></tr><tr><td>Heat flux</td><td>{fmt(q_hot_ss,2)} W/m²</td><td>{fmt(q_cold_ss,2)} W/m²</td></tr></table><p>Mean conductive wall heat flux: {fmt(q_wall_ss,2)} W/m²</p></body></html>"""
open(STEADY_REPORT, "w", encoding="utf-8").write(steady_report)
print(f"Saved {STEADY_CSV} and {STEADY_REPORT}")

# =============================================================================
# TRANSIENT
# =============================================================================
T = np.full(n_cells, T0)
t = 0.0
next_save = 0.0
Times, Profiles, Qhot, Qcold, Qwall = [], [], [], [], []

while t < TRANSIENT_TOTAL_S - 1.0e-10:
    if t >= next_save - 1.0e-10:
        qh, qc, qw, _, _, _ = calculate_results(T)
        Times.append(t)
        Profiles.append((T - 273.15).copy())
        Qhot.append(qh)
        Qcold.append(qc)
        Qwall.append(qw)
        next_save += SAVE_INTERVAL_S
    else:
        dt_step = min(DT, next_save - t, TRANSIENT_TOTAL_S - t)
        T = solve_step(T, dt_step)
        t += dt_step

if abs(Times[-1] - TRANSIENT_TOTAL_S) > 1.0e-8:
    qh, qc, qw, _, _, _ = calculate_results(T)
    Times.append(TRANSIENT_TOTAL_S)
    Profiles.append((T - 273.15).copy())
    Qhot.append(qh)
    Qcold.append(qc)
    Qwall.append(qw)

Times = np.asarray(Times)
Profiles = np.asarray(Profiles)
Qhot = np.asarray(Qhot)
Qcold = np.asarray(Qcold)
Qwall = np.asarray(Qwall)

with open(TRANSIENT_CSV, "w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    writer.writerow(["time_s", "q_hot_W_m2", "q_cold_W_m2", "q_wall_W_m2"] + [f"T_{value:.3f}_mm_C" for value in x_centers_mm])
    for i in range(len(Times)):
        writer.writerow([Times[i], Qhot[i], Qcold[i], Qwall[i], *Profiles[i]])

_, _, _, _, Ts_hot_final, Ts_cold_final = calculate_results(T)
limit_rows = ""
for limit in COLD_FACE_LIMITS_C:
    value = crossing_time(Times, Profiles[:, -1], limit)
    result = "Not reached" if value is None else f"{value:.1f} s, {value/60.0:.2f} min"
    limit_rows += f"<tr><td>{limit:.1f} °C</td><td>{result}</td></tr>"

transient_report = f"""<!doctype html><html><head><meta charset='utf-8'><title>Multilayer transient report</title><style>body{{font-family:Arial;margin:35px;max-width:950px}}table{{border-collapse:collapse;width:100%;margin:10px 0}}th,td{{border:1px solid #777;padding:7px}}th{{background:#dce6f1;text-align:left}}</style></head><body><h1>Multilayer transient report</h1><p>Simulation duration: {fmt(Times[-1],1)} s. Saved states: {len(Times)}.</p><h2>Final transient state</h2><table><tr><th>Quantity</th><th>Value</th></tr><tr><td>Hot surface temperature</td><td>{fmt(Ts_hot_final-273.15,2)} °C</td></tr><tr><td>Cold surface temperature</td><td>{fmt(Ts_cold_final-273.15,2)} °C</td></tr><tr><td>Hot film heat flux</td><td>{fmt(Qhot[-1],2)} W/m²</td></tr><tr><td>Cold film heat flux</td><td>{fmt(Qcold[-1],2)} W/m²</td></tr></table><h2>Cold face temperature limits</h2><table><tr><th>Limit</th><th>First crossing time</th></tr>{limit_rows}</table><p>Open {TRANSIENT_DASHBOARD} for the interactive profile animation.</p></body></html>"""
open(TRANSIENT_REPORT, "w", encoding="utf-8").write(transient_report)
print(f"Saved {TRANSIENT_CSV} and {TRANSIENT_REPORT}")

# =============================================================================
# INTERACTIVE TRANSIENT ANIMATION
# =============================================================================
js_times = ",".join(f"{v:.9g}" for v in Times)
js_x = ",".join(f"{v:.9g}" for v in x_centers_mm)
js_profiles = ",".join("[" + ",".join(f"{v:.9g}" for v in row) + "]" for row in Profiles)
js_interfaces = ",".join(f"{v:.9g}" for v in interfaces_mm)

page = r'''<!doctype html><html><head><meta charset="utf-8"><style>body{font-family:Arial;background:#f3f5f7;margin:0}main{width:1440px;margin:20px auto;padding:25px;background:#fff}canvas{border:1px solid #777}.controls{margin:12px 0;display:flex;gap:12px;align-items:center}input{width:750px}#readout{padding:10px;background:#f3f7fb;border-left:4px solid #1f4e79;font-family:Consolas}</style></head><body><main><h1>Multilayer transient temperature profile</h1><p>__LAYERS__</p><canvas id="c" width="1400" height="600"></canvas><div class="controls"><button id="play">Play</button><button id="reset">Reset</button><input id="slider" type="range" min="0" max="__MAX__" value="0"><b id="time"></b></div><div id="readout">Move over the plot for local temperature.</div><script>const times=[__TIMES__],xs=[__X__],profiles=[__PROFILES__],interfaces=[__INTERFACES__],L=__L__,hot=__HOT__,cold=__COLD__,maxUse=__MAXUSE__;let n=0,timer=null;const c=document.getElementById('c'),g=c.getContext('2d'),slider=document.getElementById('slider'),time=document.getElementById('time'),readout=document.getElementById('readout'),p={l:180,r:1040,t:90,b:505};const X=x=>p.l+x/L*(p.r-p.l),Y=T=>p.b-T/1000*(p.b-p.t);function text(v,x,y,a){g.font='13px Arial';g.fillStyle='#14202b';g.textAlign=a||'left';g.fillText(v,x,y)}function line(x1,y1,x2,y2,color,w,d){g.strokeStyle=color;g.lineWidth=w;g.setLineDash(d?[5,5]:[]);g.beginPath();g.moveTo(x1,y1);g.lineTo(x2,y2);g.stroke();g.setLineDash([])}function draw(){g.clearRect(0,0,1400,600);for(let T=0;T<=1000;T+=100){line(p.l,Y(T),p.r,Y(T),'#ddd',1);text(T,p.l-12,Y(T)+5,'right')}for(let x=0;x<=L+1e-9;x+=L/4){line(X(x),p.t,X(x),p.b,'#ddd',1);text(x.toFixed(1),X(x),p.b+28,'center')}g.strokeRect(p.l,p.t,p.r-p.l,p.b-p.t);line(p.l,Y(maxUse),p.r,Y(maxUse),'#16803c',3);text('Maximum use: '+maxUse+' °C',p.r+18,Y(maxUse)+5);interfaces.forEach(x=>line(X(x),p.t,X(x),p.b,'#555',1,true));let row=profiles[n];g.strokeStyle='#c9242b';g.lineWidth=3;g.beginPath();row.forEach((T,i)=>i?g.lineTo(X(xs[i]),Y(T)):g.moveTo(X(xs[i]),Y(T)));g.stroke();text('Hot gas: '+hot+' °C',p.l-48,Y(hot)-8,'right');text('Cold gas: '+cold+' °C',p.r+90,Y(cold)+5);text('Depth from inside to outside, mm',(p.l+p.r)/2,550,'center');time.textContent='t = '+times[n].toFixed(1)+' s  ('+(times[n]/60).toFixed(2)+' min)'}function localT(x){let row=profiles[n];for(let i=0;i<xs.length-1;i++){if(x<=xs[i+1]){let r=(x-xs[i])/(xs[i+1]-xs[i]);return row[i]+r*(row[i+1]-row[i])}}return row[row.length-1]}c.onmousemove=e=>{let r=c.getBoundingClientRect(),px=(e.clientX-r.left)*1400/r.width;if(px<p.l||px>p.r)return;let x=(px-p.l)/(p.r-p.l)*L;readout.textContent='t = '+times[n].toFixed(1)+' s    x = '+x.toFixed(3)+' mm    T = '+localT(x).toFixed(2)+' °C'};slider.oninput=()=>{n=+slider.value;draw()};document.getElementById('reset').onclick=()=>{n=0;slider.value=0;draw()};document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else{timer=setInterval(()=>{n=(n+1)%times.length;slider.value=n;draw()},90)}};draw();</script></main></body></html>'''
page = (page.replace("__LAYERS__", html.escape(" | ".join(f"{item['name']} {item['thickness_mm']} mm" for item in LAYERS)))
        .replace("__MAX__", str(len(Times)-1)).replace("__TIMES__", js_times)
        .replace("__X__", js_x).replace("__PROFILES__", js_profiles)
        .replace("__INTERFACES__", js_interfaces).replace("__L__", str(L_mm))
        .replace("__HOT__", str(T_GAS_HOT_C)).replace("__COLD__", str(T_GAS_COLD_C))
        .replace("__MAXUSE__", str(PLOT_MAX_TEMP_C)))
open(TRANSIENT_DASHBOARD, "w", encoding="utf-8").write(page)
print(f"Saved {TRANSIENT_DASHBOARD}")
print("Done.")

