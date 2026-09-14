"""
Complete standalone WDS calculation.
Creates steady state and transient CSV data, two HTML reports, and an
interactive transient profile animation. No separate append file is required.
"""

import csv
import html
import numpy as np
import heatrapy as htp

# =============================================================================
# USER SETTINGS
# =============================================================================
MATERIAL_NAME = "WDS"
MATERIALS_PATH = (r"C:\Users\orhan.kibrisli\PycharmProjects\heat_transfer\.venv"
                  r"\Lib\site-packages\heatrapy\database\\")
MATERIAL_MAX_TEMP_C = 950.0

L_MM = 200.0
DX_MM = 10.0
DT = 50.0
T_INITIAL_C = 30.0
T_GAS_HOT_C = 900.0
T_GAS_COLD_C = 30.0
H_HOT = 150.0
H_COLD = 12.09

STEADY_CHUNK_TIME_S = 5000.0
STEADY_MAX_TIME_S = 2000000.0
STEADY_TOLERANCE_K = 1.0e-4
TRANSIENT_TOTAL_TIME_S = 12000.0
TRANSIENT_OUTPUT_INTERVAL_S = 60.0
COLD_FACE_LIMITS_C = [50.0, 60.0, 80.0, 100.0]
PICARD_TOLERANCE_K = 1.0e-6
PICARD_MAX_ITERATIONS = 100

STEADY_CSV = "wds_steady_state_profile.csv"
STEADY_REPORT = "wds_steady_state_report.html"
TRANSIENT_CSV = "wds_transient_data.csv"
TRANSIENT_REPORT = "wds_transient_report.html"
TRANSIENT_DASHBOARD = "wds_transient_interactive.html"

# =============================================================================
# FINITE VOLUME SOLVER
# =============================================================================
L = L_MM / 1000.0
if L_MM <= 0 or DX_MM <= 0 or DT <= 0 or H_HOT <= 0 or H_COLD <= 0:
    raise ValueError("Geometry, time step, and film coefficients must be positive.")

T_INITIAL = T_INITIAL_C + 273.15
T_GAS_HOT = T_GAS_HOT_C + 273.15
T_GAS_COLD = T_GAS_COLD_C + 273.15
n_intervals = int(round(L / (DX_MM / 1000.0)))
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


def props(T):
    k = np.array([material.k0(float(v)) for v in T], dtype=float)
    rho = np.array([material.rho0(float(v)) for v in T], dtype=float)
    cp = np.array([material.cp0(float(v)) for v in T], dtype=float)
    if np.any(k <= 0) or np.any(rho <= 0) or np.any(cp <= 0):
        raise ValueError("WDS database returned a nonpositive property.")
    return k, rho, cp


def hmean(a, b):
    return 2.0 * a * b / (a + b)


def solve_step(T_old, dt):
    T = T_old.copy()
    for _ in range(PICARD_MAX_ITERATIONS):
        k, rho, cp = props(T)
        G = hmean(k[:-1], k[1:]) / DX
        C = rho * cp * cv_width / dt
        A = np.zeros((n_nodes, n_nodes), dtype=float)
        b = C * T_old
        A[0, 0], A[0, 1] = C[0] + H_HOT + G[0], -G[0]
        b[0] += H_HOT * T_GAS_HOT
        for i in range(1, n_nodes - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = -G[i - 1], C[i] + G[i - 1] + G[i], -G[i]
        A[-1, -2], A[-1, -1] = -G[-1], C[-1] + G[-1] + H_COLD
        b[-1] += H_COLD * T_GAS_COLD
        candidate = np.linalg.solve(A, b)
        if np.max(np.abs(candidate - T)) < PICARD_TOLERANCE_K:
            return candidate
        T = candidate
    raise RuntimeError("Picard iteration failed. Reduce DT.")


def advance(T_start, seconds):
    T = T_start.copy()
    elapsed = 0.0
    while elapsed < seconds - 1.0e-12:
        dtime = min(DT, seconds - elapsed)
        T = solve_step(T, dtime)
        elapsed += dtime
    return T


def fluxes(T):
    k, _, _ = props(T)
    qwall = -hmean(k[:-1], k[1:]) * np.diff(T) / DX
    qhot = H_HOT * (T_GAS_HOT - T[0])
    qcold = H_COLD * (T[-1] - T_GAS_COLD)
    return qhot, qcold, float(np.mean(qwall)), k


def fmt(v, n=2):
    return f"{v:.{n}f}"


def crossing_time(t, y, limit):
    found = np.where(y >= limit)[0]
    if len(found) == 0:
        return None
    i = int(found[0])
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (limit - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1]))

# =============================================================================
# STEADY STATE
# =============================================================================
print(f"Material: {MATERIAL_NAME}")
print(f"Mesh: {n_nodes} nodes, {n_intervals} intervals, dx = {DX * 1000.0:.4f} mm")
print("Running steady state calculation")
T_ss = np.full(n_nodes, T_INITIAL)
previous = None
t_ss = 0.0
converged = False
max_delta = float("nan")
while t_ss < STEADY_MAX_TIME_S:
    interval = min(STEADY_CHUNK_TIME_S, STEADY_MAX_TIME_S - t_ss)
    T_ss = advance(T_ss, interval)
    t_ss += interval
    if previous is not None:
        max_delta = float(np.max(np.abs(T_ss - previous)))
        if max_delta < STEADY_TOLERANCE_K:
            converged = True
            break
    previous = T_ss.copy()

qh_ss, qc_ss, qw_ss, k_ss = fluxes(T_ss)
TssC = T_ss - 273.15
with open(STEADY_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["x_mm", "T_C", "k_W_mK"])
    w.writerows(zip(x_mm, TssC, k_ss))

steady_rows = "".join("<tr><td>%.2f</td><td>%.2f</td><td>%.5f</td></tr>" % (x, T, k) for x, T, k in zip(x_mm, TssC, k_ss))
steady_html = """<!doctype html><html><head><meta charset="utf-8"><title>Steady state report</title><style>body{font-family:Arial;margin:35px;max-width:950px;color:#14202b}h1{font-size:25px}h2{border-bottom:2px solid #1f4e79;padding-bottom:4px}table{border-collapse:collapse;width:100%;margin:10px 0}th,td{border:1px solid #7f8c8d;padding:7px}th{background:#dce6f1;text-align:left}.grid{display:grid;grid-template-columns:1fr 1fr;gap:25px}</style></head><body>
<h1>Steady state wall heat loss report</h1><p><b>STATUS</b></p><div class="grid"><div><h2>Boundary conditions</h2><table><tr><th>Parameter</th><th>Inside</th><th>Outside</th></tr><tr><td>Gas temperature</td><td>HOTGAS °C</td><td>COLDGAS °C</td></tr><tr><td>Surface temperature</td><td>HOTSURF °C</td><td>COLDSURF °C</td></tr><tr><td>Heat transfer coefficient</td><td>HHOT W/m²K</td><td>HCOLD W/m²K</td></tr><tr><td>Heat flux</td><td>QHOT W/m²</td><td>QCOLD W/m²</td></tr></table></div><div><h2>Wall</h2><table><tr><th>Item</th><th>Value</th></tr><tr><td>Thickness</td><td>THICK mm</td></tr><tr><td>Heat loss</td><td>QWALL W/m²</td></tr><tr><td>Mesh step</td><td>DX mm</td></tr><tr><td>Solver</td><td>Implicit FVM with Picard iteration</td></tr></table></div></div><h2>Temperature profile</h2><table><tr><th>x, mm</th><th>Temperature, °C</th><th>k, W/mK</th></tr>ROWS</table></body></html>"""
steady_html = (steady_html.replace("STATUS", "Converged" if converged else "Maximum calculation time reached")
               .replace("HOTGAS", fmt(T_GAS_HOT_C, 1)).replace("COLDGAS", fmt(T_GAS_COLD_C, 1))
               .replace("HOTSURF", fmt(TssC[0], 1)).replace("COLDSURF", fmt(TssC[-1], 1))
               .replace("HHOT", fmt(H_HOT, 2)).replace("HCOLD", fmt(H_COLD, 2))
               .replace("QHOT", fmt(qh_ss, 2)).replace("QCOLD", fmt(qc_ss, 2))
               .replace("THICK", fmt(L_MM, 2)).replace("QWALL", fmt(qw_ss, 2))
               .replace("DX", fmt(DX * 1000.0, 3)).replace("ROWS", steady_rows))
with open(STEADY_REPORT, "w", encoding="utf-8") as f:
    f.write(steady_html)
print(f"Saved {STEADY_CSV} and {STEADY_REPORT}")

# =============================================================================
# TRANSIENT
# =============================================================================
print("Running transient calculation")
T = np.full(n_nodes, T_INITIAL)
t = 0.0
next_save = 0.0
Times, Profiles, Qhot, Qcold, Qwall = [], [], [], [], []
while t < TRANSIENT_TOTAL_TIME_S - 1.0e-10:
    if t >= next_save - 1.0e-10:
        qh, qc, qw, _ = fluxes(T)
        Times.append(t); Profiles.append((T - 273.15).copy()); Qhot.append(qh); Qcold.append(qc); Qwall.append(qw)
        next_save += TRANSIENT_OUTPUT_INTERVAL_S
    else:
        dtime = min(DT, next_save - t, TRANSIENT_TOTAL_TIME_S - t)
        T = solve_step(T, dtime)
        t += dtime

if abs(Times[-1] - TRANSIENT_TOTAL_TIME_S) > 1.0e-8:
    qh, qc, qw, _ = fluxes(T)
    Times.append(TRANSIENT_TOTAL_TIME_S); Profiles.append((T - 273.15).copy()); Qhot.append(qh); Qcold.append(qc); Qwall.append(qw)
Times, Profiles = np.asarray(Times), np.asarray(Profiles)
Qhot, Qcold, Qwall = np.asarray(Qhot), np.asarray(Qcold), np.asarray(Qwall)

with open(TRANSIENT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["time_s", "q_hot_W_m2", "q_cold_W_m2", "q_wall_W_m2"] + [f"T_x_{x:.3f}_mm_C" for x in x_mm])
    for i in range(len(Times)):
        w.writerow([Times[i], Qhot[i], Qcold[i], Qwall[i], *Profiles[i]])

limit_rows = ""
for limit in COLD_FACE_LIMITS_C:
    value = crossing_time(Times, Profiles[:, -1], limit)
    text = "Not reached" if value is None else f"{value:.1f} s, {value / 60.0:.2f} min"
    limit_rows += f"<tr><td>{limit:.1f} °C</td><td>{text}</td></tr>"
transient_html = """<!doctype html><html><head><meta charset="utf-8"><title>Transient report</title><style>body{font-family:Arial;margin:35px;max-width:950px;color:#14202b}h1{font-size:25px}h2{border-bottom:2px solid #1f4e79;padding-bottom:4px}table{border-collapse:collapse;width:100%;margin:10px 0}th,td{border:1px solid #7f8c8d;padding:7px}th{background:#dce6f1;text-align:left}.grid{display:grid;grid-template-columns:1fr 1fr;gap:25px}.note{padding:10px;background:#f3f7fb;border-left:4px solid #1f4e79}</style></head><body><h1>Transient wall heat transfer report</h1><div class="grid"><div><h2>Calculation</h2><table><tr><th>Item</th><th>Value</th></tr><tr><td>Total time</td><td>TOTAL s</td></tr><tr><td>Time step</td><td>DTIME s</td></tr><tr><td>Saved interval</td><td>INTERVAL s</td></tr><tr><td>Saved states</td><td>NSTATE</td></tr></table></div><div><h2>Final state</h2><table><tr><th>Item</th><th>Value</th></tr><tr><td>Hot surface</td><td>HOTS °C</td></tr><tr><td>Cold surface</td><td>COLDS °C</td></tr><tr><td>Hot film flux</td><td>QH W/m²</td></tr><tr><td>Cold film flux</td><td>QC W/m²</td></tr></table></div></div><h2>Cold face threshold crossings</h2><table><tr><th>Threshold</th><th>First crossing</th></tr>LIMITS</table><div class="note">Open wds_transient_interactive.html for the animated temperature profile and live coordinate readout.</div></body></html>"""
transient_html = (transient_html.replace("TOTAL", fmt(Times[-1], 1)).replace("DTIME", fmt(DT, 2))
                  .replace("INTERVAL", fmt(TRANSIENT_OUTPUT_INTERVAL_S, 1)).replace("NSTATE", str(len(Times)))
                  .replace("HOTS", fmt(Profiles[-1, 0], 2)).replace("COLDS", fmt(Profiles[-1, -1], 2))
                  .replace("QH", fmt(Qhot[-1], 2)).replace("QC", fmt(Qcold[-1], 2)).replace("LIMITS", limit_rows))
with open(TRANSIENT_REPORT, "w", encoding="utf-8") as f:
    f.write(transient_html)
print(f"Saved {TRANSIENT_CSV} and {TRANSIENT_REPORT}")

# =============================================================================
# INTERACTIVE TRANSIENT HTML
# The template uses replacement tokens instead of a Python f string. This is
# intentional, because JavaScript braces then cannot cause an f string error.
# =============================================================================

js_times = ",".join(f"{v:.12g}" for v in Times)
js_x = ",".join(f"{v:.12g}" for v in x_mm)
js_profiles = ",".join(
    "[" + ",".join(f"{v:.12g}" for v in row) + "]"
    for row in Profiles
)

dashboard = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Transient animation</title>

<style>
body {
    margin: 0;
    background: #f3f5f7;
    font-family: Arial, Helvetica, sans-serif;
    color: #14202b;
}

main {
    width: 1440px;
    margin: 20px auto;
    padding: 26px;
    background: white;
    box-shadow: 0 1px 7px #aeb7c0;
}

h1 {
    margin: 0 0 6px;
    font-size: 25px;
}

canvas {
    display: block;
    border: 1px solid #65717d;
    background: white;
    cursor: crosshair;
}

.controls {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 13px 0;
}

button {
    min-width: 76px;
    padding: 7px 12px;
    cursor: pointer;
}

input {
    width: 760px;
}

#timevalue {
    min-width: 190px;
    font-weight: bold;
}

#readout {
    padding: 10px;
    background: #f3f7fb;
    border-left: 4px solid #1f4e79;
    font-family: Consolas, monospace;
}
</style>
</head>

<body>
<main>

<h1>Transient temperature profile animation</h1>

<p>__MAT__ Ultra, thickness __L__ mm</p>

<canvas id="graph" width="1400" height="600"></canvas>

<div class="controls">
    <button id="play">Play</button>
    <button id="reset">Reset</button>

    <input
        id="slider"
        type="range"
        min="0"
        max="__MAXINDEX__"
        value="0"
    >

    <span id="timevalue"></span>
</div>

<div id="readout">
    Move the pointer over the graph to inspect wall position and temperature.
</div>

<script>

const times = [__TIMES__];
const xdata = [__X__];
const profiles = [__PROFILES__];

const L = __L__;
const hotGas = __HOTGAS__;
const coldGas = __COLDGAS__;
const maxUse = __MAXUSE__;

const canvas = document.getElementById("graph");
const g = canvas.getContext("2d");

const slider = document.getElementById("slider");
const timeValue = document.getElementById("timevalue");
const readout = document.getElementById("readout");
const playButton = document.getElementById("play");

let index = 0;
let timer = null;

/*
Large left and right margins are intentional.

Left margin:
Hot gas and hot surface labels sit fully outside the plot.

Right margin:
Cold surface, cold gas, and WDS maximum use temperature labels sit fully
outside the plot without clipping.
*/
const p = {
    l: 180,
    r: 1040,
    t: 100,
    b: 515
};

function X(x) {
    return p.l + x / L * (p.r - p.l);
}

function Y(T) {
    return p.b - T / 1000 * (p.b - p.t);
}

function line(x1, y1, x2, y2, color, width, dashed) {
    g.strokeStyle = color;
    g.lineWidth = width;
    g.setLineDash(dashed ? [7, 5] : []);

    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(x2, y2);
    g.stroke();

    g.setLineDash([]);
}

function text(value, x, y, size, align) {
    g.fillStyle = "#14202b";
    g.font = size + "px Arial";
    g.textAlign = align || "left";
    g.fillText(value, x, y);
}

function draw() {
    g.clearRect(0, 0, canvas.width, canvas.height);

    g.fillStyle = "white";
    g.fillRect(0, 0, canvas.width, canvas.height);

    text(
        "Temperature profile at selected transient time",
        p.l,
        35,
        21
    );

    text(
        "Temperature, °C",
        p.l,
        65,
        15
    );

    for (let T = 0; T <= 1000; T += 100) {
        const y = Y(T);

        line(
            p.l,
            y,
            p.r,
            y,
            "#d6dce4",
            1,
            false
        );

        text(
            String(T),
            p.l - 14,
            y + 5,
            12,
            "right"
        );
    }

    for (let x = 0; x <= L + 1e-9; x += L / 4) {
        const px = X(x);

        line(
            px,
            p.t,
            px,
            p.b,
            "#d6dce4",
            1,
            false
        );

        text(
            x.toFixed(1),
            px,
            p.b + 30,
            12,
            "center"
        );
    }

    g.strokeStyle = "#14202b";
    g.lineWidth = 1.5;
    g.strokeRect(
        p.l,
        p.t,
        p.r - p.l,
        p.b - p.t
    );

    line(
        p.l,
        Y(maxUse),
        p.r,
        Y(maxUse),
        "#16803c",
        3,
        false
    );

    text(
        "WDS maximum use temperature: " +
        maxUse.toFixed(0) +
        " °C",
        p.r + 18,
        Y(maxUse) + 5,
        13
    );

    const row = profiles[index];
    const hotSurface = row[0];
    const coldSurface = row[row.length - 1];

    line(
        p.l - 65,
        Y(hotGas),
        p.l,
        Y(hotSurface),
        "#65717d",
        2,
        true
    );

    line(
        p.r,
        Y(coldSurface),
        p.r + 65,
        Y(coldGas),
        "#65717d",
        2,
        true
    );

    g.strokeStyle = "#c9242b";
    g.lineWidth = 3;

    g.beginPath();

    for (let i = 0; i < xdata.length; i++) {
        if (i === 0) {
            g.moveTo(
                X(xdata[i]),
                Y(row[i])
            );
        } else {
            g.lineTo(
                X(xdata[i]),
                Y(row[i])
            );
        }
    }

    g.stroke();

    line(
        p.l,
        Y(hotGas),
        p.l - 42,
        Y(hotGas),
        "#65717d",
        1.5,
        false
    );

    text(
        "Hot gas: " +
        hotGas.toFixed(0) +
        " °C",
        p.l - 50,
        Y(hotGas) - 10,
        13,
        "right"
    );

    line(
        p.l,
        Y(hotSurface),
        p.l - 42,
        Y(hotSurface) + 28,
        "#65717d",
        1.5,
        false
    );

    text(
        "Hot surface: " +
        hotSurface.toFixed(1) +
        " °C",
        p.l - 50,
        Y(hotSurface) + 35,
        13,
        "right"
    );

    line(
        p.r,
        Y(coldSurface),
        p.r + 45,
        Y(coldSurface) - 28,
        "#65717d",
        1.5,
        false
    );

    text(
        "Cold surface: " +
        coldSurface.toFixed(1) +
        " °C",
        p.r + 53,
        Y(coldSurface) - 34,
        13
    );

    line(
        p.r + 65,
        Y(coldGas),
        p.r + 92,
        Y(coldGas),
        "#65717d",
        1.5,
        false
    );

    text(
        "Cold gas: " +
        coldGas.toFixed(0) +
        " °C",
        p.r + 100,
        Y(coldGas) + 5,
        13
    );

    text(
        "Wall depth from inside to outside, mm",
        (p.l + p.r) / 2,
        canvas.height - 24,
        15,
        "center"
    );

    timeValue.textContent =
        "t = " +
        times[index].toFixed(1) +
        " s  (" +
        (times[index] / 60).toFixed(2) +
        " min)";
}

function localTemperature(x) {
    const row = profiles[index];

    if (x <= xdata[0]) {
        return row[0];
    }

    if (x >= xdata[xdata.length - 1]) {
        return row[row.length - 1];
    }

    for (let i = 0; i < xdata.length - 1; i++) {
        if (x <= xdata[i + 1]) {
            const ratio =
                (x - xdata[i]) /
                (xdata[i + 1] - xdata[i]);

            return row[i] +
                ratio * (row[i + 1] - row[i]);
        }
    }

    return row[row.length - 1];
}

canvas.addEventListener("mousemove", function(event) {
    const rect = canvas.getBoundingClientRect();

    const px =
        (event.clientX - rect.left) *
        canvas.width /
        rect.width;

    if (px < p.l || px > p.r) {
        return;
    }

    const x =
        (px - p.l) /
        (p.r - p.l) *
        L;

    const T = localTemperature(x);

    readout.textContent =
        "t = " +
        times[index].toFixed(1) +
        " s    x = " +
        x.toFixed(3) +
        " mm    T = " +
        T.toFixed(2) +
        " °C";
});

canvas.addEventListener("mouseleave", function() {
    readout.textContent =
        "Move the pointer over the graph to inspect wall position and temperature.";
});

slider.addEventListener("input", function() {
    index = Number(slider.value);
    draw();
});

document.getElementById("reset").onclick = function() {
    index = 0;
    slider.value = 0;
    draw();
};

playButton.onclick = function() {
    if (timer !== null) {
        clearInterval(timer);
        timer = null;
        playButton.textContent = "Play";
        return;
    }

    playButton.textContent = "Pause";

    timer = setInterval(function() {
        index = (index + 1) % times.length;
        slider.value = index;
        draw();
    }, 90);
};

draw();
</script>

</main>
</body>
</html>'''

dashboard = (
    dashboard
    .replace("__MAT__", html.escape(MATERIAL_NAME))
    .replace("__L__", str(L_MM))
    .replace("__MAXINDEX__", str(len(Times) - 1))
    .replace("__TIMES__", js_times)
    .replace("__X__", js_x)
    .replace("__PROFILES__", js_profiles)
    .replace("__HOTGAS__", str(T_GAS_HOT_C))
    .replace("__COLDGAS__", str(T_GAS_COLD_C))
    .replace("__MAXUSE__", str(MATERIAL_MAX_TEMP_C))
)

with open(TRANSIENT_DASHBOARD, "w", encoding="utf-8") as f:
    f.write(dashboard)

print(f"Saved {TRANSIENT_DASHBOARD}")