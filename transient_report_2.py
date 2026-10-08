"""
Multilayer wall transient reporting.
Reads wds_transient_data.csv and wds_transient_meta.json, then writes
the static HTML report and the interactive animation.
"""

import csv
import html
import json
import numpy as np

# =============================================================================
# REPORT SETTINGS
# =============================================================================
TRANSIENT_CSV = "wds_transient_data.csv"
TRANSIENT_META = "wds_transient_meta.json"
TRANSIENT_REPORT = "wds_transient_report.html"
TRANSIENT_DASHBOARD = "wds_transient_interactive.html"
COLD_FACE_LIMITS_C = [50.0, 60.0, 80.0, 100.0]

# =============================================================================
# LOAD DATA
# =============================================================================
with open(TRANSIENT_META, "r", encoding="utf-8") as f:
    meta = json.load(f)

with open(TRANSIENT_CSV, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = np.array([[float(v) for v in row] for row in reader], dtype=float)

T_START = header.index("h_cold_W_m2K") + 1
x_mm = np.array([float(h.split("_")[2]) for h in header[T_START:]])
Times = data[:, 0]
Qhot = data[:, 1]
Qcold = data[:, 2]
Qwall = data[:, 3]
Hhot = data[:, 4]
Hcold = data[:, 5]
Profiles = data[:, T_START:]

MATERIAL_NAME = meta["material_name"]
LAYERS = meta["layers"]
L_MM = meta["l_mm"]
DT = meta["dt_s"]
T_GAS_HOT_C = meta["t_gas_hot_c"]
T_GAS_COLD_C = meta["t_gas_cold_c"]
OUTPUT_INTERVAL = meta["output_interval_s"]


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


def layer_max_temperature(layer):
    mask = (x_mm >= layer["start_mm"] - 1e-6) & (x_mm <= layer["end_mm"] + 1e-6)
    return float(np.max(Profiles[:, mask]))

# =============================================================================
# STATIC TRANSIENT REPORT
# =============================================================================
limit_rows = ""
for limit in COLD_FACE_LIMITS_C:
    value = crossing_time(Times, Profiles[:, -1], limit)
    text = "Not reached" if value is None else f"{value:.1f} s, {value / 60.0:.2f} min"
    limit_rows += f"<tr><td>{limit:.1f} °C</td><td>{text}</td></tr>"

layer_rows = ""
for lay in LAYERS:
    peak = layer_max_temperature(lay)
    status = "OK" if peak <= lay["max_temp_c"] else "Exceeded"
    layer_rows += (
        f"<tr><td>{html.escape(lay['name'])}</td>"
        f"<td>{lay['start_mm']:.1f} to {lay['end_mm']:.1f}</td>"
        f"<td>{lay['max_temp_c']:.0f}</td>"
        f"<td>{peak:.1f}</td><td>{status}</td></tr>"
    )

film_rows = (
    f"<tr><td>Inside film</td><td>{html.escape(meta.get('film_model_hot', ''))}</td>"
    f"<td>{Hhot[0]:.2f}</td><td>{Hhot[-1]:.2f}</td></tr>"
    f"<tr><td>Outside film</td><td>{html.escape(meta.get('film_model_cold', ''))}</td>"
    f"<td>{Hcold[0]:.2f}</td><td>{Hcold[-1]:.2f}</td></tr>"
)

transient_html = """<!doctype html><html><head><meta charset="utf-8"><title>Transient report</title><style>body{font-family:Arial;margin:35px;max-width:950px;color:#14202b}h1{font-size:25px}h2{border-bottom:2px solid #1f4e79;padding-bottom:4px}table{border-collapse:collapse;width:100%;margin:10px 0}th,td{border:1px solid #7f8c8d;padding:7px}th{background:#dce6f1;text-align:left}.grid{display:grid;grid-template-columns:1fr 1fr;gap:25px}.note{padding:10px;background:#f3f7fb;border-left:4px solid #1f4e79}</style></head><body><h1>Transient wall heat transfer report</h1><p>Wall: __MAT__, total thickness __L__ mm</p><div class="grid"><div><h2>Calculation</h2><table><tr><th>Item</th><th>Value</th></tr><tr><td>Total time</td><td>__TOTAL__ s</td></tr><tr><td>Time step</td><td>__DTIME__ s</td></tr><tr><td>Saved interval</td><td>__INTERVAL__ s</td></tr><tr><td>Saved states</td><td>__NSTATE__</td></tr></table></div><div><h2>Final state</h2><table><tr><th>Item</th><th>Value</th></tr><tr><td>Hot surface</td><td>__HOTS__ °C</td></tr><tr><td>Cold surface</td><td>__COLDS__ °C</td></tr><tr><td>Hot film flux</td><td>__QH__ W/m²</td></tr><tr><td>Cold film flux</td><td>__QC__ W/m²</td></tr></table></div></div><h2>Boundary films</h2><table><tr><th>Film</th><th>Model</th><th>Start, W/m²K</th><th>End, W/m²K</th></tr>__FILMS__</table><h2>Layers and temperature limits</h2><table><tr><th>Layer</th><th>Position, mm</th><th>Max use, °C</th><th>Peak, °C</th><th>Status</th></tr>__LAYERS__</table><h2>Cold face threshold crossings</h2><table><tr><th>Threshold</th><th>First crossing</th></tr>__LIMITS__</table><div class="note">Open wds_transient_interactive.html for the animated temperature profile and live coordinate readout.</div></body></html>"""

transient_html = (transient_html
                  .replace("__MAT__", html.escape(MATERIAL_NAME))
                  .replace("__L__", fmt(L_MM, 1))
                  .replace("__TOTAL__", fmt(Times[-1], 1))
                  .replace("__DTIME__", fmt(DT, 2))
                  .replace("__INTERVAL__", fmt(OUTPUT_INTERVAL, 1))
                  .replace("__NSTATE__", str(len(Times)))
                  .replace("__HOTS__", fmt(Profiles[-1, 0], 2))
                  .replace("__COLDS__", fmt(Profiles[-1, -1], 2))
                  .replace("__QH__", fmt(Qhot[-1], 2))
                  .replace("__QC__", fmt(Qcold[-1], 2))
                  .replace("__FILMS__", film_rows)
                  .replace("__LAYERS__", layer_rows)
                  .replace("__LIMITS__", limit_rows))

with open(TRANSIENT_REPORT, "w", encoding="utf-8") as f:
    f.write(transient_html)
print(f"Saved {TRANSIENT_REPORT}")

# =============================================================================
# INTERACTIVE TRANSIENT HTML
# =============================================================================
js_times = ",".join(f"{v:.12g}" for v in Times)
js_x = ",".join(f"{v:.12g}" for v in x_mm)
js_profiles = ",".join("[" + ",".join(f"{v:.12g}" for v in row) + "]" for row in Profiles)

dashboard = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Transient animation</title>
<style>
body{margin:0;background:#f3f5f7;font-family:Arial,Helvetica,sans-serif;color:#14202b}
main{width:1440px;margin:20px auto;padding:26px;background:white;box-shadow:0 1px 7px #aeb7c0}
h1{margin:0 0 6px;font-size:25px}
canvas{display:block;border:1px solid #65717d;background:white;cursor:crosshair}
.controls{display:flex;align-items:center;gap:12px;margin:13px 0}
button{min-width:76px;padding:7px 12px;cursor:pointer}
input{width:760px}
#timevalue{min-width:190px;font-weight:bold}
#readout{padding:10px;background:#f3f7fb;border-left:4px solid #1f4e79;font-family:Consolas,monospace}
</style>
</head>
<body>
<main>
<h1>Transient temperature profile animation</h1>
<p>__MAT__, total thickness __L__ mm</p>
<canvas id="graph" width="1400" height="600"></canvas>
<div class="controls">
    <button id="play">Play</button>
    <button id="reset">Reset</button>
    <input id="slider" type="range" min="0" max="__MAXINDEX__" value="0">
    <span id="timevalue"></span>
</div>
<div id="readout">Move the pointer over the graph to inspect wall position and temperature.</div>

<script>
const times = [__TIMES__];
const xdata = [__X__];
const profiles = [__PROFILES__];

const L = __L__;
const hotGas = __HOTGAS__;
const coldGas = __COLDGAS__;
const layers = __LAYERS__;

/* Automatic y axis: highest value in profiles, gas temperatures and layer limits */
let dataMax = Math.max(hotGas, coldGas);
for (let j = 0; j < layers.length; j++) {
    dataMax = Math.max(dataMax, layers[j].max_temp_c);
}
for (let r = 0; r < profiles.length; r++) {
    for (let i = 0; i < profiles[r].length; i++) {
        if (profiles[r][i] > dataMax) { dataMax = profiles[r][i]; }
    }
}

const steps = [50, 100, 200, 250, 500, 1000, 2000];
let yStep = steps[steps.length - 1];
for (let s = 0; s < steps.length; s++) {
    if (dataMax * 1.05 / steps[s] <= 10) { yStep = steps[s]; break; }
}
const yMax = Math.ceil(dataMax * 1.05 / yStep) * yStep;

const canvas = document.getElementById("graph");
const g = canvas.getContext("2d");
const slider = document.getElementById("slider");
const timeValue = document.getElementById("timevalue");
const readout = document.getElementById("readout");
const playButton = document.getElementById("play");

let index = 0;
let timer = null;

/* Large left and right margins keep all labels outside the plot area. */
const p = {l: 180, r: 1040, t: 100, b: 515};

function X(x) { return p.l + x / L * (p.r - p.l); }
function Y(T) { return p.b - T / yMax * (p.b - p.t); }

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

    text("Temperature profile at selected transient time", p.l, 35, 21);
    text("Temperature, °C", p.l, 65, 15);

    for (let T = 0; T <= yMax + 1e-9; T += yStep) {
        const y = Y(T);
        line(p.l, y, p.r, y, "#d6dce4", 1, false);
        text(String(T), p.l - 14, y + 5, 12, "right");
    }

    for (let x = 0; x <= L + 1e-9; x += L / 4) {
        const px = X(x);
        line(px, p.t, px, p.b, "#d6dce4", 1, false);
        text(x.toFixed(1), px, p.b + 30, 12, "center");
    }

    g.strokeStyle = "#14202b";
    g.lineWidth = 1.5;
    g.strokeRect(p.l, p.t, p.r - p.l, p.b - p.t);

    for (let j = 0; j < layers.length; j++) {
        const a = X(layers[j].start_mm);
        const b = X(layers[j].end_mm);
        const yl = Y(layers[j].max_temp_c);
        line(a, yl, b, yl, "#16803c", 3, false);
        text(layers[j].name + " max use " + layers[j].max_temp_c.toFixed(0) + " °C",
             (a + b) / 2, yl - 8, 12, "center");
        text(layers[j].name, (a + b) / 2, p.t - 10, 13, "center");
        if (j > 0) {
            line(a, p.t, a, p.b, "#8a5a00", 1.5, true);
        }
    }

    const row = profiles[index];
    const hotSurface = row[0];
    const coldSurface = row[row.length - 1];

    line(p.l - 65, Y(hotGas), p.l, Y(hotSurface), "#65717d", 2, true);
    line(p.r, Y(coldSurface), p.r + 65, Y(coldGas), "#65717d", 2, true);

    g.strokeStyle = "#c9242b";
    g.lineWidth = 3;
    g.beginPath();
    for (let i = 0; i < xdata.length; i++) {
        if (i === 0) { g.moveTo(X(xdata[i]), Y(row[i])); }
        else { g.lineTo(X(xdata[i]), Y(row[i])); }
    }
    g.stroke();

    line(p.l, Y(hotGas), p.l - 42, Y(hotGas), "#65717d", 1.5, false);
    text("Hot gas: " + hotGas.toFixed(0) + " °C", p.l - 50, Y(hotGas) - 10, 13, "right");

    line(p.l, Y(hotSurface), p.l - 42, Y(hotSurface) + 28, "#65717d", 1.5, false);
    text("Hot surface: " + hotSurface.toFixed(1) + " °C", p.l - 50, Y(hotSurface) + 35, 13, "right");

    line(p.r, Y(coldSurface), p.r + 45, Y(coldSurface) - 28, "#65717d", 1.5, false);
    text("Cold surface: " + coldSurface.toFixed(1) + " °C", p.r + 53, Y(coldSurface) - 34, 13);

    line(p.r + 65, Y(coldGas), p.r + 92, Y(coldGas), "#65717d", 1.5, false);
    text("Cold gas: " + coldGas.toFixed(0) + " °C", p.r + 100, Y(coldGas) + 5, 13);

    text("Wall depth from inside to outside, mm", (p.l + p.r) / 2, canvas.height - 24, 15, "center");

    timeValue.textContent = "t = " + times[index].toFixed(1) + " s  (" +
                            (times[index] / 60).toFixed(2) + " min)";
}

function localTemperature(x) {
    const row = profiles[index];
    if (x <= xdata[0]) { return row[0]; }
    if (x >= xdata[xdata.length - 1]) { return row[row.length - 1]; }
    for (let i = 0; i < xdata.length - 1; i++) {
        if (x <= xdata[i + 1]) {
            const ratio = (x - xdata[i]) / (xdata[i + 1] - xdata[i]);
            return row[i] + ratio * (row[i + 1] - row[i]);
        }
    }
    return row[row.length - 1];
}

function layerAt(x) {
    for (let j = 0; j < layers.length; j++) {
        if (x <= layers[j].end_mm + 1e-9) { return layers[j].name; }
    }
    return layers[layers.length - 1].name;
}

canvas.addEventListener("mousemove", function(event) {
    const rect = canvas.getBoundingClientRect();
    const px = (event.clientX - rect.left) * canvas.width / rect.width;
    if (px < p.l || px > p.r) { return; }
    const x = (px - p.l) / (p.r - p.l) * L;
    const T = localTemperature(x);
    readout.textContent = "t = " + times[index].toFixed(1) + " s    x = " +
                          x.toFixed(3) + " mm    T = " + T.toFixed(2) + " °C    layer: " + layerAt(x);
});

canvas.addEventListener("mouseleave", function() {
    readout.textContent = "Move the pointer over the graph to inspect wall position and temperature.";
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

dashboard = (dashboard
             .replace("__MAT__", html.escape(MATERIAL_NAME))
             .replace("__L__", f"{L_MM:.12g}")
             .replace("__MAXINDEX__", str(len(Times) - 1))
             .replace("__TIMES__", js_times)
             .replace("__X__", js_x)
             .replace("__PROFILES__", js_profiles)
             .replace("__HOTGAS__", str(T_GAS_HOT_C))
             .replace("__COLDGAS__", str(T_GAS_COLD_C))
             .replace("__LAYERS__", json.dumps(LAYERS)))

with open(TRANSIENT_DASHBOARD, "w", encoding="utf-8") as f:
    f.write(dashboard)
print(f"Saved {TRANSIENT_DASHBOARD}")
