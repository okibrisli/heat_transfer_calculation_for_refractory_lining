"""
Steady state wall report (one HTML file).
Reads wds_steady_data.csv and wds_steady_meta.json.
"""

import csv
import html
import json

# =============================================================================
# REPORT SETTINGS
# =============================================================================
STEADY_CSV = "wds_steady_data.csv"
STEADY_META = "wds_steady_meta.json"
STEADY_REPORT = "wds_steady_report.html"
TITLE = "Steady state heat transfer report"

# =============================================================================
# LOAD DATA
# =============================================================================
with open(STEADY_META, "r", encoding="utf-8") as f:
    meta = json.load(f)

x_mm, T_C = [], []
with open(STEADY_CSV, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        x_mm.append(float(row["x_mm"]))
        T_C.append(float(row["T_C"]))

project = meta.get("project", {})
layers = meta["layers"]
L_MM = meta["l_mm"]
Tg_hot, Tg_cold = meta["t_gas_hot_c"], meta["t_gas_cold_c"]
Ts_hot, Ts_cold = meta["t_surface_hot_c"], meta["t_surface_cold_c"]
esc = html.escape

# =============================================================================
# PROFILE DRAWING (same look as the transient animation)
# =============================================================================
CW, CH = 1400, 640
p = {"l": 180, "r": 1040, "t": 125, "b": 540}

data_max = max(Tg_hot, Tg_cold, max(T_C), max(l["max_temp_c"] for l in layers))
y_step = 2000
for st in (50, 100, 200, 250, 500, 1000, 2000):
    if data_max * 1.05 / st <= 10:
        y_step = st
        break
y_max = -(-data_max * 1.05 // y_step) * y_step


def X(x):
    return p["l"] + x / L_MM * (p["r"] - p["l"])


def Y(T):
    return p["b"] - T / y_max * (p["b"] - p["t"])


out = []


def line(x1, y1, x2, y2, color, width, dashed=False):
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
               f'stroke="{color}" stroke-width="{width}"{dash}/>')


def text(value, x, y, size, anchor="start", weight="normal", halo=False):
    extra = ' style="paint-order:stroke;stroke:white;stroke-width:4px;stroke-linejoin:round"' if halo else ""
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
               f'font-weight="{weight}" fill="#14202b"{extra}>{esc(value)}</text>')


out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CW} {CH}" '
           f'width="100%" font-family="Arial, Helvetica, sans-serif">')
out.append(f'<rect x="0" y="0" width="{CW}" height="{CH}" fill="white"/>')

text("Temperature profile through the wall at steady state", p["l"], 35, 21)
text("Temperature, °C", p["l"], 70, 15)

T_tick = 0.0
while T_tick <= y_max + 1e-9:
    y = Y(T_tick)
    line(p["l"], y, p["r"], y, "#d6dce4", 1)
    text(f"{T_tick:g}", p["l"] - 14, y + 5, 12, "end")
    T_tick += y_step

for k in range(5):
    xv = L_MM * k / 4.0
    px = X(xv)
    line(px, p["t"], px, p["b"], "#d6dce4", 1)
    text(f"{xv:.1f}", px, p["b"] + 30, 12, "middle")

out.append(f'<rect x="{p["l"]}" y="{p["t"]}" width="{p["r"] - p["l"]}" height="{p["b"] - p["t"]}" '
           f'fill="none" stroke="#14202b" stroke-width="1.5"/>')

for j, lay in enumerate(layers):
    a, b = X(lay["start_mm"]), X(lay["end_mm"])
    narrow = (b - a) < 90
    fs_name, fs_lim = (11, 10) if narrow else (13, 12)
    text(lay["name"], 0.5 * (a + b), p["t"] - 26, fs_name, "middle", "bold")
    text(f'{lay["thickness_mm"]:g} mm', 0.5 * (a + b), p["t"] - 10, fs_name - 1, "middle")
    yl = Y(lay["max_temp_c"])
    line(a, yl, b, yl, "#16803c", 3)
    text(f'{lay["name"]} max use {lay["max_temp_c"]:.0f} °C', 0.5 * (a + b), yl - 8, fs_lim, "middle")
    if j > 0:
        line(a, p["t"], a, p["b"], "#8a5a00", 1.5, True)

line(p["l"] - 65, Y(Tg_hot), p["l"], Y(Ts_hot), "#65717d", 2, True)
line(p["r"], Y(Ts_cold), p["r"] + 65, Y(Tg_cold), "#65717d", 2, True)

pts = " ".join(f"{X(x):.2f},{Y(T):.2f}" for x, T in zip(x_mm, T_C))
out.append(f'<polyline points="{pts}" fill="none" stroke="#c9242b" stroke-width="3" stroke-linejoin="round"/>')

for lay in layers[1:]:
    xi, yi = X(lay["start_mm"]), Y(lay["t_hot_side_c"])
    out.append(f'<circle cx="{xi:.1f}" cy="{yi:.1f}" r="4.5" fill="#c9242b" stroke="white" stroke-width="1.5"/>')
    text(f'{lay["t_hot_side_c"]:.1f} °C', xi + 10, yi - 10, 13, "start", "bold", True)

line(p["l"], Y(Tg_hot), p["l"] - 42, Y(Tg_hot), "#65717d", 1.5)
text(f"Hot gas: {Tg_hot:.0f} °C", p["l"] - 50, Y(Tg_hot) - 10, 13, "end")
line(p["l"], Y(Ts_hot), p["l"] - 42, Y(Ts_hot) + 28, "#65717d", 1.5)
text(f"Hot surface: {Ts_hot:.1f} °C", p["l"] - 50, Y(Ts_hot) + 35, 13, "end")
line(p["r"], Y(Ts_cold), p["r"] + 45, Y(Ts_cold) - 28, "#65717d", 1.5)
text(f"Cold surface: {Ts_cold:.1f} °C", p["r"] + 53, Y(Ts_cold) - 34, 13, "start")
line(p["r"] + 65, Y(Tg_cold), p["r"] + 92, Y(Tg_cold), "#65717d", 1.5)
text(f"Cold gas: {Tg_cold:.0f} °C", p["r"] + 100, Y(Tg_cold) + 5, 13, "start")

text(f"Wall depth from inside to outside, mm (total {L_MM:g} mm)",
     0.5 * (p["l"] + p["r"]), CH - 22, 15, "middle")
out.append("</svg>")
svg = "\n".join(out)

# =============================================================================
# TABLES
# =============================================================================
layer_rows = ""
for lay in layers:
    over = lay["t_hot_side_c"] > lay["max_temp_c"]
    cls = ' class="n bad"' if over else ' class="n"'
    status = '<span class="bad">Exceeded</span>' if over else '<span class="ok">OK</span>'
    layer_rows += (
        f"<tr><td>{esc(lay['name'])}</td><td class='n'>{lay['thickness_mm']:g}</td>"
        f"<td class='n'>{lay['k_eff']:.3f}</td><td class='n'>{lay['t_mean_c']:.0f}</td>"
        f"<td{cls}>{lay['t_hot_side_c']:.1f}</td>"
        f"<td class='n'>{lay['max_temp_c']:.0f}</td><td>{status}</td></tr>"
    )
layer_rows += (
    f"<tr class='tot'><td>Total</td><td class='n'>{L_MM:g}</td><td></td><td></td>"
    f"<td class='n'>{Ts_cold:.1f}</td><td></td><td></td></tr>"
)

css = """
body{margin:0;background:#f3f5f7;font-family:Arial,Helvetica,sans-serif;color:#14202b;font-size:13px}
main{max-width:1100px;margin:20px auto;padding:28px;background:white;box-shadow:0 1px 7px #aeb7c0}
h1{font-size:19px;margin:0 0 12px}
h2{font-size:15px;color:#14202b;border-bottom:2px solid #1f4e79;padding-bottom:4px;margin:26px 0 8px}
table{border-collapse:collapse;width:100%}
.head td{border:1px solid #14202b;padding:2px 8px;font-size:13px}
.head td:first-child{width:200px}
.data th{background:#dce6f1;text-align:left;border:1px solid #7f8c8d;padding:6px 9px}
.data td{border:1px solid #c3cad2;padding:6px 9px}
.data tr:nth-child(even) td{background:#f8fafc}
.data td.n,.data th.n{text-align:right}
.data tr.tot td{font-weight:bold;background:#eef3f8;border-top:2px solid #1f4e79}
.bad{color:#b00020;font-weight:bold}
.ok{color:#16803c;font-weight:bold}
.note{margin-top:16px;padding:10px;background:#f3f7fb;border-left:4px solid #1f4e79;font-size:12px}
svg{display:block;border:1px solid #65717d}
@media print{body{background:white}main{box-shadow:none;margin:0;max-width:none}}
"""

conv = "Converged" if meta["converged"] else "Not converged"
doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{esc(TITLE)}</title><style>{css}</style></head>
<body><main>
<h1>{esc(TITLE)}</h1>
<table class="head">
<tr><td>Customer</td><td>{esc(str(project.get('customer', '')))}</td></tr>
<tr><td>Plant</td><td>{esc(str(project.get('plant', '')))}</td></tr>
<tr><td>Detail</td><td>{esc(str(project.get('detail', '')))}</td></tr>
<tr><td>Name</td><td>{esc(str(project.get('name', '')))}</td></tr>
</table>

<h2>Boundary conditions</h2>
<table class="data">
<tr><th>Parameter</th><th class="n">Inside</th><th class="n">Outside</th><th>Unit</th></tr>
<tr><td>Ambient temperature (input)</td><td class="n">{Tg_hot:.0f}</td><td class="n">{Tg_cold:.0f}</td><td>°C</td></tr>
<tr><td>Surface temperature (calculated)</td><td class="n">{Ts_hot:.1f}</td><td class="n">{Ts_cold:.1f}</td><td>°C</td></tr>
<tr><td>Heat transition coefficient</td><td class="n">{meta['h_hot']:.2f}</td><td class="n">{meta['h_cold']:.3f}</td><td>W/m²K</td></tr>
<tr><td>Calculation formula</td><td class="n"></td><td class="n">Constant h (input)</td><td></td></tr>
<tr><td>Heat loss per m²</td><td class="n">{meta['q_hot']:.1f}</td><td class="n">{meta['q_cold']:.1f}</td><td>W/m²</td></tr>
</table>

<h2>Wall layers from inside to outside</h2>
<table class="data">
<tr><th>Material</th><th class="n">mm</th><th class="n">W/m·K</th><th class="n">Mean, °C</th><th class="n">Border, °C</th><th class="n">Max use, °C</th><th>Status</th></tr>
{layer_rows}
</table>

<h2>Temperature profile</h2>
{svg}

<div class="note">Solver: finite volume with Picard iteration and temperature dependent conductivity. {conv} after {meta['iterations']} iterations. Flux imbalance between inside and outside film: {meta['imbalance_pct']:.2e} %. Layer conductivity is the equivalent value from flux, thickness and temperature drop. Border is the temperature at the hot side of each layer, the total row shows the outside surface.</div>
</main></body></html>"""

with open(STEADY_REPORT, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"Saved {STEADY_REPORT}")