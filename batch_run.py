"""
batch_run.py
Run the steady and transient wall scripts for many configurations in one go.

How it works
  Every setting in your scripts (T_GAS_HOT_C, LAYERS, INSIDE_FILM, OUTSIDE_FILM,
  DT, PROJECT, TITLE, ...) can be overridden per case. The runner rewrites the
  top level assignments of each script in memory, so your scripts stay untouched.
  Each case gets its own folder inside RESULTS_DIR, and every output file is
  prefixed with the case name.

Wall (material) selection per case
  "wall": "wds_al_250"                          preset from WALLS below
  "wall": [("WDS", 100), ("Al", 50)]            shorthand: (material, thickness mm)
  "wall": [("WDS", 100, 5.0, 950.0)]            optional dx_mm and max_temp_c
  "wall": [{"name": "WDS", "thickness_mm": 80}] dictionary form
  Missing dx_mm and max_temp_c are taken from MATERIAL_LIBRARY.
  "LAYERS" with full dictionaries still works as before (do not combine both).

Usage
  python batch_run.py                       run the CASES list below
  python batch_run.py --config cases.json   run cases from a JSON file
  python batch_run.py --workers 4           run four cases in parallel
  python batch_run.py --only baseline,floor run only the named cases
  python batch_run.py --list                show the cases without running
"""

import argparse
import ast
import contextlib
import copy
import csv
import itertools
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

# =============================================================================
# BATCH SETTINGS
# =============================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Script file names inside the project folder (set them to your actual names)
STEADY_CALC = "steady_calc_2.py"
STEADY_REPORT = "steady_report_2.py"
TRANSIENT_CALC = "transient_calc_2.py"
TRANSIENT_REPORT = "transient_report_2.py"

RESULTS_DIR = "batch_results"       # created inside the project folder
RUN_FOLDER_PER_BATCH = True         # True: batch_results/2026-10-08_0451/..., False: reuse one folder
WORKERS = 1                         # parallel processes, 1 = one case after another

# Settings applied to every case unless the case sets them again
COMMON = {
    # "T_GAS_COLD_C": 30.0,
    "PROJECT": {"customer": "Testo_1234",
                "name": "O.K."},
}

# Output file names of the scripts. The runner adds the case name as prefix.
OUTPUT_NAMES = {
    "STEADY_CSV": "steady_data.csv",
    "STEADY_META": "steady_meta.json",
    "STEADY_REPORT": "steady_report.html",
    "TRANSIENT_CSV": "transient_data.csv",
    "TRANSIENT_META": "transient_meta.json",
    "TRANSIENT_REPORT": "transient_report.html",
    "TRANSIENT_DASHBOARD": "transient_interactive.html",
}

# =============================================================================
# MATERIAL LIBRARY AND WALL PRESETS
# Material names must exist in the heatrapy database folder.
# max_temp_c: max use temperature, dx_mm: default mesh size for that material.
# =============================================================================
MATERIAL_LIBRARY = {
    "WDS": {"max_temp_c": 950.0, "dx_mm": 10.0},
    "Al":  {"max_temp_c": 600.0, "dx_mm": 5.0},
    # "MyBrick": {"max_temp_c": 1400.0, "dx_mm": 10.0},
}

# Named wall builds, hot face first: (material, thickness in mm)
WALLS = {
    "wds_150": [("WDS", 150)],
    "wds_al_250": [("WDS", 150), ("Al", 50), ("Al", 50)],
    "wds_100": [("WDS", 100)],
}
DEFAULT_DX_MM = 10.0     # used for materials that are not in the library


# =============================================================================
# CASE DEFINITIONS
# A case is a dictionary:
#   "name"  unique case name (used for folder and file names)
#   "run"   "steady", "transient" or "both" (default "both")
#   "wall"  wall preset name or layer list (see top of this file)
#   every other key is a script setting to override, for example
#   "T_GAS_HOT_C": 1000.0
# Dictionary settings (PROJECT, INSIDE_FILM, OUTSIDE_FILM) are merged into the
# script value, so {"OUTSIDE_FILM": {"orientation": "F"}} changes only that key.
# Lists (LAYERS) are replaced as a whole.
# =============================================================================
def merge_value(old, new):
    if isinstance(old, dict) and isinstance(new, dict):
        merged = dict(old)
        for key, value in new.items():
            merged[key] = merge_value(old.get(key), value)
        return merged
    return copy.deepcopy(new)


def sweep(prefix, axes, run="both", common=None):
    """
    Build the cartesian product of several axes.
    axes = {"axis name": {"label": {setting overrides}, "label2": {...}}, ...}
    Case names look like prefix_label1_label2.
    """
    names = list(axes)
    out = []
    for combo in itertools.product(*[list(axes[n].items()) for n in names]):
        case = {"name": "_".join([prefix] + [label for label, _ in combo]), "run": run}
        case.update(copy.deepcopy(common or {}))
        for _, overrides in combo:
            for key, value in overrides.items():
                case[key] = merge_value(case.get(key), value)
        out.append(case)
    return out


CASES = [

    # 1
    {"name": "Test_123", "run": "steady",
     "T_GAS_HOT_C": 900.0, "T_GAS_COLD_C": 30.0,
     "wall": "wds_al_250",
     "INSIDE_FILM": {"mode": "manual", "h": 150.0},
     "OUTSIDE_FILM": {"mode": "astm_c680", "orientation": "V"}},

    # 2 Layer list in shorthand: (material, thickness mm)
    # {"name": "two_layers", "run": "steady",
    #  "wall": [("WDS", 100), ("Al", 50)],
    #  "OUTSIDE_FILM": {"mode": "astm_c680", "orientation": "V"}},

    # 3 Sweep over walls and hot gas temperatures
    # *sweep("walls",
    #        {"wall": {"w150": {"wall": "wds_150"},
    #                  "w100": {"wall": "wds_100"},
    #                  "w250": {"wall": "wds_al_250"}},
    #         "gas": {"800C": {"T_GAS_HOT_C": 800.0},
    #                 "1000C": {"T_GAS_HOT_C": 1000.0}}},
    #        run="steady"),
]


# =============================================================================
# ENGINE (normally no changes needed below this line)
# =============================================================================
def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.]+", "_", str(text)).strip("_")


def build_case(raw):
    """Combine COMMON with a case. Returns (merged case, keys the case set itself)."""
    merged = copy.deepcopy(COMMON)
    for key, value in raw.items():
        merged[key] = merge_value(merged.get(key), value)
    strict = {k for k in raw if k not in ("name", "run", "wall")}
    return merged, strict


def resolve_wall(spec):
    """Turn a wall preset name or a shorthand list into the LAYERS format."""
    if isinstance(spec, str):
        if spec not in WALLS:
            raise KeyError(f"Unknown wall preset '{spec}'. Known presets: {sorted(WALLS)}")
        spec = WALLS[spec]
    layers = []
    for item in spec:
        if isinstance(item, dict):
            d = dict(item)
        else:
            item = list(item)
            if len(item) < 2:
                raise ValueError(f"Layer needs at least (material, thickness): {item}")
            d = {"name": item[0], "thickness_mm": item[1]}
            if len(item) > 2 and item[2] is not None:
                d["dx_mm"] = item[2]
            if len(item) > 3 and item[3] is not None:
                d["max_temp_c"] = item[3]
        if "name" not in d or "thickness_mm" not in d:
            raise ValueError(f"Layer needs 'name' and 'thickness_mm': {d}")
        lib = MATERIAL_LIBRARY.get(d["name"], {})
        max_temp = d.get("max_temp_c", lib.get("max_temp_c"))
        if max_temp is None:
            raise ValueError(f"No max_temp_c for material '{d['name']}'. "
                             f"Add it to MATERIAL_LIBRARY or give it in the layer.")
        thickness = float(d["thickness_mm"])
        dx = d.get("dx_mm", lib.get("dx_mm", min(DEFAULT_DX_MM, thickness / 5.0)))
        layers.append({"name": d["name"], "thickness_mm": thickness,
                       "dx_mm": float(dx), "max_temp_c": float(max_temp)})
    if not layers:
        raise ValueError("Wall has no layers.")
    return layers


def wall_text(layers):
    return " | ".join(f"{l['name']} {l['thickness_mm']:g}" for l in layers)


def override_source(source, overrides):
    """Replace top level assignments NAME = ... by the override values."""
    tree = ast.parse(source)
    applied = set()
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in overrides):
            # Values read back from the calculation results (meta[...]) are not settings
            if any(isinstance(n, ast.Name) and n.id == "meta" for n in ast.walk(node.value)):
                continue
            name = node.targets[0].id
            value = overrides[name]
            if isinstance(value, dict):
                try:
                    value = merge_value(ast.literal_eval(node.value), value)
                except (ValueError, SyntaxError):
                    pass
            node.value = ast.parse(repr(value), mode="eval").body
            applied.add(name)
    ast.fix_missing_locations(tree)
    return tree, applied


def run_script(script_name, overrides, log):
    path = os.path.join(PROJECT_DIR, script_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Script not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    tree, applied = override_source(source, overrides)
    print(f"--- {script_name}", file=log)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        exec(compile(tree, path, "exec"), {"__name__": "__main__", "__file__": path})
    return applied


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(case_dir, meta_files):
    info = {}
    steady_meta = os.path.join(case_dir, meta_files["steady"])
    transient_meta = os.path.join(case_dir, meta_files["transient_meta"])
    if os.path.isfile(steady_meta):
        m = read_json(steady_meta)
        info["wall"] = wall_text(m["layers"])
        info.update({
            "steady_T_cold_C": round(m["t_surface_cold_c"], 2),
            "steady_T_hot_C": round(m["t_surface_hot_c"], 2),
            "steady_q_W_m2": round(m["q_hot"], 2),
            "steady_h_cold": round(m["h_cold"], 3),
            "steady_h_hot": round(m["h_hot"], 3),
            "steady_converged": m["converged"],
        })
    if os.path.isfile(transient_meta) and "wall" not in info:
        m = read_json(transient_meta)
        info["wall"] = " | ".join(f"{l['name']} {l['end_mm'] - l['start_mm']:g}" for l in m["layers"])
    trans_csv = os.path.join(case_dir, meta_files["transient_csv"])
    if os.path.isfile(trans_csv):
        with open(trans_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [[float(v) for v in r] for r in reader]
        first_T = next(i for i, h in enumerate(header) if h.startswith("T_x_"))
        cold_col = len(header) - 1
        info.update({
            "transient_T_hot_end_C": round(rows[-1][first_T], 2),
            "transient_T_cold_end_C": round(rows[-1][cold_col], 2),
            "transient_T_cold_peak_C": round(max(r[cold_col] for r in rows), 2),
            "transient_q_cold_end_W_m2": round(rows[-1][header.index("q_cold_W_m2")], 2),
        })
    return info


def run_case(raw_case, batch_dir, project_dir):
    """Run one case. Top level function so it also works with process pools."""
    start = time.time()
    case, strict_keys = build_case(raw_case)
    name = safe_name(case["name"])
    case_dir = os.path.join(batch_dir, name)
    os.makedirs(case_dir, exist_ok=True)
    result = {"name": name, "status": "OK", "message": "", "folder": case_dir}

    run = case.get("run", "both")
    if run not in ("steady", "transient", "both"):
        result.update(status="FAILED", message=f"Unknown run option: {run}", seconds=0.0, run=run)
        return result

    overrides = {k: v for k, v in case.items() if k not in ("name", "run", "wall")}
    resolved_layers = None
    try:
        if "wall" in case:
            if "LAYERS" in case:
                raise ValueError("Use either 'wall' or 'LAYERS' in a case, not both.")
            resolved_layers = resolve_wall(case["wall"])
            overrides["LAYERS"] = resolved_layers
            strict_keys = strict_keys | {"LAYERS"}
            result["wall"] = wall_text(resolved_layers)
    except Exception as exc:
        result.update(status="FAILED", message=f"{type(exc).__name__}: {exc}", seconds=0.0, run=run)
        return result

    for key, suffix in OUTPUT_NAMES.items():
        overrides[key] = f"{name}_{suffix}"

    config = dict(case)
    if resolved_layers is not None:
        config["resolved_layers"] = resolved_layers
    with open(os.path.join(case_dir, f"{name}_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    sequence = []
    if run in ("steady", "both"):
        sequence += [STEADY_CALC, STEADY_REPORT]
    if run in ("transient", "both"):
        sequence += [TRANSIENT_CALC, TRANSIENT_REPORT]

    old_cwd = os.getcwd()
    old_path = list(sys.path)
    sys.path.insert(0, project_dir)
    # Settings from COMMON are optional (a transient only case ignores steady only keys)
    user_keys = {k for k in overrides if k in strict_keys and k not in OUTPUT_NAMES}

    # Check the setting names before anything runs, so typos fail immediately
    try:
        known = set()
        for script in sequence:
            with open(os.path.join(project_dir, script), "r", encoding="utf-8") as f:
                known |= override_source(f.read(), overrides)[1]
        unknown = user_keys - known
        if unknown:
            raise KeyError(f"Settings not found in the scripts: {sorted(unknown)}")
    except Exception as exc:
        sys.path[:] = old_path
        result.update(status="FAILED", message=f"{type(exc).__name__}: {exc}", seconds=0.0, run=run)
        return result

    log_path = os.path.join(case_dir, f"{name}_log.txt")
    try:
        os.chdir(case_dir)
        with open(log_path, "w", encoding="utf-8") as log:
            try:
                for script in sequence:
                    run_script(script, overrides, log)
            except Exception:
                traceback.print_exc(file=log)
                result.update(status="FAILED", message=traceback.format_exc().strip().splitlines()[-1])
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path

    files = {"steady": f"{name}_{OUTPUT_NAMES['STEADY_META']}",
             "transient_meta": f"{name}_{OUTPUT_NAMES['TRANSIENT_META']}",
             "transient_csv": f"{name}_{OUTPUT_NAMES['TRANSIENT_CSV']}"}
    if result["status"] == "OK":
        try:
            result.update(summarize(case_dir, files))
        except Exception as exc:
            result.update(status="FAILED", message=f"Summary failed: {exc}")
    result["seconds"] = round(time.time() - start, 1)
    result["run"] = run
    return result


def write_summary(results, batch_dir):
    keys = []
    for r in results:
        for k in r:
            if k not in keys:
                keys.append(k)
    csv_path = os.path.join(batch_dir, "batch_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    rows = ""
    for r in results:
        links = []
        for key, label in (("steady_report", "steady report"), ("transient_report", "transient report"),
                           ("transient_interactive", "animation"), ("log", "log")):
            suffix = {"steady_report": OUTPUT_NAMES["STEADY_REPORT"],
                      "transient_report": OUTPUT_NAMES["TRANSIENT_REPORT"],
                      "transient_interactive": OUTPUT_NAMES["TRANSIENT_DASHBOARD"],
                      "log": "log.txt"}[key]
            rel = f"{r['name']}/{r['name']}_{suffix}"
            if os.path.isfile(os.path.join(batch_dir, rel)):
                links.append(f'<a href="{rel}">{label}</a>')
        cls = "ok" if r["status"] == "OK" else "bad"
        rows += (f"<tr><td>{r['name']}</td><td class='{cls}'>{r['status']}</td>"
                 f"<td>{r.get('wall', '')}</td>"
                 f"<td>{r.get('steady_T_cold_C', '')}</td><td>{r.get('steady_q_W_m2', '')}</td>"
                 f"<td>{r.get('steady_h_cold', '')}</td><td>{r.get('transient_T_cold_end_C', '')}</td>"
                 f"<td>{r.get('seconds', '')}</td><td>{' | '.join(links)}</td>"
                 f"<td>{r['message']}</td></tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Batch summary</title>
<style>body{{font-family:Arial;margin:30px;color:#14202b}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #7f8c8d;padding:6px 9px;font-size:13px}}th{{background:#dce6f1;text-align:left}}
.ok{{color:#16803c;font-weight:bold}}.bad{{color:#b00020;font-weight:bold}}</style></head><body>
<h1>Batch summary</h1><p>{datetime.now():%Y-%m-%d %H:%M}, {len(results)} cases</p>
<table><tr><th>Case</th><th>Status</th><th>Wall (material mm)</th><th>Steady cold face, °C</th><th>Steady q, W/m²</th>
<th>Steady h cold, W/m²K</th><th>Transient cold face end, °C</th><th>Seconds</th><th>Files</th><th>Message</th></tr>
{rows}</table></body></html>"""
    with open(os.path.join(batch_dir, "batch_summary.html"), "w", encoding="utf-8") as f:
        f.write(page)
    return csv_path


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if isinstance(cases, dict):
        cases = cases.get("cases", [])
    return cases


def main():
    parser = argparse.ArgumentParser(description="Batch runner for the wall scripts")
    parser.add_argument("--config", help="JSON file with a list of cases")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--only", help="comma separated case names to run")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    args = parser.parse_args()

    cases = load_cases(args.config) if args.config else CASES
    if args.only:
        wanted = {safe_name(n) for n in args.only.split(",")}
        cases = [c for c in cases if safe_name(c["name"]) in wanted]

    names = [safe_name(c["name"]) for c in cases]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        sys.exit(f"Duplicate case names: {sorted(duplicates)}")

    if args.list:
        for c in cases:
            merged, _ = build_case(c)
            try:
                text = wall_text(resolve_wall(merged["wall"])) if "wall" in merged else "(from script)"
            except Exception as exc:
                text = f"ERROR: {exc}"
            print(merged["name"], "|", merged.get("run", "both"), "|", text)
        return

    root = os.path.join(PROJECT_DIR, RESULTS_DIR)
    batch_dir = os.path.join(root, datetime.now().strftime("%Y-%m-%d_%H%M%S")) if RUN_FOLDER_PER_BATCH else root
    os.makedirs(batch_dir, exist_ok=True)
    print(f"Results folder: {batch_dir}")
    print(f"{len(cases)} cases, {args.workers} worker(s)")

    results = []
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, c, batch_dir, PROJECT_DIR) for c in cases]
            for fut in futures:
                r = fut.result()
                results.append(r)
                print(f"[{len(results)}/{len(cases)}] {r['name']}: {r['status']} ({r.get('seconds', '')} s) {r['message']}")
    else:
        for i, c in enumerate(cases, 1):
            r = run_case(c, batch_dir, PROJECT_DIR)
            results.append(r)
            print(f"[{i}/{len(cases)}] {r['name']}: {r['status']} ({r.get('seconds', '')} s) {r['message']}")

    write_summary(results, batch_dir)
    failed = [r["name"] for r in results if r["status"] != "OK"]
    print(f"Finished in {time.time() - t0:.1f} s. Failed: {failed if failed else 'none'}")
    print(f"Summary: {os.path.join(batch_dir, 'batch_summary.csv')}")


if __name__ == "__main__":
    main()