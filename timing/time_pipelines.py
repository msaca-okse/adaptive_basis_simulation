"""
Wall-clock time of the reconstruction pipelines for the simulated samples, all on one machine, and the
accuracy of the maps they produce.

    python timing/time_pipelines.py --workdir /path/to/scratch/copy          every stage, in a copy
    python timing/time_pipelines.py --inplace --stages "TT reconstruction" \\
        --reuse timing/pipeline_timings_2026-10-02.json --metrics             only the TT, in place

The notebooks of every stage are executed in order, with the parameters they contain, and the wall
time of every notebook is recorded:

    pbp:          peak segmentation, pbp indexing, refinement (the pbp map; it also builds the basis)
    TT-Adaptive:  peak segmentation, pbp indexing, refinement, integration, TT reconstruction
    TT-Uniform:   integration, TT reconstruction

The integration is shared by both TT pipelines and counted in both totals. Notebooks that only choose
parameters or inspect results (segment_peaks/01_find_parameters, 03_inspect, indexing/01_inspect_peaks)
are not timed.

By default the repository's code and notebooks are copied to `workdir` (with `data/` linked to this
repository's data), so the timed runs write their outputs there and leave this repository's results
untouched. With --stages only those stages run; the times of the others are taken from --reuse (an
earlier result file, which must come from the same machine). Their inputs then have to exist, so such
a run is made --inplace, in this repository, and updates its outputs (e.g. the reconstructions).
--metrics then runs visualization/<sample>/figure_panels.ipynb (not timed) and reports the median
misorientation to the ground truth and the grain-boundary deviation of every map.

Results go to timing/pipeline_timings.{json,md} in this repository; the json is written after every
notebook, so a partial run keeps what was measured.
"""
import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime

import nbformat
from nbclient import NotebookClient

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLES = ("domains_mosaicity_1p0deg", "domains_mosaicity_10p0deg")
STAGES = [  # (stage, pipelines that need it, notebooks relative to the repository; {s} = sample)
    ("Peak segmentation", ("pbp", "adaptive"), ["adaptive_basis/{s}/segment_peaks/02_segment.ipynb"]),
    ("Peak indexing", ("pbp", "adaptive"), ["adaptive_basis/{s}/indexing/02_pbp_index.ipynb",
                                            "adaptive_basis/{s}/indexing/03_refine.ipynb"]),
    ("Integration", ("adaptive", "uniform"), ["integration/{s}/01_inspect_integrate.ipynb"]),
    ("TT reconstruction", ("adaptive",), ["texture_tomography/{s}/textomo_adaptive.ipynb"]),
    ("TT reconstruction", ("uniform",), ["texture_tomography/{s}/textomo_uniform.ipynb"]),
]
# table columns: (heading, notebook) -- the indexing stage is shown as its two notebooks
COLUMNS = [
    ("Peak segmentation", "adaptive_basis/{s}/segment_peaks/02_segment.ipynb"),
    ("PBP indexing", "adaptive_basis/{s}/indexing/02_pbp_index.ipynb"),
    ("Refinement + basis", "adaptive_basis/{s}/indexing/03_refine.ipynb"),
    ("Integration", "integration/{s}/01_inspect_integrate.ipynb"),
    ("TT reconstruction", "texture_tomography/{s}/textomo_{p}.ipynb"),
]
PIPELINES = [("pbp", "pbp", (0, 1, 2)), ("adaptive", "TT-Adaptive", (0, 1, 2, 3, 4)), ("uniform", "TT-Uniform", (3, 4))]
METRICS_NB = "visualization/{s}/figure_panels.ipynb"
COPY = ["simtools", "integration", "adaptive_basis", "texture_tomography", "requirements.txt"]


def make_copy(workdir):
    if os.path.exists(workdir):
        raise FileExistsError(f"{workdir} exists; remove it or choose another --workdir")
    os.makedirs(workdir)
    for item in COPY:
        src = os.path.join(REPO, item)
        dst = os.path.join(workdir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
        else:
            shutil.copy2(src, dst)
    for sample in SAMPLES:  # the basis is an output of the pipeline: start without it
        basis = os.path.join(workdir, "adaptive_basis", sample, "basis.npy")
        if os.path.exists(basis):
            os.remove(basis)
    os.symlink(os.path.join(REPO, "data"), os.path.join(workdir, "data"))


def run_notebook(path):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(path)}})
    t0 = time.perf_counter()
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)  # executed notebook, with outputs
    return time.perf_counter() - t0


def machine():
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    return {"host": platform.node(), "gpu": gpu[0] if gpu else "?", "cpu": cpu,
            "cores": len(os.sched_getaffinity(0)), "job": os.environ.get("LSB_JOBID")}


def diffractom_version():
    if os.environ.get("DIFFRACTOM_VERSION"):  # e.g. set by a job that runs an exported tree
        return os.environ["DIFFRACTOM_VERSION"]
    try:
        import diffractom
        d = os.path.dirname(os.path.dirname(os.path.dirname(diffractom.__file__)))
        sha = subprocess.run(["git", "-C", d, "log", "-1", "--format=%h %cs %s"], capture_output=True,
                             text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", d, "status", "--short", "src"], capture_output=True, text=True).stdout
        return sha + (" (with uncommitted changes)" if dirty.strip() else "")
    except Exception as e:  # noqa: BLE001
        return f"? ({e})"


def fmt(seconds):
    m = seconds / 60
    return f"{m:.1f} min" if m < 60 else f"{m / 60:.2f} h ({m:.0f} min)"


def label(sample):
    return sample.split("_")[-1].replace("p", ".").replace("deg", " deg")


def notebook_times(results):
    """{notebook: (seconds, date measured, host)} from a result file (any version of this script)."""
    t = {}
    for r in results["runs"]:
        for nb, s in r["notebooks"].items():
            t[nb] = (s, r.get("measured", results["date"]), r.get("host", results["machine"]["host"]))
    return t


def read_metrics(sample, root):
    out = os.path.join(root, "processed", "visualization", sample)
    m = {}
    with open(os.path.join(out, "misorientation_statistics.csv")) as f:
        for r in csv.DictReader(f):
            m.setdefault(r["map"], {})["median misorientation (deg)"] = float(r["median (deg)"])
    with open(os.path.join(out, "boundary_statistics.csv")) as f:
        for r in csv.DictReader(f):
            if r["boundary"] == "grain":
                m.setdefault(r["map"], {})["grain boundary deviation, mean (um)"] = float(r["mean deviation (um)"])
                m[r["map"]]["grain boundary deviation, mean (display pixels)"] = float(r["mean deviation (display pixels)"])
    return m


def write_table(results, path):
    mach = results["machine"]
    times = notebook_times(results)
    reused = sorted({f"{d} on {h}" for _, d, h in times.values() if d != results["date"]})
    lines = [
        "# Pipeline timings (simulated samples)",
        "",
        f"Measured {results['date']} with `timing/time_pipelines.py` on {mach['gpu']}, {mach['cores']} cores of "
        f"{mach['cpu']} (host {mach['host']}); diffractom {results.get('diffractom', '?')}. Wall time of the notebooks "
        "of each stage, with the parameters they contain; the integration is shared by both TT pipelines and "
        "counted in both totals."
        + (f" Times marked † were not rerun: measured {', '.join(reused)}"
           + (", a node with the same hardware (A40, same CPU model and memory)." if any(not r.endswith(mach['host']) for r in reused)
              else ".") if reused else ""),
        "",
        "| Sample | Pipeline | " + " | ".join(c for c, _ in COLUMNS) + " | Total |",
        "|---|---|" + "---|" * (len(COLUMNS) + 1),
    ]
    for sample in SAMPLES:
        for key, name, cols in PIPELINES:
            cells, total, complete = [], 0.0, True
            for i, (_, nb) in enumerate(COLUMNS):
                if i not in cols:
                    cells.append("--")
                    continue
                t = times.get(nb.format(s=sample, p=key if key != "pbp" else "adaptive"))
                if t is None:
                    cells.append("not run")
                    complete = False
                    continue
                cells.append(fmt(t[0]) + (" †" if t[1] != results["date"] else ""))
                total += t[0]
            lines.append(f"| {label(sample)} | {name} | " + " | ".join(cells) + f" | **{fmt(total) if complete else '--'}** |")
    if results.get("metrics"):
        lines += ["", "## Accuracy of the maps", "",
                  "From `visualization/<sample>/figure_panels.ipynb` on the maps of this run: the median "
                  "misorientation to the ground truth over the sample, and the mean distance of the grain "
                  f"boundaries (KAM >= 4 deg) to the nearest ground-truth grain boundary (1 voxel = 4 display pixels).",
                  "", "| Sample | Map | Median misorientation (deg) | Grain-boundary deviation (µm) | (voxels) |",
                  "|---|---|---|---|---|"]
        for sample in SAMPLES:
            for name, m in results["metrics"].get(sample, {}).items():
                lines.append(f"| {label(sample)} | {name.replace('tt_', 'TT-').replace('adaptive', 'Adaptive').replace('uniform', 'Uniform')} | "
                             f"{m['median misorientation (deg)']:.3f} | "
                             f"{m['grain boundary deviation, mean (um)']:.4f} | "
                             f"{m['grain boundary deviation, mean (display pixels)'] / 4:.2f} |")
    lines += ["", "Per notebook:", "", "| Notebook | Wall time | Measured | Host |", "|---|---|---|---|"]
    for nb, (s, d, h) in times.items():
        lines.append(f"| `{nb}` | {fmt(s)} | {d} | {h} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", help="scratch directory for the copy (must not exist)")
    parser.add_argument("--inplace", action="store_true", help="run in this repository instead of a copy")
    parser.add_argument("--stages", default=None, help="comma-separated stages to run (default: all)")
    parser.add_argument("--reuse", default=None, help="result file to take the other stages' times from")
    parser.add_argument("--metrics", action="store_true", help="run the figure_panels notebooks and report accuracy")
    parser.add_argument("--allow-other-host", action="store_true",
                        help="accept --reuse times from another host (only one with the same hardware)")
    args = parser.parse_args()
    if args.inplace == bool(args.workdir):
        parser.error("give either --workdir or --inplace")
    stages = {s.strip() for s in args.stages.split(",")} if args.stages else {s for s, _, _ in STAGES}
    if not stages <= {s for s, _, _ in STAGES}:
        parser.error(f"unknown stage in {stages}")
    root = REPO if args.inplace else os.path.abspath(args.workdir)
    if not args.inplace:
        make_copy(root)
    out_json = os.path.join(REPO, "timing", "pipeline_timings.json")
    out_md = os.path.join(REPO, "timing", "pipeline_timings.md")
    today = datetime.now().strftime("%Y-%m-%d")
    results = {"date": today, "machine": machine(), "diffractom": diffractom_version(),
               "workdir": root, "runs": []}
    print(json.dumps(results["machine"]), results["diffractom"], flush=True)

    reuse = None
    if args.reuse:
        reuse = json.load(open(args.reuse))
        if reuse["machine"]["host"] != results["machine"]["host"] and not args.allow_other_host:
            raise SystemExit(f"--reuse was measured on {reuse['machine']['host']}, this is {results['machine']['host']}")
        reuse_t = notebook_times(reuse)

    done = {}  # notebooks already run (the integration is shared)
    for sample in SAMPLES:
        for stage, pipelines, notebooks in STAGES:
            times, measured, host = {}, today, results["machine"]["host"]
            for nb in notebooks:
                rel = nb.format(s=sample)
                if stage in stages:
                    if rel not in done:
                        print(f"[{datetime.now():%H:%M:%S}] {rel} ...", flush=True)
                        done[rel] = run_notebook(os.path.join(root, rel))
                        print(f"  {fmt(done[rel])}", flush=True)
                    times[rel] = done[rel]
                elif reuse and rel in reuse_t:
                    times[rel], measured, host = reuse_t[rel]
            if not times:
                continue
            for pipeline in pipelines:
                results["runs"].append({"sample": sample, "pipeline": pipeline, "stage": stage,
                                        "notebooks": times, "seconds": sum(times.values()), "measured": measured,
                                        "host": host})
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
    if args.metrics:
        results["metrics"] = {}
        for sample in SAMPLES:
            rel = METRICS_NB.format(s=sample)
            print(f"[{datetime.now():%H:%M:%S}] {rel} (not timed) ...", flush=True)
            run_notebook(os.path.join(root, rel))
            results["metrics"][sample] = read_metrics(sample, root)
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
    write_table(results, out_md)
    print(open(out_md).read(), flush=True)


if __name__ == "__main__":
    main()
