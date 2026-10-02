"""
Wall-clock time of the full reconstruction pipelines for the simulated samples, all on one machine.

    python timing/time_pipelines.py --workdir /path/to/scratch/copy

The repository's code and notebooks are copied to `workdir` (with `data/` linked to this repository's
data), so the timed runs write their outputs there and leave this repository's results untouched. The
notebooks of every stage are then executed in order, with the parameters they contain, and the wall time
of every notebook is recorded:

    TT-Adaptive: peak segmentation, peak indexing, integration, TT reconstruction
    TT-Uniform:  integration, TT reconstruction

The integration is shared by both pipelines and counted in both totals. Notebooks that only choose
parameters or inspect results (segment_peaks/01_find_parameters, 03_inspect, indexing/01_inspect_peaks)
are not timed. Results go to timing/pipeline_timings.{json,md} in this repository; the json is written
after every notebook, so a partial run keeps what was measured.
"""
import argparse
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
    ("Peak segmentation", ("adaptive",), ["adaptive_basis/{s}/segment_peaks/02_segment.ipynb"]),
    ("Peak indexing", ("adaptive",), ["adaptive_basis/{s}/indexing/02_pbp_index.ipynb",
                                      "adaptive_basis/{s}/indexing/03_refine.ipynb"]),
    ("Integration", ("adaptive", "uniform"), ["integration/{s}/01_inspect_integrate.ipynb"]),
    ("TT reconstruction", ("adaptive",), ["texture_tomography/{s}/textomo_adaptive.ipynb"]),
    ("TT reconstruction", ("uniform",), ["texture_tomography/{s}/textomo_uniform.ipynb"]),
]
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
        nbformat.write(nb, path)  # executed copy, with outputs, stays in the work directory
    return time.perf_counter() - t0


def machine():
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    return {"host": platform.node(), "gpu": gpu[0] if gpu else "?", "cpu": cpu,
            "cores": len(os.sched_getaffinity(0)), "job": os.environ.get("LSB_JOBID")}


def fmt(seconds):
    m = seconds / 60
    return f"{m:.1f} min" if m < 60 else f"{m / 60:.2f} h ({m:.0f} min)"


def write_table(results, path):
    lines = [
        "# Pipeline timings (simulated samples)",
        "",
        f"Measured {results['date']} with `timing/time_pipelines.py`, all stages in one job on one machine: "
        f"{results['machine']['gpu']}, {results['machine']['cores']} cores of {results['machine']['cpu']} "
        f"(host {results['machine']['host']}). Wall time of the notebooks of each stage, with the parameters "
        "they contain; the integration is shared by both pipelines and counted in both totals.",
        "",
        "| Sample | Pipeline | Peak segmentation | Peak indexing | Integration | TT reconstruction | Total |",
        "|---|---|---|---|---|---|---|",
    ]
    for sample in SAMPLES:
        t = {(r["stage"], r["pipeline"]): r["seconds"] for r in results["runs"] if r["sample"] == sample}
        for pipeline in ("adaptive", "uniform"):
            cells, total = [], 0.0
            for stage in ("Peak segmentation", "Peak indexing", "Integration", "TT reconstruction"):
                s = t.get((stage, pipeline))
                cells.append("--" if s is None else fmt(s))
                total += s or 0.0
            label = "TT-Adaptive" if pipeline == "adaptive" else "TT-Uniform"
            lines.append(f"| {sample.split('_')[-1].replace('p', '.').replace('deg', ' deg')} | {label} | "
                         + " | ".join(cells) + f" | **{fmt(total)}** |")
    lines += ["", "Per notebook:", "", "| Notebook | Wall time |", "|---|---|"]
    seen = set()
    for r in results["runs"]:
        for nb, s in r["notebooks"].items():
            if nb not in seen:
                seen.add(nb)
                lines.append(f"| `{nb}` | {fmt(s)} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, help="scratch directory for the copy (must not exist)")
    args = parser.parse_args()
    workdir = os.path.abspath(args.workdir)
    make_copy(workdir)
    out_json = os.path.join(REPO, "timing", "pipeline_timings.json")
    out_md = os.path.join(REPO, "timing", "pipeline_timings.md")
    results = {"date": datetime.now().strftime("%Y-%m-%d"), "machine": machine(), "workdir": workdir, "runs": []}
    print(json.dumps(results["machine"]), flush=True)

    done = {}  # notebooks already run (the integration is shared)
    for sample in SAMPLES:
        for stage, pipelines, notebooks in STAGES:
            times = {}
            for nb in notebooks:
                rel = nb.format(s=sample)
                if rel not in done:
                    print(f"[{datetime.now():%H:%M:%S}] {rel} ...", flush=True)
                    done[rel] = run_notebook(os.path.join(workdir, rel))
                    print(f"  {fmt(done[rel])}", flush=True)
                times[rel] = done[rel]
            for pipeline in pipelines:
                results["runs"].append({"sample": sample, "pipeline": pipeline, "stage": stage,
                                        "notebooks": times, "seconds": sum(times.values())})
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
    write_table(results, out_md)
    print(open(out_md).read(), flush=True)


if __name__ == "__main__":
    main()
