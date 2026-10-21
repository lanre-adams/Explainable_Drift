"""FastAPI server: run pilots in the background, poll progress, view results, download PDF/CSV/JSON."""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import fields
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .engine import Config, load_data, run_pilot
from .report import build_pdf

ROOT = Path(__file__).resolve().parent
RUNS = Path("/app/runs") if Path("/app").exists() else ROOT.parent / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Explanation-Drift Pilot")
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
WORKER = threading.Semaphore(1)  # one run at a time: runs are CPU-heavy


def _save(run_id, obj, name):
    (RUNS / run_id).mkdir(exist_ok=True)
    (RUNS / run_id / name).write_text(json.dumps(obj, default=float))


def _worker(run_id, cfg):
    with WORKER:
        JOBS[run_id].update(status="running", message="Loading data", started=time.time())
        def prog(done, total, msg):
            JOBS[run_id].update(done=done, total=total, message=msg)
        try:
            res = run_pilot(cfg, prog)
            df = res.pop("_windows")
            d = RUNS / run_id; d.mkdir(exist_ok=True)
            df.to_csv(d / "windows.csv", index=False)
            _save(run_id, res, "results.json")
            (d / "report.pdf").write_bytes(build_pdf(res, run_id))
            JOBS[run_id].update(status="done", message="Complete", done=JOBS[run_id]["total"], finished=time.time())
        except Exception as e:  # surface the error to the UI
            JOBS[run_id].update(status="error", message=f"{type(e).__name__}: {e}")
        _save(run_id, JOBS[run_id], "job.json")


@app.on_event("startup")
def _startup():
    for d in RUNS.iterdir():
        if (d / "job.json").exists():
            JOBS[d.name] = json.loads((d / "job.json").read_text())
    threading.Thread(target=load_data, daemon=True).start()  # warm the data cache


@app.get("/api/defaults")
def defaults():
    c = Config()
    return dict(config={f.name: getattr(c, f.name) for f in fields(Config)}, test_sets=["KDDTest+", "KDDTest-21"])


@app.post("/api/estimate")
def estimate(body: dict):
    cfg = _cfg(body)
    return dict(windows=cfg.total_windows(), seconds=round(cfg.total_windows() * 0.33 * cfg.window_size / 1000 + 8 * cfg.seeds))


def _cfg(body):
    names = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in body.items() if k in names}).validate()


@app.post("/api/runs")
def create_run(body: dict):
    cfg = _cfg(body)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    with LOCK:
        queued = sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))
        JOBS[run_id] = dict(id=run_id, status="queued", message="Waiting for a free worker" if queued else "Starting",
                            done=0, total=cfg.total_windows(), created=time.time(), config=vars(cfg).copy())
    threading.Thread(target=_worker, args=(run_id, cfg), daemon=True).start()
    return JOBS[run_id]


@app.get("/api/runs")
def list_runs():
    return sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    job = JOBS.get(run_id)
    if not job:
        raise HTTPException(404, "Unknown run")
    out = dict(job)
    if job["status"] == "done":
        out["results"] = json.loads((RUNS / run_id / "results.json").read_text())
    return JSONResponse(out)


def _file(run_id, name, media, dl):
    p = RUNS / run_id / name
    if run_id not in JOBS or not p.exists():
        raise HTTPException(404, "Not available")
    return FileResponse(p, media_type=media, filename=dl)


@app.get("/api/runs/{run_id}/report.pdf")
def report(run_id: str):
    return _file(run_id, "report.pdf", "application/pdf", f"explanation-drift-pilot-{run_id}.pdf")


@app.get("/api/runs/{run_id}/windows.csv")
def windows(run_id: str):
    return _file(run_id, "windows.csv", "text/csv", f"windows-{run_id}.csv")


@app.get("/api/runs/{run_id}/results.json")
def results(run_id: str):
    return _file(run_id, "results.json", "application/json", f"results-{run_id}.json")


@app.get("/api/health")
def health():
    return dict(ok=True)


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
