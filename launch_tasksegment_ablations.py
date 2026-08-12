#!/usr/bin/env python3
"""Dynamic 5-GPU queue for TaskSegment ablations.

Assumes scripts/train.py supports these no-bank variants:
  no_text_conditioning, no_visual_prototype, no_dense_guidance

Usage:
  python launch_tasksegment_ablations.py --preflight
  nohup python launch_tasksegment_ablations.py > paper_artifacts/ablation_queue/launcher.out 2>&1 &
  python launch_tasksegment_ablations.py --status
"""
from __future__ import annotations

import argparse, json, os, queue, shlex, signal, subprocess, sys, threading, traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

VERSION = "2026-07-27-dynamic-v1"
DEFAULT_GPUS = [0, 1, 4, 5, 6]
DEFAULT_SEEDS = [42, 3407, 2026]
ABLATIONS = ["no_text_conditioning", "no_dense_guidance", "no_visual_prototype"]
EXPERIMENT = {
    "no_text_conditioning": "tasksegment_no_text_conditioning",
    "no_dense_guidance": "tasksegment_no_dense_guidance_nobank",
    "no_visual_prototype": "tasksegment_no_visual_prototype",
}
STOP = threading.Event()
ACTIVE = {}
ACTIVE_LOCK = threading.Lock()

@dataclass(frozen=True)
class Job:
    ablation: str
    seed: int
    @property
    def experiment(self): return EXPERIMENT[self.ablation]
    @property
    def job_id(self): return f"{self.experiment}__seed{self.seed}"

def now(): return datetime.now(timezone.utc).isoformat()
def stamp(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def qcmd(parts): return " ".join(shlex.quote(str(x)) for x in parts)
def parse_ints(s): return [int(x) for x in s.split(",") if x.strip()]
def parse_strs(s): return [x.strip() for x in s.split(",") if x.strip()]

def run_dir(root, job): return root / "paper_artifacts/runs" / job.experiment / f"seed_{job.seed}"
def checkpoint(root, job): return run_dir(root, job) / "checkpoint.pth"
def summary(root, job): return run_dir(root, job) / "predictions/summary_metrics.csv"
def complete(root, job):
    c, s = checkpoint(root, job), summary(root, job)
    return c.exists() and c.stat().st_size > 1_000_000 and s.exists() and s.stat().st_size > 100

class Logger:
    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
    def write(self, event, **kw):
        rec = {"time": now(), "event": event, **kw}
        line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f: f.write(line + "\n")
        print(line, flush=True)

def backup_partial(root, job, log):
    d = run_dir(root, job)
    if not d.exists() or complete(root, job): return
    dst = d.with_name(d.name + "_incomplete_" + stamp())
    n = 1
    while dst.exists():
        dst = d.with_name(d.name + "_incomplete_" + stamp() + f"_{n}"); n += 1
    d.rename(dst)
    log.write("partial_backed_up", job_id=job.job_id, backup=str(dst))

def train_cmd(root, py, text, job):
    d = run_dir(root, job)
    return [str(py), "-u", str(root/"scripts/train.py"),
        "--data-root", str(root/"data"), "--text-dir", str(text),
        "--save-path", str(d/"checkpoint.pth"), "--image-size", "512", "512",
        "--device", "cuda:0", "--seed", str(job.seed), "--deterministic",
        "--ablation", job.ablation, "--epochs", "150", "--batch-size", "4",
        "--episodes-per-epoch", "1500", "--grad-accum-steps", "4",
        "--lr", "5e-5", "--min-lr", "1e-6", "--weight-decay", "1e-4",
        "--warmup-epochs", "5", "--patience", "20", "--domain-sampling-alpha", "0.0",
        "--prototype-blend-target", "0.9", "--prototype-warmup-epochs", "5",
        "--prototype-ramp-epochs", "10", "--prototype-refresh-samples", "128",
        "--prototype-eval-support-samples", "4", "--prototype-refresh-batch-size", "4",
        "--prototype-refresh-interval", "1", "--cache-dataset", "--preload-val-cache", "--amp"]

def predict_cmd(root, py, text, job):
    d = run_dir(root, job)
    return [str(py), "-u", str(root/"scripts/predict.py"),
        "--model-path", str(d/"checkpoint.pth"), "--data-root", str(root/"data"),
        "--text-dir", str(text), "--save-dir", str(d/"predictions"),
        "--image-size", "512", "512", "--device", "cuda:0", "--seed", str(job.seed),
        "--cache-dataset", "--preload-cache", "--num-vis-per-organ", "0", "--threshold", "0.5"]

def run_process(cmd, cwd, env, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8", buffering=1) as f:
        f.write("\n" + "="*100 + f"\n# launcher_started={now()}\n# command={qcmd(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        with ACTIVE_LOCK: ACTIVE[p.pid] = p
        try: rc = p.wait()
        finally:
            with ACTIVE_LOCK: ACTIVE.pop(p.pid, None)
        f.write(f"# launcher_finished={now()} return_code={rc}\n")
        return rc

def signal_handler(sig, frame):
    del frame
    STOP.set(); print(f"received signal {sig}", flush=True)
    with ACTIVE_LOCK: procs = list(ACTIVE.values())
    for p in procs:
        try: os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError: pass

def worker(gpu, q, root, py, text, log, dry):
    while not STOP.is_set():
        try: job = q.get_nowait()
        except queue.Empty:
            log.write("worker_empty", gpu=gpu); return
        try:
            if complete(root, job):
                log.write("skip_complete", gpu=gpu, job_id=job.job_id); continue
            backup_partial(root, job, log)
            d = run_dir(root, job); d.mkdir(parents=True, exist_ok=True)
            lock = d/".ablation_queue.lock"
            if lock.exists():
                log.write("skip_locked", gpu=gpu, job_id=job.job_id); continue
            lock.write_text(json.dumps({"gpu":gpu,"pid":os.getpid(),"time":now()}), encoding="utf-8")
            tc, pc = train_cmd(root, py, text, job), predict_cmd(root, py, text, job)
            log.write("start", gpu=gpu, job_id=job.job_id, train=qcmd(tc), predict=qcmd(pc))
            if dry:
                lock.unlink(missing_ok=True); log.write("dry_complete", gpu=gpu, job_id=job.job_id); continue
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["PYTHONUNBUFFERED"] = "1"
            rc = run_process(tc, root, env, d/"train.log")
            if rc:
                lock.rename(d/".failed_train"); log.write("train_failed", gpu=gpu, job_id=job.job_id, rc=rc); continue
            if not checkpoint(root, job).exists():
                lock.rename(d/".missing_checkpoint"); log.write("missing_checkpoint", gpu=gpu, job_id=job.job_id); continue
            rc = run_process(pc, root, env, d/"predict.log")
            if rc:
                lock.rename(d/".failed_predict"); log.write("predict_failed", gpu=gpu, job_id=job.job_id, rc=rc); continue
            if not summary(root, job).exists():
                lock.rename(d/".missing_summary"); log.write("missing_summary", gpu=gpu, job_id=job.job_id); continue
            marker = {"run_id":job.job_id,"experiment":job.experiment,"ablation":job.ablation,
                      "seed":job.seed,"gpu":gpu,"checkpoint":str(checkpoint(root,job)),
                      "completed_at":now(),"launcher":VERSION,"prototype_bank_blending":False}
            (d/"completed.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
            lock.unlink(missing_ok=True); log.write("complete", gpu=gpu, job_id=job.job_id)
        except Exception:
            log.write("exception", gpu=gpu, job_id=getattr(job,"job_id","?"), traceback=traceback.format_exc())
        finally: q.task_done()

def preflight(root, py, text, jobs):
    missing = [p for p in [root/"scripts/train.py", root/"scripts/predict.py", root/"data", text, py] if not p.exists()]
    if missing: return ["missing: " + str(p) for p in missing]
    p = subprocess.run([str(py), str(root/"scripts/train.py"), "--help"], cwd=root, capture_output=True, text=True)
    h = (p.stdout or "") + (p.stderr or "")
    errors = [] if p.returncode == 0 else ["train.py --help failed"]
    for a in sorted({j.ablation for j in jobs}):
        if a not in h: errors.append(f"train.py does not advertise ablation: {a}")
    return errors

def status(path):
    if not path.exists(): print("status file missing:", path); return 1
    latest = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try: r = json.loads(line)
        except: continue
        if r.get("job_id"): latest[r["job_id"]] = r
    for k,v in sorted(latest.items()): print(f"{k:52s} {v.get('event'):18s} gpu={v.get('gpu','-')} {v.get('time','')}")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tank/wk/TSV3")
    ap.add_argument("--python", default="/root/miniconda3/envs/TS/bin/python")
    ap.add_argument("--text-dir", default="/tank/wk/TSV2/text_features_llm")
    ap.add_argument("--gpus", type=parse_ints, default=DEFAULT_GPUS)
    ap.add_argument("--seeds", type=parse_ints, default=DEFAULT_SEEDS)
    ap.add_argument("--ablations", type=parse_strs, default=ABLATIONS)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--version", action="store_true")
    a = ap.parse_args()
    if a.version: print(VERSION); return 0
    root, py, text = Path(a.root).resolve(), Path(a.python).resolve(), Path(a.text_dir).resolve()
    sf = root/"paper_artifacts/ablation_queue/events.jsonl"
    if a.status: return status(sf)
    unknown = [x for x in a.ablations if x not in EXPERIMENT]
    if unknown: print("unknown ablations:", unknown, file=sys.stderr); return 2
    jobs = [Job(ab, seed) for ab in a.ablations for seed in a.seeds]
    errors = preflight(root, py, text, jobs)
    if errors:
        print("preflight failed:", file=sys.stderr)
        for e in errors: print("-", e, file=sys.stderr)
        return 2
    print("GPUs:", a.gpus); print("jobs:")
    for i,j in enumerate(jobs,1): print(f"{i:02d}. {j.job_id} ({j.ablation})")
    if a.preflight: print("preflight passed"); return 0
    signal.signal(signal.SIGTERM, signal_handler); signal.signal(signal.SIGINT, signal_handler)
    log = Logger(sf); log.write("launcher_start", version=VERSION, pid=os.getpid(), gpus=a.gpus)
    q = queue.Queue()
    for j in jobs:
        if complete(root,j): log.write("already_complete", job_id=j.job_id)
        else: q.put(j)
    threads=[]
    for gpu in a.gpus:
        t=threading.Thread(target=worker,args=(gpu,q,root,py,text,log,a.dry_run),name=f"gpu{gpu}")
        t.start(); threads.append(t)
    for t in threads: t.join()
    log.write("launcher_finish", remaining=q.qsize(), stopped=STOP.is_set())
    return 130 if STOP.is_set() else (1 if q.qsize() else 0)

if __name__ == "__main__": raise SystemExit(main())
