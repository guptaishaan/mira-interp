#!/usr/bin/env python3
"""Durable launcher for one registered generation phase on physical GPUs6/7."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "selection", "confirmation"), required=True)
    args = parser.parse_args()
    directory = ROOT/"results/rollout_steering_v1"
    status_path = directory/(args.phase+"_runner.json")
    if status_path.exists() or (directory/(args.phase+".json")).exists():
        raise RuntimeError("Inspect existing phase/runner output before resumption")
    assignments = [(7,None)] if args.phase=="pilot" else [(6,0),(7,1)]
    # These two physical devices were authorized in this session; never select others.
    for gpu,_ in assignments:
        processes = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],text=True)
        if processes.strip():
            raise RuntimeError(f"GPU{gpu} has an active process; inspect ownership before launch")
    state = {"status":"running", "phase":args.phase,"assignments":assignments,"workers":[],"started_unix":time.time()}
    write(status_path,state)
    jobs=[]
    for gpu,index in assignments:
        suffix = "" if index is None else f"_worker{index}"
        log=directory/(args.phase+suffix+".log")
        command=[str(ROOT/".venv/bin/python"),str(ROOT/"scripts/rollout_steering.py"),"--phase",args.phase]
        if index is not None: command.extend(["--worker-index",str(index)])
        environment=os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu),NUMPY_MADVISE_HUGEPAGE="0",OPENBLAS_NUM_THREADS="8",OMP_NUM_THREADS="8")
        stream=log.open("x")
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        row={"gpu":gpu,"worker_index":index,"pid":process.pid,"log":str(log),"exit_code":None}
        state["workers"].append(row);jobs.append((process,stream,row))
        write(status_path,state)
    while any(process.poll() is None for process,_,_ in jobs):
        for process,_,row in jobs:row["exit_code"]=process.poll()
        write(status_path,state)
        time.sleep(3)
    for process,stream,row in jobs:
        row["exit_code"]=process.wait();stream.close()
    if any(row["exit_code"]!=0 for _,_,row in jobs):
        state.update(status="failed",elapsed_seconds=time.time()-state["started_unix"])
        write(status_path,state)
        raise RuntimeError("Generation worker failed; preserve and inspect its output")
    if args.phase!="pilot":
        aggregate=[str(ROOT/".venv/bin/python"),str(ROOT/"scripts/rollout_steering.py"),"--phase",args.phase,"--aggregate"]
        with (directory/(args.phase+"_aggregate.log")).open("x") as output:
            completed=subprocess.run(aggregate,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        state["aggregate_exit_code"]=completed.returncode
        if completed.returncode:
            state["status"]="failed";write(status_path,state)
            raise RuntimeError("Aggregate integrity checks failed")
    state.update(status="passed_generation_pending_independent_audit",elapsed_seconds=time.time()-state["started_unix"])
    write(status_path,state)


if __name__=="__main__":main()
