#!/usr/bin/env python3
"""Durable launcher for one registered generation phase on physical GPUs6/7."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


def stop_owned(jobs):
    """Each Popen owns a new session; never signal an untracked process group."""
    errors = []
    active = []
    for process, stream, row in jobs:
        row["exit_code"] = process.poll()
        if row["exit_code"] is None:
            active.append((process, stream, row))
    for process, _, _ in active:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(str(error))
    deadline = time.monotonic() + 10
    for process, _, row in active:
        try:
            process.wait(timeout=max(0, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        # Also remove descendants whose parent exited after SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(str(error))
        try:
            row["exit_code"] = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            errors.append(f"Owned process {process.pid} did not exit after SIGKILL")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "selection", "confirmation"), required=True)
    parser.add_argument("--version", type=int, choices=(1,2), default=2)
    args = parser.parse_args()
    directory = ROOT/f"results/rollout_steering_v{args.version}"
    producer = ROOT/"scripts"/("rollout_steering.py" if args.version==1 else "rollout_steering_v2.py")
    status_path = directory/(args.phase+"_runner.json")
    if status_path.exists() or (directory/(args.phase+".json")).exists():
        raise RuntimeError("Inspect existing phase/runner output before resumption")
    assignments = [(7,None)] if args.phase=="pilot" else [(6,0),(7,1)]
    planned_logs = [directory/(args.phase+("" if index is None else f"_worker{index}")+".log")
                    for _,index in assignments]
    if args.phase != "pilot":
        planned_logs.append(directory/(args.phase+"_aggregate.log"))
        if any((directory/f"{args.phase}_worker{index}.json").exists() for _,index in assignments):
            raise RuntimeError("Existing worker reports must be inspected before resumption")
    if any(path.exists() for path in planned_logs):
        raise RuntimeError("Existing phase logs must be inspected before resumption")
    # These two physical devices were authorized in this session; never select others.
    for gpu,_ in assignments:
        processes = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],text=True)
        if processes.strip():
            raise RuntimeError(f"GPU{gpu} has an active process; inspect ownership before launch")
    state = {"status":"running", "phase":args.phase,"version":args.version,"assignments":assignments,
             "pid":os.getpid(),"workers":[],"started_unix":time.time(),"current_stage":"workers"}
    write(status_path,state)
    jobs, streams = [], []
    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    def launch(command, log, environment, row):
        stream = log.open("x")
        streams.append(stream)  # Close even when Popen fails before returning a PID.
        process = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        row.update(pid=process.pid,process_group=process.pid,log=str(log),exit_code=None)
        jobs.append((process,stream,row))
        return process
    try:
        for gpu,index in assignments:
            suffix = "" if index is None else f"_worker{index}"
            log=directory/(args.phase+suffix+".log")
            command=[str(ROOT/".venv/bin/python"),str(producer),"--phase",args.phase]
            if index is not None: command.extend(["--worker-index",str(index)])
            environment=os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=str(gpu),NUMPY_MADVISE_HUGEPAGE="0",OPENBLAS_NUM_THREADS="8",OMP_NUM_THREADS="8")
            row={"gpu":gpu,"worker_index":index}
            state["workers"].append(row)
            launch(command,log,environment,row)
            write(status_path,state)
        while True:
            for process,_,row in jobs:
                row["exit_code"] = process.poll()
            if any(row["exit_code"] not in (None,0) for _,_,row in jobs):
                raise RuntimeError("Generation worker failed; stop siblings and preserve output")
            write(status_path,state)
            if all(row["exit_code"] == 0 for _,_,row in jobs):
                break
            time.sleep(0.5)
        if args.phase!="pilot":
            state["current_stage"] = "aggregate"
            aggregate=[str(ROOT/".venv/bin/python"),str(producer),"--phase",args.phase,"--aggregate"]
            environment=os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES="",NUMPY_MADVISE_HUGEPAGE="0",OPENBLAS_NUM_THREADS="8",OMP_NUM_THREADS="8")
            row={"gpu":None,"kind":"aggregate"}
            state["aggregate"] = row
            process = launch(aggregate,directory/(args.phase+"_aggregate.log"),environment,row)
            write(status_path,state)
            row["exit_code"] = state["aggregate_exit_code"] = process.wait()
            if row["exit_code"] != 0:
                raise RuntimeError("Aggregate integrity checks failed")
        report = json.loads((directory/(args.phase+".json")).read_text())
        expected = "passed_rollout_steering_pilot" if args.phase=="pilot" else "passed_rollout_generation"
        if report.get("status") != expected or report.get("n_records") != {"pilot":2,"selection":420,"confirmation":924}[args.phase]:
            raise RuntimeError("Producer exited without complete phase coverage")
        state.update(status="passed_generation_pending_independent_audit",current_stage=None,
                     elapsed_seconds=time.time()-state["started_unix"])
        write(status_path,state)
    except BaseException as error:
        # Finish owned-child cleanup even if another termination signal arrives.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        cleanup_errors = stop_owned(jobs)
        state.update(status="failed",error_type=type(error).__name__,error=str(error),
                     cleanup_errors=cleanup_errors,elapsed_seconds=time.time()-state["started_unix"])
        write(status_path,state)
        raise
    finally:
        for stream in streams:
            stream.close()
        for sig,handler in previous_handlers.items():
            signal.signal(sig,handler)


if __name__=="__main__":main()
