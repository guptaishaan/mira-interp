#!/usr/bin/env python3
"""Execute the frozen selection then confirmation study, auditing every stage."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
GEN=ROOT/"results/rollout_steering_v2"
EVAL=ROOT/"results/generated_evaluation_v2"
PY=str(ROOT/".venv/bin/python")
STATUS=GEN/"study_status.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(value):
    temporary=STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(STATUS)


def require(condition,message):
    if not condition:raise RuntimeError(message)


def main():
    require(not STATUS.exists(),"Inspect an existing supervisor status before resumption")
    resource_path=GEN/"resource_review.json"
    resource=read(resource_path)
    require(resource["status"]=="approved_by_executing_agent_within_user_authorization"
            and resource["registration_sha256"]==sha(GEN/"registration.json"),"Matching resource review required")
    for entry in resource["required_passed_audits"]:
        require(sha(entry["path"])==entry["sha256"],"Reviewed pilot audit changed")
    require(shutil.disk_usage("/data2/ishaangp/mira-interp").free > resource["projected_generated_bytes"]+25*1024**3,
            "Insufficient space for the complete registered run plus reserve")
    for phase in ("selection","confirmation"):
        require(not (GEN/(phase+".json")).exists() and not (EVAL/(phase+".json")).exists(),"Main outputs must start fresh")
    state={"status":"running","pid":os.getpid(),"started_unix":time.time(),"resource_review_sha256":sha(resource_path),
           "script_sha256":sha(Path(__file__)),"completed_stages":[],"current_stage":None}
    save(state)
    try:
        for phase,count in (("selection",420),("confirmation",924)):
            stages=[
                ("generation",[PY,"scripts/run_rollout_phase.py","--phase",phase,"--version","2"],
                 GEN/(phase+".json"),"passed_rollout_generation",None),
                ("generation_audit",[PY,"scripts/audit_rollout_generation.py","--phase",phase,"--directory",str(GEN)],
                 GEN/(phase+"_audit.json"),"passed_rollout_generation_audit",None),
                ("video_measurement",[PY,"scripts/evaluate_rollouts.py","--phase",phase,"--generation-dir",str(GEN),
                 "--output-dir",str(EVAL),"--cache-dir","/data2/ishaangp/mira-interp/generated_evaluation_v2"],
                 EVAL/(phase+".json"),"passed_frozen_generated_video_evaluation",6),
                ("measurement_audit",[PY,"scripts/audit_generated_measurements.py","--phase",phase,"--generation-dir",str(GEN),
                 "--evaluation-dir",str(EVAL),"--cache-dir","/data2/ishaangp/mira-interp/generated_evaluation_v2"],
                 EVAL/(phase+"_audit.json"),"passed_generated_measurement_audit",None)]
            for name,command,result_path,expected,gpu in stages:
                stage=phase+"_"+name
                state.update(current_stage=stage,stage_started_unix=time.time());save(state)
                environment=os.environ.copy()
                environment.update(NUMPY_MADVISE_HUGEPAGE="0",OPENBLAS_NUM_THREADS="8",OMP_NUM_THREADS="8")
                if gpu is not None:
                    active=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader"],text=True)
                    require(not active.strip(),"Evaluation GPU is occupied; inspect ownership")
                    environment["CUDA_VISIBLE_DEVICES"]=str(gpu)
                log=GEN/(stage+"_supervisor.log")
                with log.open("x") as output:
                    process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=output,stderr=subprocess.STDOUT)
                    state["stage_pid"]=process.pid;save(state)
                    code=process.wait()
                require(code==0,f"{stage} exited{code}; inspect {log}")
                report=read(result_path)
                require(report["status"]==expected,f"{stage} did not produce its passed result")
                if name=="generation":require(report["n_records"]==count,"Incomplete generation grid")
                if name=="generation_audit":require(report["manifest_sha256"]==sha(GEN/(phase+".json")),"Generation audit binding differs")
                if name=="video_measurement":require(report["records"]==count,"Incomplete video measurements")
                if name=="measurement_audit":require(report["evaluation_sha256"]==sha(EVAL/(phase+".json")),"Measurement audit binding differs")
                state["completed_stages"].append({"stage":stage,"path":str(result_path),"sha256":sha(result_path),
                                                  "elapsed_seconds":time.time()-state["stage_started_unix"]})
                save(state)
        state.update(status="passed_generation_and_measurement_pending_publication",current_stage=None,
                     elapsed_seconds=time.time()-state["started_unix"],physical_control_established=False)
        save(state)
    except Exception as error:
        state.update(status="failed",error_type=type(error).__name__,error=str(error),elapsed_seconds=time.time()-state["started_unix"])
        save(state)
        raise


if __name__=="__main__":main()
