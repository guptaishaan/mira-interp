#!/usr/bin/env python3
"""Execute the frozen selection then confirmation study, auditing every stage."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
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


def pilot_gates(resource):
    specifications={
        "generation_pilot_audit":(GEN/"pilot_audit.json","passed_rollout_generation_audit"),
        "storage_parity_audit":(GEN/"pilot_parity_audit.json","passed_storage_only_pilot_parity"),
        "measurement_pilot_audit":(EVAL/"pilot_audit.json","passed_generated_measurement_audit")}
    entries=resource["required_passed_audits"]
    require(len(entries)==3 and {entry["name"] for entry in entries}==set(specifications),"Exact three named pilot gates required")
    reports={}
    for entry in entries:
        path,status=specifications[entry["name"]]
        require(Path(entry["path"]).resolve()==path.resolve() and sha(path)==entry["sha256"],"Reviewed named pilot audit changed")
        reports[entry["name"]]=read(path)
        require(reports[entry["name"]]["status"]==status,"A named pilot gate did not pass")
    generation=read(GEN/"pilot.json");measurement=read(EVAL/"pilot.json")
    require(generation["status"]=="passed_rollout_steering_pilot" and generation["n_records"]==2
            and generation["registration_sha256"]==sha(GEN/"registration.json"),"Current generation pilot is incomplete")
    require(reports["generation_pilot_audit"]["manifest_sha256"]==sha(GEN/"pilot.json"),"Generation pilot audit is stale")
    parity=reports["storage_parity_audit"]
    for key,path in {"old_manifest_sha256":ROOT/"results/rollout_steering_v1/pilot.json",
        "new_manifest_sha256":GEN/"pilot.json", "old_audit_sha256":ROOT/"results/rollout_steering_v1/pilot_audit.json",
        "new_audit_sha256":GEN/"pilot_audit.json", "old_registration_sha256":ROOT/"results/rollout_steering_v1/registration.json",
        "new_registration_sha256":GEN/"registration.json", "review_sha256":ROOT/"results/rollout_storage_v2_review.json"}.items():
        require(parity[key]==sha(path),"Storage parity gate is stale")
    require(all(parity[key] is True for key in ("all_model_output_arrays_bitwise_equal",
        "all_actions_traces_raw_hashes_sites_probe_effects_exact","same_frozen_science_and_implementation_except_storage",
        "only_encoding_paths_provenance_and_runtime_metadata_differ")),"Storage parity did not fully pass")
    require(measurement["status"]=="passed_frozen_generated_video_evaluation" and measurement["records"]==2
            and measurement["generation_manifest_sha256"]==sha(GEN/"pilot.json")
            and measurement["generation_audit_sha256"]==sha(GEN/"pilot_audit.json")
            and measurement["protocol_sha256"]==sha(ROOT/"configs/generated_evaluation_v1.json")
            and measurement["repeated_predictions_bitwise_equal"] is True
            and reports["measurement_pilot_audit"]["evaluation_sha256"]==sha(EVAL/"pilot.json"),"Current measured pilot gate is stale")


def interrupted(signum,frame):
    raise KeyboardInterrupt(f"Supervisor received signal{signum}")


def stop_owned_group(process):
    if process is None or process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def main():
    require(not STATUS.exists(),"Inspect an existing supervisor status before resumption")
    resource_path=GEN/"resource_review.json"
    resource=read(resource_path)
    require(resource["status"]=="approved_by_executing_agent_within_user_authorization"
            and resource["registration_sha256"]==sha(GEN/"registration.json"),"Matching resource review required")
    pilot_gates(resource)
    require(shutil.disk_usage("/data2/ishaangp/mira-interp").free > resource["projected_generated_bytes"]+25*1024**3,
            "Insufficient space for the complete registered run plus reserve")
    for phase in ("selection","confirmation"):
        require(not (GEN/(phase+".json")).exists() and not (EVAL/(phase+".json")).exists(),"Main outputs must start fresh")
    state={"status":"running","pid":os.getpid(),"started_unix":time.time(),"resource_review_sha256":sha(resource_path),
           "script_sha256":sha(Path(__file__)),"completed_stages":[],"current_stage":None}
    paths={resource_path,GEN/"registration.json",GEN/"pilot.json",GEN/"pilot_audit.json",GEN/"pilot_parity_audit.json",
           EVAL/"pilot.json",EVAL/"pilot_audit.json",ROOT/"configs/generated_evaluation_v1.json",Path(__file__)}
    paths.update(ROOT/name for name in ("scripts/run_rollout_phase.py","scripts/audit_rollout_generation.py",
        "scripts/audit_generated_measurements.py","scripts/audit_rollout_storage_parity.py","scripts/package_development.py"))
    paths.update(ROOT/name for name in read(GEN/"registration.json")["code_sha256"])
    paths.update(ROOT/name for name in read(ROOT/"configs/generated_evaluation_v1.json")["code_sha256"])
    state["fixed_bindings"]={str(path.resolve()):sha(path) for path in paths}
    def unchanged():
        require(all(sha(path)==digest for path,digest in state["fixed_bindings"].items()),"Fixed supervisor prerequisite or code changed")
        require(all(sha(row["path"])==row["sha256"] for row in state["completed_stages"]),"Completed stage report changed")
        pilot_gates(resource)
    save(state)
    signal.signal(signal.SIGTERM,interrupted)
    process=None
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
                unchanged()
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
                    process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
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
                unchanged()
                save(state)
        unchanged()
        state.update(status="passed_generation_and_measurement_pending_publication",current_stage=None,
                     elapsed_seconds=time.time()-state["started_unix"],physical_control_established=False)
        save(state)
    except BaseException as error:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:stop_owned_group(process)
        except BaseException as cleanup_error:
            state["cleanup_error"]={"type":type(cleanup_error).__name__,"message":str(cleanup_error)}
        state.update(status="failed",error_type=type(error).__name__,error=str(error),elapsed_seconds=time.time()-state["started_unix"])
        save(state)
        raise


if __name__=="__main__":main()
