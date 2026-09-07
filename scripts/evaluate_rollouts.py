#!/usr/bin/env python3
"""Frozen generated-video measurement; no adaptation or path selection."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "8"
import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mira/src")]
from mira_interp.model_loading import sha256
from mira_interp.rollout_evaluator import FrozenRolloutEvaluator, absolute_ball, validate_pairs

CODE = ["scripts/evaluate_rollouts.py", "src/mira_interp/rollout_evaluator.py",
        "src/mira_interp/video_evaluator.py", "src/mira_interp/video_evaluator_finetune.py",
        "src/mira_interp/video_evaluator_training.py"]
PATHS = ("probe_linear", "affine_forward", "quadratic_forward", "random_norm", "wrong_variable_ball_x")
DOSES = (-600., -300., 0., 300., 600.)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def code_hashes():
    return {name: sha256(ROOT/name) for name in CODE}


def register(args):
    require(not (args.generation_dir/"pilot.json").exists(), "Register measurements before generated pilot outcomes")
    report = ROOT/"results/videomae_finetune_v1.json"
    audit = ROOT/"results/videomae_finetune_v1_audit.json"
    require(read(audit)["status"] == "passed_development_video_evaluator_audit"
            and read(audit)["analysis_sha256"] == sha256(report), "Audited fixed evaluator required")
    config = {"status": "registered_before_generated_video_evaluation", "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": code_hashes(), "evaluator_report_sha256": sha256(report), "evaluator_audit_sha256": sha256(audit),
        "checkpoint_sha256": read(report)["checkpoint"]["sha256"], "primary": "view0 absolute ball location.z, estimated as ego.z plus ball-minus-ego.z",
        "secondary": "all4-view mean estimated ball height; all12 raw role changes;11-coordinate normalized role collateral excluding ball-minus-ego.z",
        "endpoint_frames": [17,19,21,23], "seconds_after_context": [.1,.2,.3,.4], "paths": list(PATHS), "requested_doses": list(DOSES),
        "ordering_metrics": "Spearman versus actual clipped dose, linear slope versus actual clipped dose, positive adjacent fraction, nondecreasing with nonzero span; constants are not successful steering",
        "contrast": "estimated height at requested+600 minus requested-600; fixed signed contrast, not selected from results",
        "collateral": "RMS standardized paired role change over11 non-target role coordinates at the two extreme doses; training head discovery scales remain frozen",
        "aggregation": "Average two paired seeds per match before equal-match means and1000 percentile match-bootstrap draws",
        "bootstrap_seed": 2026090703, "bootstrap_draws": 1000,
        "inference": "FP32 checkpoint parameters, BF16 autocast, deterministic algorithms, math attention; no optimizer",
        "generated_video_accuracy_established": False, "uncertainty_limit": "Bootstrap reflects match sampling only, not evaluator bias, model-seed uncertainty or simultaneous inference",
        "no_adaptation_or_selection": True,
        "clipped_duplicates": "Retain all nominal doses; duplicate effective doses must agree within1e-6uu to count as nondecreasing; retain inconsistency flags",
        "undefined_ranks": "Average finite seeds within each match, then equal-weight eligible matches; report all missing seed and match counts"}
    write(args.protocol, config)
    print("Registered fixed generated-video measurements before outcomes.", flush=True)


def verify(args):
    config = read(args.protocol)
    require(config["code_sha256"] == code_hashes(), "Frozen measurement implementation changed")
    for key, name in (("evaluator_report_sha256", "videomae_finetune_v1.json"),
                      ("evaluator_audit_sha256", "videomae_finetune_v1_audit.json")):
        require(config[key] == sha256(ROOT/"results"/name), "Frozen evaluator gate changed")
    if args.phase in ("selection", "confirmation"):
        previous = "pilot" if args.phase == "selection" else "selection"
        result_path = args.output_dir/(previous+".json")
        result, audit = read(result_path), read(args.output_dir/(previous+"_audit.json"))
        require(result["status"] == "passed_frozen_generated_video_evaluation"
                and result["protocol_sha256"] == sha256(args.protocol)
                and audit["status"] == "passed_generated_measurement_audit"
                and audit["evaluation_sha256"] == sha256(result_path),
                "Previous measured phase and independent metric audit must pass first")
    manifest_path = args.generation_dir/(args.phase+".json")
    audit_path = args.generation_dir/(args.phase+"_audit.json")
    manifest, audit = read(manifest_path), read(audit_path)
    require(manifest["status"] in ("passed_rollout_steering_pilot", "passed_rollout_generation")
            and audit["status"] == "passed_rollout_generation_audit" and audit["manifest_sha256"] == sha256(manifest_path),
            "Independent generation completion audit must pass before video scoring")
    records = manifest["records"]
    validate_pairs(records)
    require({row["split"] for row in records} == {args.phase}, "Measurement phase differs")
    return config, manifest, manifest_path, audit_path


def ordering(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    require(x.shape == y.shape == (5,) and np.isfinite(x).all() and np.isfinite(y).all()
            and np.all(np.diff(x) >= 0), "Five ordered finite dose/response pairs required")
    distinct = np.diff(x) > 0
    duplicate_consistent = all(np.ptp(y[x==value]) <= 1e-6 for value in np.unique(x))
    if not distinct.any():
        return {"spearman": None, "slope": None, "positive_adjacent_fraction": None,
                "nondecreasing_with_nonzero_span": False, "distinct_effective_doses": 1,
                "duplicate_effective_responses_consistent": duplicate_consistent}
    rx, ry = rankdata(x), rankdata(y)
    correlation = None if np.ptp(ry) == 0 else float(np.corrcoef(rx, ry)[0,1])
    slope = float(np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean()))
    delta = np.diff(y)[distinct]
    return {"spearman": correlation, "slope": slope, "positive_adjacent_fraction": float(np.mean(delta>0)),
            "nondecreasing_with_nonzero_span": bool(duplicate_consistent and np.all(delta >= -1e-6) and y[-1]-y[0]>1e-3),
            "duplicate_effective_responses_consistent": duplicate_consistent,
            "distinct_effective_doses": int(len(np.unique(x)))}


def interval(values, indices):
    values = np.asarray(values, dtype=float)
    means = values[indices].mean(axis=1)
    return {"mean": float(values.mean()), "match_bootstrap_ci95": np.quantile(means,[.025,.975]).tolist()}


def summarize(records, prediction, y_scale, config):
    lookup = {row["record_id"]:i for i,row in enumerate(records)}
    ball = absolute_ball(prediction)
    per_pair = []
    for baseline in records:
        if baseline["record_id"] != baseline["baseline_record_id"]:
            continue
        b = lookup[baseline["record_id"]]
        for path in PATHS:
            rows = [row for row in records if row["baseline_record_id"] == baseline["record_id"] and row["intervention_type"] == path]
            require(len(rows) == 4 and {row["dose"] for row in rows} == set(DOSES)-{0.}, "Complete frozen dose grid required")
            rows = sorted(rows+[baseline],key=lambda row:row["dose"])
            ids = [lookup[row["record_id"]] for row in rows]
            effective = np.array([row["height"]["effective_dose"] for row in rows])
            require(effective[2] == 0, "Shared baseline must have zero effective dose")
            shifts = prediction[ids]-prediction[b]
            collateral = np.sqrt(np.mean(np.square((shifts[[0,4],0]/y_scale)[:,:,[i for i in range(12) if i!=8]]),axis=(0,2)))
            target = ball[ids,0,:,2]-ball[b,0,:,2]
            across = ball[ids,:,:,2].mean(axis=1)-ball[b,:,:,2].mean(axis=0)
            per_pair.append({"match_id":baseline["match_id"], "seed":baseline["seed"], "path":path,
                "effective_doses":effective.tolist(), "view0_height_changes":target.tolist(),
                "allviews_mean_height_changes":across.tolist(), "view0_extreme_contrast":(target[-1]-target[0]).tolist(),
                "allviews_extreme_contrast":(across[-1]-across[0]).tolist(), "view0_nontarget_role_rms":collateral.tolist(),
                "ordering_by_time":[ordering(effective,target[:,t]) for t in range(4)],
                "any_clipped_dose":any(row["height"]["target_clipped"] for row in rows),
                "decoded_context_changed":any(not row["decoded_context_bitwise_equal_to_baseline"] for row in rows)})
    matches=sorted({row["match_id"] for row in per_pair})
    indices=np.random.default_rng(config["bootstrap_seed"]).integers(len(matches),size=(config["bootstrap_draws"],len(matches)))
    summary=[]
    for path in PATHS:
        paired={match:[row for row in per_pair if row["match_id"]==match and row["path"]==path] for match in matches}
        require(all(len(rows)==2 for rows in paired.values()), "Both paired seeds required per match/path")
        for t in range(4):
            item={"path":path,"time_index":t,"seconds_after_context":config["seconds_after_context"][t],"matches":len(matches)}
            for key in ("view0_extreme_contrast","allviews_extreme_contrast","view0_nontarget_role_rms"):
                values=[np.mean([row[key][t] for row in paired[match]]) for match in matches]
                item[key]=interval(values,indices)
                if key=="view0_extreme_contrast": item["positive_contrast_matches"]=int(np.sum(np.array(values)>0))
            item["view0_mean_height_by_requested_dose"] = np.mean([np.mean([row["view0_height_changes"] for row in paired[m]],axis=0) for m in matches],axis=0)[:,t].tolist()
            item["nondecreasing_nonzero_pair_fraction"] = float(np.mean([row["ordering_by_time"][t]["nondecreasing_with_nonzero_span"] for row in per_pair if row["path"]==path]))
            ranks=[row["ordering_by_time"][t]["spearman"] for row in per_pair if row["path"]==path]
            finite_match_means=[]
            for match in matches:
                finite=[row["ordering_by_time"][t]["spearman"] for row in paired[match]
                        if row["ordering_by_time"][t]["spearman"] is not None]
                if finite: finite_match_means.append(float(np.mean(finite)))
            item["mean_match_spearman"] = float(np.mean(finite_match_means)) if finite_match_means else None
            item["eligible_rank_matches"] = len(finite_match_means)
            item["undefined_rank_matches"] = len(matches)-len(finite_match_means)
            item["undefined_rank_pairs"] = sum(value is None for value in ranks)
            item["inconsistent_duplicate_dose_pairs"] = sum(not row["ordering_by_time"][t]["duplicate_effective_responses_consistent"]
                                                           for row in per_pair if row["path"]==path)
            for control in ("random_norm", "wrong_variable_ball_x"):
                differences=[]
                for match in matches:
                    active=np.mean([row["view0_extreme_contrast"][t] for row in paired[match]])
                    controls=[row for row in per_pair if row["match_id"]==match and row["path"]==control]
                    differences.append(active-np.mean([row["view0_extreme_contrast"][t] for row in controls]))
                item["view0_contrast_minus_"+control] = interval(differences,indices)
            summary.append(item)
    return {"per_pair":per_pair,"summary":summary,"matches":len(matches),"paired_seeds_per_match":2}


def evaluate(args):
    config,manifest,manifest_path,audit_path=verify(args)
    output=args.output_dir/(args.phase+".json")
    require(not output.exists(),"Completed measurements are immutable")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,"Expose one explicitly assigned GPU")
    evaluator=FrozenRolloutEvaluator(ROOT,device="cuda")
    records=manifest["records"]; predictions=[]; started=time.monotonic(); repeat_passed=False
    config_hash=sha256(args.protocol); source_hash=sha256(manifest_path)
    for i,row in enumerate(records):
        path=Path(row["generated_artifact_path"])
        require(sha256(path)==row["generated_sha256"],"Generated video artifact changed")
        cache=args.cache_dir/args.phase/(row["record_id"]+".npz")
        if cache.exists():
            with np.load(cache,allow_pickle=False) as saved:
                require(str(saved["generated_sha256"])==row["generated_sha256"] and str(saved["protocol_sha256"])==config_hash,
                        "Cached frozen evaluation differs")
                values=saved["role_predictions"]
        else:
            with np.load(path,allow_pickle=False) as saved: frames=saved["frames"]
            values=evaluator.predict(frames)["role_predictions"]
            if args.phase=="pilot" and not repeat_passed:
                require(np.array_equal(values,evaluator.predict(frames)["role_predictions"]),"Repeated frozen video evaluation changed")
                repeat_passed=True
            cache.parent.mkdir(parents=True,exist_ok=True)
            with cache.open("xb") as stream:
                np.savez_compressed(stream,role_predictions=values,generated_sha256=row["generated_sha256"],protocol_sha256=config_hash)
        require(values.shape==(4,4,12) and np.isfinite(values).all(),"Incomplete/nonfinite generated estimates")
        predictions.append(values)
        if (i+1)%20==0 or i+1==len(records):
            print(json.dumps({"phase":args.phase,"scored":i+1,"total":len(records),"elapsed_seconds":round(time.monotonic()-started,2)}),flush=True)
    predictions=np.stack(predictions)
    if args.phase=="pilot" and not repeat_passed:
        with np.load(records[0]["generated_artifact_path"],allow_pickle=False) as saved:frames=saved["frames"]
        require(np.array_equal(predictions[0],evaluator.predict(frames)["role_predictions"]),"Cached pilot does not replay")
        repeat_passed=True
    y_scale=evaluator.tail.head.y_scale.detach().float().cpu().numpy().astype(float)
    artifact=args.output_dir/(args.phase+"_predictions.npz")
    artifact.parent.mkdir(parents=True,exist_ok=True)
    with artifact.open("xb") as stream:
        np.savez_compressed(stream,record_ids=np.array([row["record_id"] for row in records]),role_predictions=predictions,
                            absolute_ball_predictions=absolute_ball(predictions),role_target_scale=y_scale)
    result={"status":"passed_frozen_generated_video_evaluation","phase":args.phase,"records":len(records),
        "protocol_sha256":config_hash,"generation_manifest_sha256":source_hash,"generation_audit_sha256":sha256(audit_path),
        "prediction_artifact":{"path":str(artifact.resolve()),"sha256":sha256(artifact)},"evaluator":evaluator.provenance,
        "generated_video_accuracy_established":False,"physical_control_established":False,"elapsed_seconds":time.monotonic()-started,
        "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(),"any_context_pixels_changed":not manifest["all_context_pixels_unchanged"]}
    if args.phase=="pilot": result.update(engineering_only=True,repeated_predictions_bitwise_equal=repeat_passed)
    else: result.update(summarize(records,predictions,y_scale,config))
    verify(args)
    write(output,result)
    print(f"Completed frozen {args.phase} video measurements; independent metric audit remains required.",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("register","pilot","selection","confirmation"),required=True)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/generated_evaluation_v1.json")
    parser.add_argument("--generation-dir",type=Path,default=ROOT/"results/rollout_steering_v1")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results/generated_evaluation_v1")
    parser.add_argument("--cache-dir",type=Path,default=Path("/data2/ishaangp/mira-interp/generated_evaluation_v1"))
    args=parser.parse_args()
    register(args) if args.phase=="register" else evaluate(args)


if __name__=="__main__":main()
