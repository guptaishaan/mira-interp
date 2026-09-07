#!/usr/bin/env bash
# Detached launcher: retain logs plus an explicit exit record for each stage.
set -u
cd /ccn2/u/ishaangp/mira-interp
videomae_stage="${1:?stage required}"
case "$videomae_stage" in pilot|capture|train) ;; *) exit 2 ;; esac
export NUMPY_MADVISE_HUGEPAGE=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
videomae_python=/ccn2/u/ishaangp/miniconda3/envs/ccwm/bin/python
"$videomae_python" scripts/finetune_video_evaluator.py "$videomae_stage"
videomae_exit=$?
"$videomae_python" - "$videomae_stage" "$videomae_exit" <<'PY'
import datetime,json,pathlib,sys
stage,code=sys.argv[1],int(sys.argv[2])
pathlib.Path(f'results/videomae_finetune_{stage}_exit.json').write_text(json.dumps({'stage':stage,'exit_code':code,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
exit "$videomae_exit"
