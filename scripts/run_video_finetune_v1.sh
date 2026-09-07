#!/usr/bin/env bash
# Run this inside tmux; record completion even when Python reports a failure.
set -u
cd /ccn2/u/ishaangp/mira-interp
export NUMPY_MADVISE_HUGEPAGE=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
videomae_python=/ccn2/u/ishaangp/miniconda3/envs/ccwm/bin/python
"$videomae_python" scripts/train_video_evaluator_partial.py
videomae_exit=$?
"$videomae_python" - "$videomae_exit" <<'PY'
import datetime,json,pathlib,sys
pathlib.Path('results/videomae_finetune_v1_exit.json').write_text(json.dumps({'exit_code':int(sys.argv[1]),'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
exit "$videomae_exit"
