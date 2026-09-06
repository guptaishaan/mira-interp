#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Reuse an explicitly selected, CUDA-capable Python; all changes stay in .venv.
BASE_PYTHON="${MIRA_BASE_PYTHON:-/ccn2/u/ishaangp/miniconda3/envs/ccwm/bin/python}"
"$BASE_PYTHON" -c 'import torch; print("Base PyTorch:", torch.__version__)'
if [[ ! -x .venv/bin/python ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages .venv
fi
mkdir -p external
if [[ ! -d external/mira/.git ]]; then
    git clone https://github.com/mira-wm/mira.git external/mira
    git -C external/mira checkout 3d739ec2d31daf83559d33eb01727cea48fe90f7
fi
[[ "$(git -C external/mira rev-parse HEAD)" == 3d739ec2d31daf83559d33eb01727cea48fe90f7 ]] || {
    echo 'Unexpected MIRA revision; preserve the checkout and inspect before proceeding.' >&2
    exit 1
}
if [[ ! -d external/dinov3/.git ]]; then
    git clone https://github.com/facebookresearch/dinov3.git external/dinov3
    git -C external/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
fi
[[ "$(git -C external/dinov3 rev-parse HEAD)" == 6876159a11b4df116f30f667f8c9888617df0751 ]] || {
    echo 'Unexpected DINOv3 revision; preserve the checkout and inspect before proceeding.' >&2
    exit 1
}
.venv/bin/python - <<'PY'
import subprocess
from pathlib import Path
for repo in ['external/mira', 'external/dinov3']:
    subprocess.run(['git', '-C', repo, 'diff', '--quiet'], check=True)
    subprocess.run(['git', '-C', repo, 'diff', '--cached', '--quiet'], check=True)
    untracked = subprocess.check_output(['git', '-C', repo, 'ls-files', '--others', '--exclude-standard'], text=True).splitlines()
    unexpected = [p for p in untracked if '__pycache__' not in Path(p).parts]
    if unexpected:
        raise RuntimeError(f'Untracked source files in {repo}: {unexpected}')
PY
.venv/bin/python -m pip install --no-cache-dir 'pydantic==2.11.9'
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
PYTHONPATH=src:external/mira/src .venv/bin/python -c 'import mira, mira_interp, torch; print("Imports passed", torch.__version__)'
