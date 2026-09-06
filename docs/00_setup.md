# Step 0: workspace and dependencies

Created `/ccn2/u/ishaangp/mira-interp` as an isolated checkout of
`https://github.com/guptaishaan/mira-interp.git`. The repository was initially empty.
No PSI experiment files, models or processes were changed.

The execution node is `node14-ccn2cluster.stanford.edu`. Physical GPUs 6 and 7 are
NVIDIA A40s with 46,068 MiB each. GPUs 0–5 belong to another user's active workload
and are excluded. `CUDA_VISIBLE_DEVICES=6` or `=7` maps that physical device to
logical `cuda:0`; a saved report records both identities. Check occupancy immediately
before each GPU command. Smoke results establish measured small-run feasibility,
not capacity for the complete study or full-length attribution gradients.

NFS initially had about 40 GB free. Public model weights are therefore stored at
`/data2/ishaangp/mira-interp/assets`, with a 25 GiB reserve checked before download.
The node-local asset directory occupies about 12 GB; its files are recoverable from
pinned public URLs. All authored code and compact reports remain in the NFS repository.

## Reproduce this environment

```bash
cd /ccn2/u/ishaangp/mira-interp
bash scripts/bootstrap.sh
export PYTHONPATH="$PWD/src:$PWD/external/mira/src"
export OMP_NUM_THREADS=4
.venv/bin/python -m pytest -q
.venv/bin/python scripts/download_model.py --destination /data2/ishaangp/mira-interp/assets
```

`bootstrap.sh` reuses the existing `ccwm` Python through a project-only virtual
environment with system-site packages. It pins MIRA and DINOv3 source checkouts,
installs Pydantic 2.11.9 locally, and installs this project's package without upgrading
the shared environment. On another machine, provide `MIRA_BASE_PYTHON` pointing to
a Python environment with the packages in `configs/tested_versions.txt`.
That file records observed versions; it is not a portable CUDA dependency lock.

The measured environment is Python 3.10.14 and PyTorch 2.9.0+cu126. Upstream's
recommended compiled setup is a different pinned combination; this project uses
**eager execution** and records the tests that actually passed. No `torch.compile`
speed or stability claim is made. Full video decoding still needs separate validation
of a compatible video backend on real Rocket Science shards.

Only local packages changed. An inherited Gradio/Tomlkit dependency conflict is
outside this project; Gradio is not used by these scripts. Do not upgrade unrelated
shared packages to silence that warning.

## Authenticated data and publishing

Rocket Science access needs the user's acceptance on the dataset page and a
Hugging Face login on the node. Credentials belong in the private home directory,
never in this repository, command arguments, reports, or Git. The access report saves
only token-presence status and sanitized HTTP/error types.

```bash
.venv/bin/python -c 'from huggingface_hub import login; login()'
.venv/bin/python scripts/prepare_data.py
```

The installed GitHub connector could read repository metadata but rejected a contents
write with HTTP 403. GitHub CLI authentication is the publishing fallback. The final
execution status records whether an authenticated push was actually completed.

## Evidence

- `results/environment.json`: versions, GPU identities and starting storage state.
- `results/model_download.json`: each downloaded file's byte size and SHA-256;
  large-file hashes checked against the publisher's metadata.
- `configs/sources.json`: pinned upstream source, model, dataset and seed.
- `docs/model_access_audit.md`: official availability versus the independent reproduction.
