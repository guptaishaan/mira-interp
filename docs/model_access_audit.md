# Model access audit

Checked live on 2026-09-06. The official MIRA code is available at revision
`3d739ec2d31daf83559d33eb01727cea48fe90f7`; an accessible official pretrained world-model
or codec checkpoint was not found. A public, trained **MIRA Mini 1B reproduction by
Alakazam** provides a practical alternative for inference readiness. It must be
identified as that independent reproduction in every result.

## Official MIRA

| Check | Observation |
| --- | --- |
| [Official code and README](https://github.com/mira-wm/mira/tree/3d739ec2d31daf83559d33eb01727cea48fe90f7) | Training and evaluation examples require user-supplied checkpoint paths. The repository tree contains no `.pt`, `.pth`, `.safetensors`, `.ckpt`, or `.bin` weights. |
| [Release v1.0](https://github.com/mira-wm/mira/releases/tag/v1.0) | The GitHub API returns zero uploaded assets. The two automatically provided source archives are not pretrained weights. |
| [Checkpoint-release issue #1](https://github.com/mira-wm/mira/issues/1) | Open; no comments at audit time. Requests checkpoint publication. |
| [Kyutai MIRA collection](https://huggingface.co/collections/kyutai/mira-world-model) | Contains the paper and Rocket Science dataset, with no model repository. |
| [Kyutai public models API](https://huggingface.co/api/models?author=kyutai&limit=100) | No MIRA model among the returned models. The `mira-wm` author query returns none. |
| [Project demo](https://mira-wm.com/) | Browser application; no accessible checkpoint link was located. |

The official loader requires both world-model weights with
`world_model_config.yaml` and codec weights with `codec_config.yaml`. Its W&B
resolver retrieves a run's **local output directory**; it does not supply an
otherwise absent public checkpoint. A trained codec checkpoint already includes
its DINO parameters; separate DINOv3 weights are needed for codec training, not
for loading the trained codec. See the pinned source's
`src/mira/training/checkpoints.py` and `src/mira/codec/codec_model.py`.

Targeted local filename/directory checks found no MIRA checkpoint candidate.
This was not an exhaustive scan of unrelated projects or shared storage. No
credential contents were read. Absence in these checks does not prove that
private or unlisted official weights do not exist.

## Available independent reproduction

[Alakazam's release](https://alakazam.gg/mira-mini) links directly to its
[public model organization](https://huggingface.co/alakazamworld). The release
identifies these as independently trained models, without endorsement from the
original MIRA teams. The Hugging Face API reports all four repositories below as
public and ungated. Sizes and SHA-256 values come from live repository metadata;
the full file lists and immutable URLs are in
[`results/model_access_audit.json`](../results/model_access_audit.json).

| Model | Revision | Main weights | Codec |
| --- | --- | ---: | ---: |
| [MIRA Mini 1B](https://huggingface.co/alakazamworld/mira-mini) | `19d668ac39814e394ae4a8f698690f761facf437` | 8.37 GB | 3.60 GB |
| [MIRA Mini 4P 1B](https://huggingface.co/alakazamworld/mira-mini-4p) | `d58ee2f9bca27289554c1e652943dc0e539e8971` | 8.38 GB | 3.60 GB |
| [MIRA Mini PSD 1B](https://huggingface.co/alakazamworld/mira-mini-psd) | `ad0c8e3bec95900bb01ebacd7fddf80724a6d957` | 8.39 GB | 3.60 GB |
| [MIRA Mini 364M](https://huggingface.co/alakazamworld/mira-mini-364m) | `d4aba4a3e525b573689c78458d589e2c0df02f1c` | 3.00 GB | 1.52 GB |

The 1B bundles share identical codec weights (SHA-256
`0b56cc07c878ab231a2c1faea155f20b0d49dbc378e3b67e84b2c7026eec59f9`).
World-model/codec checkpoint HEAD requests returned HTTP 200 without login.
The model cards specify CC BY-NC-SA 4.0 for these weights; preserve attribution
and do not treat the upstream code's Apache license as the weights license.
Prefer links and pinned manifests over committing multi-gigabyte weights to Git.

## Public inputs and compatibility

The 1B context is 35,392,832 bytes; the 4P context is 141,569,792 bytes. Bounded
HTTP range reads of their ZIP central directories found exactly `frames.npy`
and `actions.npy` in each. **Neither provides physics labels or match IDs.**
These examples allow a trained inference smoke test but cannot support the
proposal's match-separated physical-state probe experiment.

The 4P config specifies 16 transformer layers, width 2048, and 288 x 512 video
per player. The codec uses 32 latent channels, a spatial reduction of 32, and
temporal reduction of 2. Bundle YAML files contain the publisher's old absolute
codec path: localize that path in a separate runtime config and retain the
original config/hash. Validate strict state-dict key/shape agreement before
running inference; similarity of configuration is not proof of compatibility.

The [public player repository](https://github.com/Alakazam-studios/alakazam-mira-mini/tree/617a585852445f3581f6b88aaf27ec4e745aa6b4)
is at `617a585852445f3581f6b88aaf27ec4e745aa6b4`. Its weight helper implements
the codec-path localization. Its CLI exposes `auto`, `1b`, and `364m`; the 4P
model-card CLI example is not supported by those declared argument choices.
Use the checkpoint's correct model class directly for an auditable smoke test.

## Gate and next action

Official-MIRA inference remains blocked on a trained checkpoint. The independent
MIRA Mini 4P bundle can pass a separate pretrained-readiness gate after download,
hash verification, strict loading, and a finite bounded forward/sample. Such a
pass establishes executable access only. Physical causal identification still
requires authorized Rocket Science video/actions/physics and match IDs. The
[dataset page](https://huggingface.co/datasets/kyutai/rocket-science) displays an
access-terms gate; public example contexts do not remove that requirement.

No checkpoint weights were downloaded and no GPU was used by this access audit.
