"""Strict pinned loading for the trained Alakazam MIRA Mini 4P checkpoint.

This is the validated pretrained smoke loading path, reusable by observation
capture. No checkpoint key is dropped, synthesized, or retained at random init.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MIRA_REV = "3d739ec2d31daf83559d33eb01727cea48fe90f7"
DINO_REV = "6876159a11b4df116f30f667f8c9888617df0751"
MODEL_REV = "d58ee2f9bca27289554c1e652943dc0e539e8971"
CONFIG_HASHES = {
    "world_model_config.yaml": "d6c49a8e92a28c1c9b3bc9d87950487095686dcee5d55c8bd029839da352fa87",
    "codec/codec_config.yaml": "c7e48bf87756e08b8f048c68cfb722365b2d53f27760c6c29dbc88d5154d5c24",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(directory: Path) -> str:
    revision = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.run(["git", "-C", str(directory), "diff", "--quiet", "HEAD", "--"]).returncode:
        raise RuntimeError(f"Tracked source changes at {directory}")
    untracked = subprocess.check_output(["git", "-C", str(directory), "ls-files", "--others", "--exclude-standard", "-z"]).decode().split("\0")
    source_files = [name for name in untracked if name and "__pycache__" not in Path(name).parts and Path(name).suffix in {".py", ".pyi", ".so"}]
    if source_files:
        raise RuntimeError(f"Untracked source files at {directory}: {source_files}")
    return revision


def load_pretrained_four_player(assets: Path, *, root: Path = ROOT, report: dict | None = None,
                               progress=None, verify_context: bool = False):
    """Return frozen/eval CPU model and loading provenance; caller chooses GPU.

    Model and codec checksums are revalidated once per worker. The caller can
    provide a report dictionary and progress callback for durable stage updates.
    """
    import torch
    from omegaconf import OmegaConf
    from mira.codec.codec_model import VideoCodec, REMOVED_CONFIG_FIELDS
    from mira.codec.config import VideoCodecConfig
    from mira.ml.config_loading import drop_removed_fields, strip_hydra_targets
    from mira.world_model.latent_world_model import _config_dict_from_yaml
    from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModelConfig
    from mira_interp.pretrained import AlakazamFourPlayerWorldModel

    assets = Path(assets).resolve()
    report = report if report is not None else {}
    progress = progress or (lambda stage: None)
    for folder, expected in [(root / "external/mira", MIRA_REV), (root / "external/dinov3", DINO_REV)]:
        if git_revision(folder) != expected:
            raise RuntimeError(f"Source revision mismatch at {folder}")
    report["source_revisions"] = {"mira": MIRA_REV, "dinov3": DINO_REV}
    report["model_repo"] = "alakazamworld/mira-mini-4p"
    report["model_revision"] = MODEL_REV
    report["loader_sha256"] = sha256(Path(__file__))
    report["adapter_sha256"] = sha256(root / "src/mira_interp/pretrained.py")
    report["torch_version"] = torch.__version__
    report["compatibility_adapter"] = "AlakazamFourPlayerWorldModel: project each player then average"
    manifest = json.loads((root / "results/model_access_audit.json").read_text())
    entry = next(x for x in manifest["third_party_pretrained_alternatives"] if x["repo_id"] == report["model_repo"])
    if entry["revision"] != MODEL_REV:
        raise ValueError("Audit model revision differs from loader pin")
    needed = ["checkpoint-90000/checkpoint.pth", "codec/checkpoint-125000/checkpoint.pth"]
    if verify_context:
        needed.append("context/default.npz")
    report["verified_assets"] = []
    for name in needed:
        spec = next(x for x in entry["files"] if x["name"] == name)
        path = assets / name
        if path.stat().st_size != spec["size_bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"Weight/input hash mismatch: {name}")
        report["verified_assets"].append({"name": name, "bytes": spec["size_bytes"], "sha256": spec["sha256"]})
    for name, expected in CONFIG_HASHES.items():
        if sha256(assets / name) != expected:
            raise ValueError(f"Original publisher config modified: {name}")
    report["original_config_sha256"] = dict(CONFIG_HASHES)
    codec_raw = OmegaConf.load(assets / "codec/codec_config.yaml")
    raw_codec = drop_removed_fields(strip_hydra_targets(OmegaConf.to_container(codec_raw.model.architecture.config, resolve=True)), REMOVED_CONFIG_FIELDS)
    raw_codec["encoder"]["compile_dino"] = False
    raw_codec["decoder"]["activation_checkpointing"] = False
    raw_wm = _config_dict_from_yaml(OmegaConf.load(assets / "world_model_config.yaml").model.architecture.config)
    raw_wm["wm_config"]["codec_checkpoint"] = str(assets / needed[1])
    raw_wm["wm_config"]["activation_checkpointing"] = False
    report["runtime_overrides"] = {"codec_checkpoint": "local verified path", "compile_dino": False, "activation_checkpointing": False}
    progress("load_codec_weights_safely")
    ckpt = torch.load(assets / needed[1], map_location="cpu", weights_only=True, mmap=True)
    hub_load = torch.hub.load

    def pinned_dino_factory(**kwargs):
        if kwargs["repo_or_dir"] != "facebookresearch/dinov3" or kwargs.get("pretrained") is not False:
            raise ValueError("Expected source-only DINO construction")
        return hub_load(str(root / "external/dinov3"), kwargs["model"], source="local", pretrained=False)

    with patch.object(torch.hub, "load", side_effect=pinned_dino_factory):
        codec = VideoCodec(VideoCodecConfig.model_validate(raw_codec), require_dino_weights=False)
    torch.nn.Module.load_state_dict(codec, ckpt["state_dict"], strict=True, assign=True)
    codec.info_from_checkpoint = {k: v for k, v in ckpt.items() if k != "state_dict"}
    report["codec_strict_load"] = {"passed": True, "state_tensors": len(ckpt["state_dict"])}
    del ckpt
    progress("construct_and_strict_load_world_model")
    with patch.object(VideoCodec, "load_from_checkpoint", return_value=codec):
        model = AlakazamFourPlayerWorldModel(MultiWrapperWorldModelConfig.model_validate(raw_wm))
    ckpt = torch.load(assets / needed[0], map_location="cpu", weights_only=True, mmap=True)
    state, expected = ckpt["state_dict"], model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    mismatch = sorted(k for k in set(expected) & set(state) if expected[k].shape != state[k].shape)
    report["world_model_strict_load"] = {"passed": False, "missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatch": mismatch, "state_tensors": len(state)}
    if missing or unexpected or mismatch:
        raise RuntimeError("Checkpoint architecture mismatch; no random fallback permitted")
    torch.nn.Module.load_state_dict(model, state, strict=True, assign=True)
    report["world_model_strict_load"]["passed"] = True
    report["parameter_count_including_codec"] = sum(p.numel() for p in model.parameters())
    report["codec_parameter_count"] = sum(p.numel() for p in codec.parameters())
    model.eval().requires_grad_(False)
    return model, report
