"""Frozen VideoMAE measurements on generated windows; these are estimates, not3D truth."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .development import ROLE_TARGET_NAMES
from .model_loading import sha256
from .video_evaluator import load_encoder, preprocess_frames
from .video_evaluator_finetune import VideoMAETail, frozen_prefix
from .video_evaluator_training import SpatialStateHead

END_FRAMES = (17, 19, 21, 23)
REQUIRED_RECORD_FIELDS = ('record_id','match_id','clip_id','split','seed','intervention_type','dose',
                          'generated_artifact_path','generated_sha256','baseline_record_id')


def window_indices():
    """Four16-frame windows, each ending at its corresponding generated tubelet."""
    return np.asarray([np.arange(end-15,end+1) for end in END_FRAMES])


def validate_generated_video(frames):
    frames = np.asarray(frames)
    if frames.shape != (4,24,3,288,512) or frames.dtype != np.float32:
        raise ValueError('Generated frames must be float32[4,24,3,288,512]')
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError('Generated frames must contain finite RGB values in0..1')
    return frames


def absolute_ball(role_predictions):
    values = np.asarray(role_predictions,dtype=np.float64)
    if values.shape[-1] != 12 or not np.isfinite(values).all():
        raise ValueError('Require finite ego6 and ball-minus-ego6 predictions')
    return values[...,:6]+values[...,6:]


def validate_pairs(records):
    """An explicit baseline must share match, clip, partition and seed."""
    lookup = {}; signatures = set(); match_roles = {}
    for row in records:
        if any(name not in row for name in REQUIRED_RECORD_FIELDS): raise ValueError('Missing generated-record contract fields')
        if row['record_id'] in lookup or not row['record_id']: raise ValueError('Duplicate/empty generated record identifier')
        if row['split'] not in ('pilot','selection','confirmation') or not np.isfinite(row['dose']): raise ValueError('Invalid evaluation partition or dose')
        if type(row['seed']) is not int: raise ValueError('Seeds must be explicit integers')
        if any(not isinstance(row[name],str) or not row[name] for name in ('match_id','clip_id','intervention_type','generated_artifact_path')):
            raise ValueError('Source and intervention identities must be nonempty strings')
        if len(row['generated_sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['generated_sha256']): raise ValueError('Missing generated-artifact checksum')
        signature = tuple(row[name] for name in ('match_id','clip_id','split','seed','intervention_type','dose'))
        if signature in signatures: raise ValueError('Duplicate intervention condition')
        signatures.add(signature); match_roles.setdefault(row['match_id'],set()).add(row['split'])
        lookup[row['record_id']] = row
    if any(len(roles)!=1 for roles in match_roles.values()): raise ValueError('A match crosses evaluation partitions')
    for row in records:
        if row['baseline_record_id'] not in lookup: raise ValueError('Explicit baseline record is missing')
        baseline = lookup[row['baseline_record_id']]
        if baseline['dose'] != 0 or baseline['baseline_record_id'] != baseline['record_id']:
            raise ValueError('Common-zero baseline must have zero dose and point to itself')
        if any(row[name] != baseline[name] for name in ('match_id','clip_id','split','seed')):
            raise ValueError('Intervention and baseline are not paired on identical source/seed')
    return lookup


def paired_changes(records, predictions):
    lookup = validate_pairs(records)
    if set(lookup) != set(predictions): raise ValueError('Predictions do not cover exactly the generated records')
    changes = {}
    for key,row in lookup.items():
        predicted = np.asarray(predictions[key],dtype=np.float64)
        baseline = np.asarray(predictions[row['baseline_record_id']],dtype=np.float64)
        if predicted.shape != (4,4,12) or baseline.shape != predicted.shape or not np.isfinite(predicted).all() or not np.isfinite(baseline).all():
            raise ValueError('Each prediction needs all4 views,4 generated endpoints,12 role coordinates')
        changes[key] = {'role_delta':predicted-baseline,'absolute_ball_delta':absolute_ball(predicted)-absolute_ball(baseline)}
    return changes


def recorded_reference_calibration(repo):
    """Empirical source-annotation error, explicitly not generated-video intervals."""
    repo = Path(repo); path = repo/'results/videomae_finetune_v1.json'; report = json.loads(path.read_text())
    artifact = Path(report['predictions']['path'])
    if report['status']!='passed_development_partial_finetune' or sha256(artifact)!=report['predictions']['sha256']:
        raise ValueError('Recorded reference predictions changed')
    result = {'source_report_sha256':sha256(path),'source_predictions_sha256':sha256(artifact),
        'partition':'development selection, also used for model/checkpoint choice',
        'generated_video_accuracy_established':False,'predictive_uncertainty_intervals_available':False}
    with np.load(artifact,allow_pickle=False) as saved:
        truth, ids = saved['y'],saved['match_ids']; groups,counts=np.unique(ids,return_counts=True)
        if len(groups)!=11 or set(counts)!={32}: raise ValueError('Recorded reference cohort changed')
        for domain in ('real','codec'):
            prediction=saved[domain+'_prediction']; error=prediction-truth; ball_error=absolute_ball(prediction)-absolute_ball(truth)
            per_match=np.stack([np.mean(np.abs(ball_error[ids==match]),axis=0) for match in groups])
            result[domain]={'role_mae':np.abs(error).mean(axis=0).tolist(),'role_rmse':np.sqrt((error**2).mean(axis=0)).tolist(),
                'absolute_ball_mae':np.abs(ball_error).mean(axis=0).tolist(),'absolute_ball_rmse':np.sqrt((ball_error**2).mean(axis=0)).tolist(),
                'absolute_ball_per_match_mae_min':per_match.min(axis=0).tolist(),'absolute_ball_per_match_mae_max':per_match.max(axis=0).tolist(),
                'matches':11,'view_rows':352}
    return result


class FrozenRolloutEvaluator:
    """Exact selected step500 pipeline, with no optimizer or adaptation interface."""
    def __init__(self, repo, *, assets='/data2/ishaangp/mira-interp/models/videomae-base', device='cpu'):
        repo = Path(repo); report_path = repo/'results/videomae_finetune_v1.json'
        report = json.loads(report_path.read_text()); audit = json.loads((repo/'results/videomae_finetune_v1_audit.json').read_text())
        if report['status'] != 'passed_development_partial_finetune' or report['selected_step'] != 500:
            raise ValueError('Require the frozen completed step500 evaluator')
        if audit['status'] != 'passed_development_video_evaluator_audit' or audit['analysis_sha256'] != sha256(report_path):
            raise ValueError('Independent evaluator audit does not bind this model')
        config_path = repo/'configs/videomae_partial_finetune_v1.json'; config = json.loads(config_path.read_text())
        if sha256(config_path) != report['registration_sha256']: raise ValueError('Evaluator registration changed')
        if any(sha256(repo/name) != digest for name,digest in config['code_sha256'].items()): raise ValueError('Verified evaluator implementation changed')
        checkpoint = Path(report['checkpoint']['path'])
        if sha256(checkpoint) != report['checkpoint']['sha256']: raise ValueError('Frozen evaluator checkpoint changed')
        state = torch.load(checkpoint,map_location='cpu',weights_only=True)
        head = SpatialStateHead(*[state['head.'+name].numpy() for name in ('x_mean','x_scale','y_mean','y_scale')])
        encoder, self.preprocess, loading = load_encoder(assets)
        tail = VideoMAETail(encoder,head); tail.load_state_dict(state,strict=True)
        self.device = torch.device(device); self.encoder = encoder.to(self.device).eval().requires_grad_(False)
        self.tail = tail.to(self.device).eval().requires_grad_(False)
        if any(p.dtype != torch.float32 for p in self.encoder.parameters()) or any(p.dtype != torch.float32 for p in self.tail.parameters()):
            raise ValueError('Evaluator checkpoint parameters must remain FP32')
        self.provenance = {'checkpoint_sha256':report['checkpoint']['sha256'],'analysis_sha256':sha256(report_path),
            'training_registration_sha256':sha256(config_path),'encoder':loading,'target_names':list(ROLE_TARGET_NAMES),
            'window_end_frames':list(END_FRAMES),'input_video_range':[0,1],
            'claim_limit':'Independent learned video estimates; no generated/intervened3D accuracy established'}

    def predict(self, frames):
        frames = validate_generated_video(frames)
        if self.device.type != 'cuda': raise ValueError('Real inference requires an explicitly assigned CUDA device to preserve calibrated BF16 arithmetic')
        if not torch.are_deterministic_algorithms_enabled(): raise ValueError('Enable deterministic algorithms before evaluator inference')
        output = []
        # Scope the attention backend; do not alter the caller\'s sampler globally.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda',dtype=torch.bfloat16):
            for end in END_FRAMES:
                video = preprocess_frames(frames[:,end-15:end+1]*255,self.preprocess,self.device)
                prefix = frozen_prefix(self.encoder,video)
                values = self.tail.predict(prefix).float().cpu().numpy()
                if values.shape != (4,12) or not np.isfinite(values).all(): raise ValueError('Generated-window evaluator prediction failed')
                output.append(values)
        roles = np.stack(output,axis=1).astype(np.float64)
        return {'role_predictions':roles,'absolute_ball_predictions':absolute_ball(roles),
            'window_end_frames':np.asarray(END_FRAMES),'seconds_after_context':(np.asarray(END_FRAMES)-15)/20,
            'role_target_names':np.asarray(ROLE_TARGET_NAMES),'absolute_ball_fields':np.asarray(['location.x','location.y','location.z','velocity.x','velocity.y','velocity.z'])}
