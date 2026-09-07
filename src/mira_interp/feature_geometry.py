"""Development-only geometry and custom sparse dictionaries; no causal claims."""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from torch import nn

from mira_interp.probes import match_weights, ridge_path


def validate_development(data, *, dimension=2048):
    required = ('X', 'velocity', 'match_ids', 'split', 'clip_ids', 'entity_ids',
                'view_index', 'frame_index', 'timestamps')
    if any(key not in data for key in required):
        raise ValueError(f"Required development arrays: {required}")
    n = len(data['X'])
    if n == 0 or data['X'].shape != (n, dimension) or data['velocity'].shape != (n, 3):
        raise ValueError("Require nonempty full-dimensional X and selected-entity velocity[N,3]")
    for key in required[2:]:
        if np.asarray(data[key]).shape != (n,):
            raise ValueError(f"{key} must have one entry per row")
    for key in ('X', 'velocity', 'timestamps', 'view_index', 'frame_index'):
        if not np.issubdtype(data[key].dtype, np.number) or not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite or nonnumeric {key}")
    for key in ('view_index', 'frame_index'):
        if not np.issubdtype(data[key].dtype, np.integer) or (data[key] < 0).any():
            raise ValueError(f"{key} must contain nonnegative integers")
    ids, roles = data['match_ids'].astype(str), data['split'].astype(str)
    if set(roles) != {'discovery', 'selection'}:
        raise ValueError("Development input must contain only discovery and selection; no confirmation")
    for key in ('match_ids', 'clip_ids', 'entity_ids'):
        if any(not value.strip() for value in data[key].astype(str)):
            raise ValueError(f"Empty {key}")
    if any(len(set(roles[ids == group])) != 1 for group in np.unique(ids)):
        raise ValueError("A match crosses split boundaries")
    counts = {role: len(np.unique(ids[roles == role])) for role in ('discovery', 'selection')}
    if min(counts.values()) < 2:
        raise ValueError("At least two independent matches are required per development role")
    keys = list(zip(ids, data['clip_ids'].astype(str), data['entity_ids'].astype(str),
                    data['view_index'], data['frame_index']))
    if len(set(keys)) != n:
        raise ValueError("Duplicate entity/view/frame observation")
    return counts


def geometry_report(x, values, match_ids, roles, *, heldout_interval, circular=False, alpha=0.01):
    """Fit physical value -> full activation; hold out values AND matches.

    Alpha and interval are supplied prospectively, never selected on evaluation.
    Circular inputs are angles in [-pi,pi); first/second harmonics replace powers.
    """
    x, u = np.asarray(x, float), np.asarray(values, float)
    ids, roles = np.asarray(match_ids).astype(str), np.asarray(roles).astype(str)
    if x.ndim != 2 or u.shape != (len(x),) or ids.shape != u.shape or roles.shape != u.shape:
        raise ValueError("Incompatible geometry arrays")
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError("Geometry arrays must be finite")
    if set(roles) != {'discovery', 'selection'} or any(len(set(roles[ids == g])) != 1 for g in np.unique(ids)):
        raise ValueError("Geometry requires disjoint discovery/selection matches")
    low, high = map(float, heldout_interval)
    discovery = roles == 'discovery'
    if not np.isfinite([low, high]).all() or not u[discovery].min() < low < high < u[discovery].max():
        raise ValueError("Held-value interval must be strictly interior to discovery support")
    if circular and ((u < -np.pi).any() or (u >= np.pi).any()):
        raise ValueError("Angles must use [-pi,pi)")
    held = (u >= low) & (u <= high)
    train, evaluate = discovery & ~held, (roles == 'selection') & held
    if min(len(np.unique(ids[train])), len(np.unique(ids[evaluate]))) < 2:
        raise ValueError("Held-value comparison needs two matches per role with eligible rows")
    w = match_weights(ids[train])
    mean = np.sum(w * u[train]); scale = np.sqrt(np.sum(w * (u[train] - mean) ** 2))
    if scale <= 1e-12:
        raise ValueError("No training physical-value variation")
    z = (u - mean) / scale
    features = {'affine': z[:, None], 'quadratic': np.column_stack((z, z*z))}
    if circular:
        first = np.column_stack((np.sin(u), np.cos(u)))
        features = {'periodic_first': first, 'periodic_second': np.column_stack((first, np.sin(2*u), np.cos(2*u)))}
    xmean = np.sum(w[:, None] * x[train], axis=0)
    variance = float(np.sum(w * np.mean((x[train] - xmean) ** 2, axis=1)))
    if variance <= 1e-15:
        raise ValueError("No training activation variation")
    reports = {}
    baseline_errors = np.mean((x[evaluate] - xmean) ** 2, axis=1)
    baseline_per_match = {g: float(baseline_errors[ids[evaluate] == g].mean())
                          for g in np.unique(ids[evaluate])}
    for name, basis in features.items():
        model = ridge_path(basis[train], x[train], w, alphas=(alpha,))[0]
        pred = model.predict(basis[evaluate])
        row_mse = np.mean((pred - x[evaluate]) ** 2, axis=1)
        per_match = {g: float(row_mse[ids[evaluate] == g].mean()) for g in np.unique(ids[evaluate])}
        error = float(np.mean(list(per_match.values())))
        reports[name] = {'raw_activation_mse': error, 'mse_over_discovery_variance': error / variance,
                         'per_match_raw_mse': per_match, 'model_sha256': model.fingerprint(),
                         'alpha': float(alpha), 'basis_dimensions': basis.shape[1]}
    names = list(reports)
    gain = {g: reports[names[0]]['per_match_raw_mse'][g] - reports[names[1]]['per_match_raw_mse'][g]
            for g in reports[names[0]]['per_match_raw_mse']}
    gain_values = np.asarray(list(gain.values()))
    rng = np.random.default_rng(20260907)
    bootstrap = gain_values[rng.integers(0, len(gain_values), (1000, len(gain_values)))].mean(1)
    return {'claim': 'conditional observational geometry; no manifold or causal conclusion',
            'heldout_interval': [low, high], 'heldout_bounds_inclusive': True, 'circular': circular,
            'fit_rows': int(train.sum()), 'evaluation_rows': int(evaluate.sum()),
            'excluded_discovery_held_value_rows': int((discovery & held).sum()),
            'fit_match_ids': sorted(set(ids[train])), 'evaluation_match_ids': sorted(set(ids[evaluate])),
            'physical_mean_fit_only': float(mean), 'physical_scale_fit_only': float(scale),
            'activation_metric': 'mean squared error in original selected-descriptor coordinates, without further projection',
            'models': reports, 'paired_mse_gain_complex_over_simple': float(np.mean(list(gain.values()))),
            'paired_gain_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'mean_baseline': {'raw_activation_mse': float(np.mean(list(baseline_per_match.values()))),
                              'per_match_raw_mse': baseline_per_match},
            'uncertainty': '1000 whole-match bootstrap draws, descriptive and conditional on this fixed comparison',
            'per_match_paired_gain': gain, 'hyperparameters_selected_on_evaluation': False}


def temporal_edges(data, *, role='discovery', expected_dt=0.1, tolerance=0.01):
    """Adjacent source indices within the SAME match, clip, entity, and view.

    frame_index is the descriptor's consecutive latent index, not source FPS index.
    Caller must separately establish that entity_ids denote verified correspondence.
    """
    if expected_dt <= 0 or tolerance < 0 or tolerance >= expected_dt or not np.isfinite([expected_dt, tolerance]).all():
        raise ValueError("Invalid temporal sampling tolerance")
    groups = {}
    for i in np.flatnonzero(data['split'].astype(str) == role):
        key = tuple(str(data[k][i]) for k in ('match_ids', 'clip_ids', 'entity_ids', 'view_index'))
        groups.setdefault(key, []).append(i)
    edges = []
    for rows in groups.values():
        rows.sort(key=lambda i: int(data['frame_index'][i]))
        for a, b in zip(rows, rows[1:]):
            if data['frame_index'][b] - data['frame_index'][a] == 1:
                dt = float(data['timestamps'][b] - data['timestamps'][a])
                if abs(dt - expected_dt) <= tolerance:
                    edges.append((a, b))
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


class TopKDictionary(nn.Module):
    """Small custom dictionary, NOT an official Goodfire BSF reproduction.

    All variants use untied maps, encoder bias, and unit decoder-row norms.
    Block selection uses signed preactivations with highest group L2 norms.
    """
    def __init__(self, dimension, *, width=512, active=32, kind='signed', group_size=8):
        super().__init__()
        if kind not in ('relu', 'signed', 'block') or not 0 < active < width or dimension <= 0:
            raise ValueError("Invalid dictionary configuration")
        block = group_size if kind == 'block' else 1
        if block <= 0 or width % block or active % block:
            raise ValueError("Width and active-coordinate budget must be divisible by block size")
        self.kind, self.width, self.active, self.block = kind, width, active, block
        self.encoder = nn.Linear(dimension, width)
        self.decoder = nn.Parameter(torch.randn(width, dimension))
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder)
            self.encoder.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def encode(self, x):
        a = self.encoder(x)
        if self.kind == 'relu':
            a = a.relu()
        groups = a.reshape(-1, self.width // self.block, self.block)
        score = groups.norm(dim=-1)
        chosen = score.topk(self.active // self.block, dim=1).indices
        mask = torch.zeros_like(score).scatter(1, chosen, 1)
        return (groups * mask[..., None]).reshape(-1, self.width)

    def support_surrogate(self, x, temperature=0.1):
        if self.kind != 'block' or temperature <= 0:
            raise ValueError("Soft block-support surrogate requires a block dictionary")
        score = self.encoder(x).reshape(-1, self.width // self.block, self.block).norm(dim=-1)
        ranked = score.topk(self.active // self.block + 1, dim=1).values
        threshold = ranked[:, -2:].mean(1, keepdim=True).detach()
        return torch.sigmoid((score - threshold) / temperature)

    def forward(self, x):
        code = self.encode(x)
        return code @ self.decoder, code


def train_dictionary(data, *, kind, width=512, active=32, group_size=8, steps=200,
                     batch_size=128, learning_rate=1e-3, seed=0, device='cpu',
                     temporal_weight=0., entity_mapping_verified=False, expected_dt=0.1,
                     temporal_control='adjacent', view_identity_verified=False):
    """Fixed-budget development training; selection data never updates weights."""
    x = np.asarray(data['X'], np.float32)
    validate_development(data, dimension=x.shape[1])
    if not 1 <= steps <= 2000 or not 1 <= batch_size <= 1024 or not 0 < learning_rate <= 0.1:
        raise ValueError("Training budget exceeds the bounded pilot limits")
    if not 1 <= width <= 8192 or not 1 <= active <= 128:
        raise ValueError("Dictionary budget is at most8192 coordinates and128 active coordinates")
    if temporal_weight < 0 or not np.isfinite(temporal_weight):
        raise ValueError("Temporal weight must be finite and nonnegative")
    if temporal_weight and (kind != 'block' or not (entity_mapping_verified or view_identity_verified)):
        raise ValueError("Temporal regularization needs verified entity or explicitly scoped view identity and block codes")
    if temporal_control not in ('adjacent', 'shuffled_within_match'):
        raise ValueError('Unknown temporal control')
    train = np.flatnonzero(data['split'].astype(str) == 'discovery')
    w = match_weights(data['match_ids'][train])
    mean = np.sum(x[train].astype(float) * w[:, None], axis=0)
    # One scalar preserves the descriptor's angles and relative distances.
    # Per-channel whitening would change the geometry being compared.
    rms = np.sqrt(np.sum(np.mean((x[train] - mean) ** 2, axis=1) * w))
    if rms < 1e-8:
        raise ValueError('No discovery activation variation')
    scale = np.full(x.shape[1], rms, dtype=np.float64)
    normalized = torch.as_tensor(((x - mean) / scale).astype(np.float32), device=device)
    torch.manual_seed(seed)
    model = TopKDictionary(x.shape[1], width=width, active=active, kind=kind, group_size=group_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    temporal_rng = np.random.default_rng(seed + 271828)
    edges = temporal_edges(data, expected_dt=expected_dt) if temporal_weight else np.empty((0, 2), int)
    if temporal_weight and len(edges) == 0:
        raise ValueError("No verified adjacent entity pairs are available")
    if temporal_weight and temporal_control == 'shuffled_within_match':
        # Preserve match, clip, entity and view. Only time correspondence changes.
        # Keep the historical option name, but report the narrower actual scope.
        edge_rng = np.random.default_rng(seed + 314159)
        for index, (a, b) in enumerate(edges.copy()):
            eligible = np.ones(len(train), bool)
            for name in ('match_ids', 'clip_ids', 'entity_ids', 'view_index'):
                eligible &= data[name][train] == data[name][a]
            candidates = train[eligible & (train != a) & (train != b)]
            if not len(candidates):
                raise ValueError('Temporal shuffle requires another time in the same tracked view')
            edges[index, 1] = edge_rng.choice(candidates)
    edge_weights = match_weights(data['match_ids'][edges[:, 0]]) if len(edges) else None
    started = time.monotonic(); losses = []
    for step in range(steps):
        rows = rng.choice(train, size=batch_size, replace=True, p=w)
        prediction, _ = model(normalized[rows])
        reconstruction = (prediction - normalized[rows]).square().mean()
        regularization = reconstruction.new_zeros(())
        if temporal_weight:
            links = edges[temporal_rng.choice(len(edges), size=max(1, batch_size // 4), replace=True, p=edge_weights)]
            regularization = (model.support_surrogate(normalized[links[:, 0]]) -
                              model.support_surrogate(normalized[links[:, 1]])).square().mean()
        loss = reconstruction + temporal_weight * regularization
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite dictionary training loss")
        optimizer.zero_grad(); loss.backward(); optimizer.step(); model.normalize_decoder()
        if step == 0 or (step + 1) % 100 == 0 or step == steps - 1:
            losses.append({'step': step + 1, 'reconstruction': float(reconstruction.detach()),
                           'temporal_surrogate': float(regularization.detach())})
    model.eval(); reports = {}
    for role in ('discovery', 'selection'):
        rows = np.flatnonzero(data['split'].astype(str) == role)
        errors, raw_errors, scalar_activity, block_activity = [], [], [], []
        scalar_seen = torch.zeros(width, dtype=torch.bool, device=device)
        block_seen = torch.zeros(width // model.block, dtype=torch.bool, device=device)
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                indices = rows[start:start + batch_size]
                prediction, code = model(normalized[indices])
                residual = (prediction - normalized[indices]).cpu().numpy()
                errors.extend(np.mean(residual ** 2, axis=1).tolist())
                raw_errors.extend(np.mean((residual * scale) ** 2, axis=1).tolist())
                scalar_activity.extend((code != 0).sum(1).cpu().tolist())
                block_activity.extend((code.reshape(-1, width // model.block, model.block).norm(dim=-1) > 0).sum(1).cpu().tolist())
                scalar_seen |= (code != 0).any(0)
                block_seen |= (code.reshape(-1, width // model.block, model.block).norm(dim=-1) > 0).any(0)
        weights = match_weights(data['match_ids'][rows])
        reports[role] = {name: float(np.dot(weights, values)) for name, values in
                         [('normalized_reconstruction_mse', errors), ('raw_reconstruction_mse', raw_errors),
                          ('active_scalar_mean', scalar_activity), ('active_block_mean', block_activity)]}
        reports[role]['match_count'] = len(np.unique(data['match_ids'][rows]))
        reports[role]['dead_scalar_fraction'] = float((~scalar_seen).float().mean())
        reports[role]['dead_block_fraction'] = float((~block_seen).float().mean())
        per_match = {str(match): float(np.mean(np.asarray(raw_errors)[data['match_ids'][rows] == match]))
                     for match in np.unique(data['match_ids'][rows])}
        reports[role]['per_match_raw_reconstruction_mse'] = per_match
        true_edges = temporal_edges(data, role=role, expected_dt=expected_dt)
        overlaps = []
        with torch.no_grad():
            for start in range(0, len(true_edges), batch_size):
                links = true_edges[start:start + batch_size]
                a = model.encode(normalized[links[:, 0]]).reshape(-1, width // model.block, model.block).norm(dim=-1) > 0
                b = model.encode(normalized[links[:, 1]]).reshape(-1, width // model.block, model.block).norm(dim=-1) > 0
                overlaps.extend(((a & b).sum(1) / (a | b).sum(1).clamp_min(1)).cpu().tolist())
        reports[role]['adjacent_support_jaccard'] = float(np.dot(match_weights(data['match_ids'][true_edges[:, 0]]), overlaps)) if overlaps else None
    report = {'kind': kind, 'group_size': model.block, 'latent_scalar_width': width,
              'input_dimension': x.shape[1], 'expansion': width / x.shape[1],
              'dictionary_regime': 'overcomplete' if width > x.shape[1] else 'undercomplete_or_square',
              'active_scalar_budget': active, 'trainable_parameters': sum(p.numel() for p in model.parameters()),
              'steps': steps, 'batch_size': batch_size, 'learning_rate': learning_rate, 'seed': seed,
              'temporal_weight': temporal_weight, 'temporal_edges': len(edges), 'expected_dt': expected_dt,
              'temporal_control': temporal_control,
              'temporal_shuffle_scope': 'same match, clip, annotation entity and view; exclude self and original next frame',
              'temporal_identity_scope': 'entity_identity' if entity_mapping_verified else ('view_identity_only' if view_identity_verified else 'not_verified'),
              'tracked_entity_claim': False,
              'temporal_penalty': 'squared difference of soft support surrogates; no coordinate smoothness penalty',
              'normalizer_fit': 'equal-match discovery mean and one global RMS; preserves Euclidean geometry', 'selection_used_for_updates': False,
              'official_bsf_replication': False, 'decoder_constraint': 'each row has unit L2 norm',
              'normalizer_sha256': hashlib.sha256(mean.tobytes() + scale.tobytes()).hexdigest(),
              'elapsed_seconds': time.monotonic() - started, 'loss_endpoints': losses, 'metrics': reports}
    return model.cpu(), {'mean': mean, 'scale': scale}, report
