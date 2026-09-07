"""Synthetic mathematical/guard tests only; these are not research experiments."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mira_interp.feature_geometry import (
    TopKDictionary, geometry_report, temporal_edges, train_dictionary, validate_development,
)


def fixture():
    rng = np.random.default_rng(21)
    return {'X': rng.normal(size=(24, 6)).astype(np.float32),
            'velocity': rng.normal(size=(24, 3)), 'match_ids': np.repeat(['a', 'b', 'c', 'd'], 6),
            'split': np.repeat(['discovery', 'discovery', 'selection', 'selection'], 6),
            'clip_ids': np.repeat(['a0', 'b0', 'c0', 'd0'], 6), 'entity_ids': np.full(24, 'ball'),
            'view_index': np.zeros(24, dtype=int), 'frame_index': np.tile(np.arange(6), 4),
            'timestamps': np.tile(np.arange(6) * 0.1, 4)}


def test_held_value_geometry_recovers_quadratic_without_using_held_training_values():
    u = np.tile(np.linspace(-2, 2, 41), 4)
    ids = np.repeat(['a', 'b', 'c', 'd'], 41)
    roles = np.repeat(['discovery', 'discovery', 'selection', 'selection'], 41)
    x = np.column_stack((u, u*u, 3*u*u + 5))
    report = geometry_report(x, u, ids, roles, heldout_interval=(-.5, .5), alpha=1e-8)
    assert report['models']['quadratic']['raw_activation_mse'] < 1e-10
    assert report['models']['affine']['raw_activation_mse'] > 1
    changed = x.copy()
    changed[(roles == 'discovery') & (abs(u) <= .5)] += 1e6
    other = geometry_report(changed, u, ids, roles, heldout_interval=(-.5, .5), alpha=1e-8)
    assert report == other
    assert set(report['fit_match_ids']).isdisjoint(report['evaluation_match_ids'])
    with pytest.raises(ValueError, match='strictly interior'):
        geometry_report(x, u, ids, roles, heldout_interval=(-2., 0.))


def test_circular_geometry_uses_periodic_coordinates_and_rejects_wrong_wrap():
    u = np.tile(np.linspace(-np.pi, np.pi, 81, endpoint=False), 4)
    ids = np.repeat(['a', 'b', 'c', 'd'], 81)
    roles = np.repeat(['discovery', 'discovery', 'selection', 'selection'], 81)
    x = np.column_stack((np.sin(u), np.cos(u)))
    report = geometry_report(x, u, ids, roles, heldout_interval=(-.5, .5), circular=True, alpha=1e-8)
    assert report['models']['periodic_first']['raw_activation_mse'] < 1e-12
    assert report['models']['periodic_first']['basis_dimensions'] == 2
    assert report['models']['periodic_second']['basis_dimensions'] == 4
    u[0] = np.pi
    with pytest.raises(ValueError, match='Angles'):
        geometry_report(x, u, ids, roles, heldout_interval=(-.5, .5), circular=True)


def test_data_guard_excludes_confirmation_and_identity_collisions():
    data = fixture()
    assert validate_development(data, dimension=6) == {'discovery': 2, 'selection': 2}
    data['split'][-1] = 'confirmation'
    with pytest.raises(ValueError, match='no confirmation'):
        validate_development(data, dimension=6)
    data = fixture(); data['match_ids'][-1] = 'a'
    with pytest.raises(ValueError, match='split boundaries'):
        validate_development(data, dimension=6)
    data = fixture(); data['frame_index'][1] = 0
    with pytest.raises(ValueError, match='Duplicate'):
        validate_development(data, dimension=6)


def test_scalar_and_block_sparsity_signs_and_parameter_budgets():
    x = torch.tensor([[-5., 4., -3., 2.]])
    models = {kind: TopKDictionary(4, width=4, active=2, kind=kind, group_size=2)
              for kind in ('relu', 'signed', 'block')}
    for model in models.values():
        with torch.no_grad():
            model.encoder.weight.copy_(torch.eye(4)); model.encoder.bias.zero_()
    torch.testing.assert_close(models['relu'].encode(x), torch.tensor([[0., 4., 0., 2.]]))
    torch.testing.assert_close(models['signed'].encode(x), torch.tensor([[-5., 4., 0., 0.]]))
    torch.testing.assert_close(models['block'].encode(x), torch.tensor([[-5., 4., 0., 0.]]))
    assert {sum(p.numel() for p in model.parameters()) for model in models.values()} == {36}
    with pytest.raises(ValueError, match='divisible'):
        TopKDictionary(4, width=16, active=3, kind='block', group_size=8)


def test_temporal_links_do_not_cross_entity_view_clip_match_or_frame_gap():
    data = fixture()
    edges = temporal_edges(data)
    assert len(edges) == 10
    data['entity_ids'][1] = 'car'
    data['view_index'][3] = 1
    data['clip_ids'][4] = 'ax'
    data['timestamps'][11] += .5
    edges = temporal_edges(data)
    assert len(edges) == 4  # b's first four pairs only
    for a, b in edges:
        for key in ('entity_ids', 'view_index', 'clip_ids', 'match_ids'):
            assert data[key][a] == data[key][b]
        assert data['frame_index'][b] - data['frame_index'][a] == 1


def test_training_and_normalizer_ignore_selection_values_and_require_temporal_evidence():
    torch.set_num_threads(1)
    data = fixture(); other = copy.deepcopy(data)
    other['X'][other['split'] == 'selection'] += 1e4
    kwargs = dict(kind='block', width=16, active=4, group_size=2, steps=2, batch_size=8, seed=2)
    model, normalization, report = train_dictionary(data, **kwargs)
    changed_model, changed_normalization, changed_report = train_dictionary(other, **kwargs)
    for key in model.state_dict():
        torch.testing.assert_close(model.state_dict()[key], changed_model.state_dict()[key], rtol=0, atol=0)
    for key in normalization:
        np.testing.assert_array_equal(normalization[key], changed_normalization[key])
    assert report['normalizer_sha256'] == changed_report['normalizer_sha256']
    assert report['metrics']['selection']['active_scalar_mean'] == 4
    assert report['metrics']['selection']['active_block_mean'] == 2
    assert np.unique(normalization['scale']).size == 1
    original_distance = np.linalg.norm(data['X'][0] - data['X'][1])
    normalized_distance = np.linalg.norm((data['X'][0] - data['X'][1]) / normalization['scale'])
    assert normalized_distance * normalization['scale'][0] == pytest.approx(original_distance)
    with pytest.raises(ValueError, match='verified entity'):
        train_dictionary(data, **kwargs, temporal_weight=.1)
    _, _, temporal_report = train_dictionary(data, **kwargs, temporal_weight=.1, entity_mapping_verified=True)
    assert temporal_report['temporal_edges'] == 10
    _, _, shuffled = train_dictionary(data, **kwargs, temporal_weight=.1,
                                       entity_mapping_verified=True, temporal_control='shuffled_within_match')
    assert shuffled['temporal_edges'] == temporal_report['temporal_edges']
    assert shuffled['temporal_control'] == 'shuffled_within_match'
    _, _, view_only = train_dictionary(data, **kwargs, temporal_weight=.1,
                                       view_identity_verified=True)
    assert view_only['temporal_identity_scope'] == 'view_identity_only'
    assert view_only['tracked_entity_claim'] is False


def test_soft_support_penalty_has_finite_nonzero_gradient():
    torch.manual_seed(8)
    model = TopKDictionary(6, width=16, active=4, kind='block', group_size=2)
    first, second = torch.randn(5, 6), torch.randn(5, 6)
    loss = (model.support_surrogate(first) - model.support_surrogate(second)).square().mean()
    loss.backward()
    grad = model.encoder.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_gate_requires_unchanged_passed_probe_and_causal_evidence(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/analyze_feature_geometry.py'
    spec = importlib.util.spec_from_file_location('feature_geometry_cli_test', path)
    script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    evidence = tmp_path / 'probe.json'; evidence.write_text(json.dumps({'status': 'passed_development_probe_audit'}))
    causal = tmp_path / 'causal.json'; causal.write_text(json.dumps({'status': 'passed_model_intervention_audit'}))
    item = {'path': evidence.name, 'sha256': script.sha256(evidence)}
    gate = {'status': 'passed_feature_development_prerequisites', 'input_npz_sha256': 'data',
            'selected_probe_sha256': 'probe', 'scope': 'development_only', 'physical_causality_claim': False,
            'causal_scope': 'internal_model_output',
            'probe_audit': item, 'causal_audit': {'path': causal.name, 'sha256': script.sha256(causal)}}
    script.validate_gate(gate, 'data', 'probe', base_dir=tmp_path)
    with pytest.raises(ValueError, match='bind'):
        script.validate_gate(gate, 'changed', 'probe', base_dir=tmp_path)
    evidence.write_text(json.dumps({'status': 'failed'}))
    with pytest.raises(ValueError, match='changed'):
        script.validate_gate(gate, 'data', 'probe', base_dir=tmp_path)
    item['sha256'] = script.sha256(evidence)
    with pytest.raises(ValueError, match='has not passed'):
        script.validate_gate(gate, 'data', 'probe', base_dir=tmp_path)


def test_descriptor_dimension_is_registered_and_current_delta_is_excluded():
    path = Path(__file__).resolve().parents[1] / 'scripts/analyze_feature_geometry.py'
    spec = importlib.util.spec_from_file_location('feature_descriptor_cli_test', path)
    script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    for dimension in (1536, 2048):
        selection = {'descriptor_identity': f'current-{dimension}', 'feature_dim': dimension,
                     'temporal_readout': 'current'}
        assert script.validate_descriptor(selection, selection) == dimension
        with pytest.raises(ValueError, match='dimension'):
            script.validate_descriptor(selection, {**selection, 'feature_dim': dimension + 1})
        with pytest.raises(ValueError, match='current descriptor'):
            script.validate_descriptor({**selection, 'temporal_readout': 'current+delta'}, selection)
    selection = {'descriptor_identity': 'delta', 'feature_dim': 4096, 'temporal_readout': 'current+delta'}
    with pytest.raises(ValueError, match='dimension'):
        script.validate_descriptor(selection, selection)
