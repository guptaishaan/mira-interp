"""Engineering tests for pixel preprocessing, token/time layout, and state identity."""
import numpy as np
import pytest
import torch

from mira_interp.data import TARGET_NAMES
from mira_interp.video_evaluator import convert_legacy_biases, endpoint_labels, pool_last_tubelet, preprocess_frames


def test_bias_conversion_preserves_original_qv_and_supplies_only_known_zero_k():
    source = {'layer.q_bias': torch.tensor([1., 2.]), 'layer.v_bias': torch.tensor([3., 4.]),
              'layer.query.weight': torch.eye(2)}
    converted, count = convert_legacy_biases(source)
    assert count == 1 and 'layer.q_bias' in source
    torch.testing.assert_close(converted['layer.query.bias'], source['layer.q_bias'])
    torch.testing.assert_close(converted['layer.value.bias'], source['layer.v_bias'])
    assert torch.count_nonzero(converted['layer.key.bias']) == 0
    with pytest.raises(ValueError, match='Ambiguous'):
        convert_legacy_biases({**source, 'layer.query.bias': torch.ones(2)})


def test_full_frame_resize_keeps_both_horizontal_edges_and_uses_source_units_once():
    torch.set_num_threads(1)
    frames = np.full((1, 16, 3, 2, 4), 127.5, np.float32)
    frames[..., 0] = 0; frames[..., -1] = 255
    output = preprocess_frames(frames, {'image_mean': [0, 0, 0], 'image_std': [1, 1, 1]}, 'cpu')
    assert output.shape == (1, 16, 3, 224, 224)
    torch.testing.assert_close(output[..., 0], torch.zeros_like(output[..., 0]), rtol=0, atol=0)
    torch.testing.assert_close(output[..., -1], torch.ones_like(output[..., -1]), rtol=0, atol=0)
    with pytest.raises(ValueError, match='source values'):
        preprocess_frames(frames + 256, {'image_mean': [0]*3, 'image_std': [1]*3}, 'cpu')


def test_last_tubelet_spatial_bins_match_independent_explicit_ranges():
    hidden = torch.full((1, 8*14*14, 768), -100.)
    last = torch.arange(196.).reshape(14, 14)
    hidden[:, -196:, :] = last.flatten()[None, :, None]
    result = pool_last_tubelet(hidden).reshape(1, 768, 3, 4)
    expected = np.zeros((3, 4))
    for row in range(3):
        for col in range(4):
            expected[row, col] = last[int(np.floor(row*14/3)):int(np.ceil((row+1)*14/3)),
                                      int(np.floor(col*14/4)):int(np.ceil((col+1)*14/4))].mean()
    np.testing.assert_allclose(result[0, 0].numpy(), expected)
    torch.testing.assert_close(result[0, 0], result[0, -1])
    with pytest.raises(ValueError, match='no CLS'):
        pool_last_tubelet(torch.zeros(1, 1569, 768))


def test_labels_use_own_view_final_frame_and_correct_local_player_without_broadcast():
    targets = np.arange(4*16*30, dtype=np.float64).reshape(4, 16, 30)
    data = {'targets': targets, 'target_names': np.asarray(TARGET_NAMES), 'player_ids': np.arange(4),
            'timestamps': np.arange(64.).reshape(4, 16)/20, 'source_frame_indices': np.arange(16)+100}
    labels = endpoint_labels(data)
    for view in range(4):
        ego = targets[view, 15, 6*(view+1):6*(view+2)]
        np.testing.assert_equal(labels['y'][view], np.r_[ego, targets[view, 15, :6]-ego])
        assert labels['timestamps'][view] == data['timestamps'][view, 15]
        assert labels['source_frame_index'][view] == 115
    data['source_frame_indices'] = np.arange(64).reshape(4, 16)
    np.testing.assert_equal(endpoint_labels(data)['source_frame_index'], [15, 31, 47, 63])
    data['target_names'] = data['target_names'][::-1]
    with pytest.raises(ValueError, match='canonical'):
        endpoint_labels(data)
