import copy
from pathlib import Path

import pytest
import torch

from traceability.common import TinySmokeNet, data_bundle, load_tensors, save_model, save_private
from traceability.keys import generate_key
from traceability.training import EmbedConfig, personalize


def test_independent_copies_and_no_secrets_in_deliverable(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    base = TinySmokeNet()
    save_model(tmp_path/'base.pt', base, 'tiny_smoke')
    for role in ['defense', 'validation']:
        x = torch.randn(8, 3, 32, 32)
        y = base(x).argmax(1)
        save_private(tmp_path/f'{role}.pt', data_bundle(x, y, role))
    key = generate_key(2, 4, 1, seed=3)
    config = EmbedConfig(epochs=1, batch_size=4, max_steps=1, max_accuracy_drop=1., robust_weight=.1,
                         train_variants=1, validation_variants=1)
    args = (tmp_path/'base.pt', key, tmp_path/'defense.pt', tmp_path/'validation.pt')
    personalize(*args, tmp_path/'both', config, 'tiny_smoke')
    personalize(*args, tmp_path/'alone', config, 'tiny_smoke', selected=[1])
    both = load_tensors(tmp_path/'both/distribution/copy_001/model.pt')
    alone = load_tensors(tmp_path/'alone/distribution/copy_001/model.pt')
    assert set(both) == {'arch', 'state_dict'}
    assert all(torch.equal(value, alone['state_dict'][name]) for name, value in both['state_dict'].items())
    unchanged = load_tensors(tmp_path/'base.pt')['state_dict']
    assert all(torch.equal(value, unchanged[name]) for name, value in base.state_dict().items())


def test_overlap_rejected(tmp_path):
    for role in ['defense', 'validation']:
        save_private(tmp_path/f'{role}.pt', data_bundle(torch.zeros(2, 3, 32, 32), torch.zeros(2).long(),
                                                       role, source='cifar10/train', indices=[1, 2]))
    with pytest.raises(ValueError, match='overlap'):
        personalize('unused.pt', generate_key(2, 4, 1), tmp_path/'defense.pt', tmp_path/'validation.pt',
                    tmp_path/'out', EmbedConfig())
