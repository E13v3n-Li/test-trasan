"""Regression checks for upstream bugs that directly affect the exported base model."""
import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import TensorDataset


def test_weighted_average_preserves_integer_buffers():
    from average.Fed import FedAvg
    result = FedAvg([{'w': torch.tensor([1.]), 'n': torch.tensor(2)},
                     {'w': torch.tensor([3.]), 'n': torch.tensor(3)}], [1, 3])
    assert result['w'].item() == 2.5
    assert result['n'].dtype == torch.int64 and result['n'].item() == 3


def test_pruned_bn_channels_emit_zero():
    from architectures.nets_ResNet18 import ResNet18
    from core.resnet_cifar.unlearning_extraction_function import create_pruned_model
    torch.set_num_threads(2)
    model = ResNet18().eval()
    with torch.no_grad():
        model.bn1.bias[0] = 5
    pruned = create_pruned_model(model, {'conv1.weight': [0]}).model.eval()
    with torch.no_grad():
        assert torch.count_nonzero(pruned.bn1(pruned.conv1(torch.rand(1, 3, 32, 32)))[:, 0]) == 0
    assert model.bn1.bias[0] == 5  # Original object untouched.


def test_final_relearning_supports_two_clients_and_partial_batch():
    from identification.utils_training import final_unharmful_retrain
    from traceability.common import TinySmokeNet
    torch.set_num_threads(2)
    model = TinySmokeNet()
    data = TensorDataset(torch.randn(5, 3, 32, 32), torch.arange(5))
    patterns = {i: [torch.zeros(3, 32, 32) for _ in range(10)] for i in range(2)}
    masks = {i: [torch.zeros(32, 32) for _ in range(10)] for i in range(2)}
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = final_unharmful_retrain(data, model, [0, 1], patterns, masks, '',
        dict(device='cpu', relearning_lr=.001, relearning_eps=1, relearning_batch_size=3))
    assert any(not torch.equal(before[k], v) for k, v in result.items())


def test_round_start_and_reserved_data(monkeypatch, tmp_path):
    import main_fed as fed
    torch.set_num_threads(2)
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    data = TensorDataset(torch.randn(1000, 3, 32, 32), torch.zeros(1000).long())
    monkeypatch.setattr(fed, 'dataset_train', data, raising=False)
    monkeypatch.setattr(fed, 'dataset_test', data, raising=False)
    monkeypatch.setattr(fed, 'dict_users', {0: list(range(500)), 1: list(range(500, 1000))}, raising=False)
    monkeypatch.setattr(fed, 'transform_test', None, raising=False)
    monkeypatch.setattr(fed, 'log_dir', str(tmp_path), raising=False)
    monkeypatch.setattr(fed, 'now_str', 'test', raising=False)
    starts, local_indices, reserved = [], [], set()
    class Local:
        def __init__(self, **kwargs):
            local_indices.extend(kwargs['idxs'])
        def train(self, net, iter_outside):
            starts.append(net.weight.item())
            with torch.no_grad():
                net.weight.add_(1)
            return net.state_dict(), 1.
    def identify(args, local, other, defense, *rest, **kw):
        reserved.update(defense.dataset.indices)
        return local, {}
    def pruned(local, indices):
        m = copy.deepcopy(local)
        m.weight.data.zero_()
        return SimpleNamespace(model=m)
    monkeypatch.setattr(fed, 'LocalUpdate', Local)
    monkeypatch.setattr(fed, 'main_cifar10_ul', identify)
    monkeypatch.setattr(fed, 'create_pruned_model', pruned)
    monkeypatch.setattr(fed, 'reverse_engineering_nn', lambda *a, **kw:
                        (0, kw['triggers_1'], kw['masks_1'], {}))
    monkeypatch.setattr(fed, 'final_unharmful_retrain', lambda data, net, *a, **kw: net.state_dict())
    monkeypatch.setattr(fed, 'test_silo_with_asr_unharmful_cifar', lambda *a, **kw: None)
    monkeypatch.setattr(fed, 'save_model', lambda *a, **kw: None)
    args = SimpleNamespace(device='cpu', dataset='cifar', all_clients=True, num_users=2,
                           defense_data_ratio=.05, trace_export_dir=None, epochs=2,
                           gpu=-1, log_dir=str(tmp_path))
    fed.train_model(args, model, {0:'white_block', 1:'white_block'}, {0:0, 1:0}, {0:True, 1:True})
    assert starts == [0., 0., 0., 0.]  # All rounds aggregated PRUNED updates.
    assert not reserved.intersection(local_indices)
