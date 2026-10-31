"""Strict checkpoint/data I/O; all image tensors use the upstream CIFAR normalization."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(x):
    return (x - x.new_tensor(MEAN).view(3, 1, 1)) / x.new_tensor(STD).view(3, 1, 1)


def denormalize(x):
    return x * x.new_tensor(STD).view(3, 1, 1) + x.new_tensor(MEAN).view(3, 1, 1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_private(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not silently overwrite keys or reference training exports.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as f:
        torch.save(obj, f)


def load_tensors(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def load_data(path, role=None):
    obj = load_tensors(path)
    x, y = obj['images'], obj['labels']
    if x.ndim != 4 or tuple(x.shape[1:]) != (3, 32, 32) or len(x) == 0:
        raise ValueError('Expected nonempty [N,3,32,32] CIFAR-normalized images')
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('Images must be finite floating-point tensors')
    if y.dtype != torch.long or y.shape != (len(x),) or y.min() < 0 or y.max() >= 10:
        raise ValueError('Expected int64 CIFAR-10 labels in [0,9]')
    if obj.get('normalization') != {'mean': list(MEAN), 'std': list(STD)}:
        raise ValueError('Missing or incompatible normalization metadata')
    if role and obj.get('role') != role:
        raise ValueError(f'Expected data role {role!r}, got {obj.get("role")!r}')
    return TensorDataset(x.float(), y), obj


def data_bundle(x, y, role, **metadata):
    return dict(images=x.cpu().float(), labels=y.cpu().long(), role=role,
                normalization={'mean': list(MEAN), 'std': list(STD)}, **metadata)


class TinySmokeNet(nn.Module):
    """Only for offline plumbing tests; never a CIFAR research baseline."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(3072, 48), nn.Tanh(), nn.Linear(48, 10))

    def forward(self, x):
        return self.layers(x)


def build_model(arch):
    if arch == 'resnet18':
        from architectures.nets_ResNet18 import ResNet18
        return ResNet18()
    if arch == 'tiny_smoke':
        return TinySmokeNet()
    raise ValueError(f'Unsupported architecture {arch!r}; first release supports CIFAR-10 ResNet18')


def load_model(path, arch='resnet18', device='cpu'):
    obj = load_tensors(path)
    if 'state_dict' in obj:
        if obj.get('arch', arch) != arch:
            raise ValueError('Checkpoint architecture mismatch')
        state = obj['state_dict']
    else:
        state = obj  # Upstream utils.util.save_model writes a bare state_dict.
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model = build_model(arch)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def save_model(path, model, arch):
    # Deliverable checkpoints intentionally contain no codeword, queries or recipient ID.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'arch': arch, 'state_dict': {k: v.detach().cpu().clone()
                                          for k, v in model.state_dict().items()}}, path)


def predict(model, images, batch_size=64, device='cpu'):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(x.to(device)).argmax(1).cpu()
                          for x in images.split(batch_size)])
