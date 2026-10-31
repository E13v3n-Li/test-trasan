"""Independent post-Sanitizer copies; baseline D and optional one-step robustness E."""
import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader

from .common import load_data, load_model, predict, save_json, save_model, seed_all, sha256
from .keys import expected, overlay, queries


@dataclass
class EmbedConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = .001
    task_weight: float = 1.
    fingerprint_weight: float = 1.
    safety_weight: float = 1.
    contribution_weight: float = 0.
    robust_weight: float = 0.
    inner_lr: float = .001
    robust_every: int = 5
    train_variants: int = 16
    validation_variants: int = 4
    max_accuracy_drop: float = .02
    seed: int = 1
    max_steps: int = 0

    def validate(self):
        if min(self.epochs, self.batch_size, self.robust_every, self.train_variants,
               self.validation_variants) < 1 or self.lr <= 0 or self.inner_lr <= 0:
            raise ValueError('Training counts and learning rates must be positive')
        if min(self.task_weight, self.fingerprint_weight, self.safety_weight,
               self.contribution_weight, self.robust_weight) < 0:
            raise ValueError('Loss weights must be nonnegative')
        if not 0 <= self.max_accuracy_drop <= 1 or self.max_steps < 0:
            raise ValueError('Invalid accuracy constraint or max_steps')


def freeze_bn(model):
    # Keep deployment running statistics; artificial batches must not dominate them.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def one_step_robust_loss(model, x, y, probes, labels, inner_lr):
    """Differentiable look-ahead through ONE clean SGD step (not general attack immunity)."""
    params = dict(model.named_parameters())
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    clean_loss = F.cross_entropy(functional_call(model, (params, buffers), (x,)), y)
    grads = torch.autograd.grad(clean_loss, tuple(params.values()), create_graph=True)
    updated = {name: value-inner_lr*grad for (name, value), grad in zip(params.items(), grads)}
    return F.cross_entropy(functional_call(model, (updated, buffers), (probes,)), labels)


def personalize(base_path, key, defense_path, validation_path, out_dir, config,
                arch='resnet18', device='cpu', contribution_path=None, selected=None):
    config.validate()
    defense, defense_meta = load_data(defense_path, 'defense')
    validation, validation_meta = load_data(validation_path, 'validation')
    if sha256(defense_path) == sha256(validation_path):
        raise ValueError('Defense and validation files must differ')
    # Check sample IDs where provided, even if someone relabels the data role.
    if (defense_meta.get('source') == validation_meta.get('source') and
            set(defense_meta.get('indices', [])).intersection(validation_meta.get('indices', []))):
        raise ValueError('Defense/validation sample overlap')
    contribution = None
    if config.contribution_weight:
        if not contribution_path:
            raise ValueError('Contribution loss requires --contribution-data')
        contribution, _ = load_data(contribution_path, 'contribution_train')
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; do not overwrite a distribution record')
    out.mkdir(parents=True, exist_ok=True)
    base = load_model(base_path, arch, device)
    val_x, val_y = validation.tensors
    base_accuracy = (predict(base, val_x, config.batch_size, device) == val_y).float().mean().item()
    train_q, train_bits, _ = queries(key, 'train', config.train_variants)
    val_q, val_bits, _ = queries(key, 'calibration', config.validation_variants)
    recipients = list(range(len(key['recipient_ids']))) if selected is None else selected
    if not recipients or len(set(recipients)) != len(recipients) or any(
            i < 0 or i >= len(key['recipient_ids']) for i in recipients):
        raise ValueError('Invalid recipient selection')
    registry = dict(key_id=key['key_id'], base_sha256=sha256(base_path), arch=arch,
                    config=asdict(config), defense_sha256=sha256(defense_path),
                    validation_sha256=sha256(validation_path), base_validation_accuracy=base_accuracy,
                    records=[])
    for recipient in recipients:
        seed_all(config.seed)  # Same optimization randomness, different identity targets.
        model = copy.deepcopy(base)  # NEVER start from the preceding recipient.
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=.9, weight_decay=5e-4)
        loader = DataLoader(defense, batch_size=config.batch_size, shuffle=True, num_workers=0)
        fp_labels = expected(key, recipient, train_bits)
        val_labels = expected(key, recipient, val_bits)
        best_score, best_state, best_epoch = -1., None, None
        history, step, started = [], 0, time.perf_counter()
        for epoch in range(config.epochs):
            model.train()
            freeze_bn(model)
            sums = {name: 0. for name in ['task', 'fingerprint', 'safety', 'contribution', 'robust']}
            count = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                indices = torch.randint(len(train_q), (len(x),))
                q, target = train_q[indices].to(device), fp_labels[indices].to(device)
                losses = {'task': F.cross_entropy(model(x), y),
                          'fingerprint': F.cross_entropy(model(q), target)}
                if config.safety_weight:
                    bit_ids = torch.randint(key['codebook'].shape[1], (len(x),))
                    # Nominal + small translations/opacity changes; evaluate others independently.
                    shift = tuple(torch.randint(-1, 2, (2,)).tolist())
                    opacity = .7 + .3 * torch.rand(()).item()
                    safe = overlay(x, key, bit_ids, shift, opacity)
                    losses['safety'] = F.cross_entropy(model(safe), y)
                if contribution is not None:
                    ix = torch.randint(len(contribution), (len(x),))
                    cx, cy = contribution.tensors
                    losses['contribution'] = F.cross_entropy(model(cx[ix].to(device)), cy[ix].to(device))
                if config.robust_weight and step % config.robust_every == 0:
                    losses['robust'] = one_step_robust_loss(model, x, y, q, target, config.inner_lr)
                total = sum(getattr(config, f'{name}_weight') * loss for name, loss in losses.items())
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                for name, loss in losses.items():
                    sums[name] += loss.item()
                count += 1
                step += 1
                if config.max_steps and step >= config.max_steps:
                    break
            accuracy = (predict(model, val_x, config.batch_size, device) == val_y).float().mean().item()
            score = (predict(model, val_q, config.batch_size, device) == val_labels).float().mean().item()
            feasible = accuracy >= base_accuracy-config.max_accuracy_drop
            if feasible and score > best_score:
                best_score, best_epoch = score, epoch+1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = dict(epoch=epoch+1, validation_accuracy=accuracy, validation_fingerprint=score,
                       accuracy_constraint_passed=feasible, **{k: v/count for k, v in sums.items()})
            history.append(row)
            print(f'{key["recipient_ids"][recipient]} epoch {epoch+1}: acc={accuracy:.4f}, fp={score:.4f}', flush=True)
            if config.max_steps and step >= config.max_steps:
                break
        private = out / 'server_private'
        private.mkdir(exist_ok=True, mode=0o700)
        save_json(private / f'{recipient:03d}_training.json', history)
        if best_state is None:
            raise RuntimeError(f'No accuracy-feasible checkpoint for recipient {recipient}; inspect validation logs')
        model.load_state_dict(best_state)
        rel = f'distribution/copy_{recipient:03d}/model.pt'
        save_model(out / rel, model, arch)
        registry['records'].append(dict(name=f'copy_{recipient:03d}', path='../'+rel,
                                       recipient=recipient, sha256=sha256(out / rel),
                                       selected_epoch=best_epoch, validation_fingerprint=best_score,
                                       seconds=time.perf_counter()-started))
        save_json(private / 'registry.json', registry)
        # This is a template; user must add independently trained unknown controls.
        save_json(private / 'manifest.json', {'records': [
            {k: r[k] for k in ['name', 'path', 'recipient']} for r in registry['records']]})
    return registry
