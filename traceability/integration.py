"""Export server-held data and recovered contribution keys after original Sanitizer training."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import data_bundle, save_private, save_model, save_json


def export_dataset(dataset, path, role, indices=None, source='cifar10/train'):
    images, labels = [], []
    for x, y in DataLoader(dataset, batch_size=128, shuffle=False):
        images.append(x.cpu())
        labels.append(y.cpu())
    if not images:
        raise ValueError('Cannot export empty dataset')
    save_private(path, data_bundle(torch.cat(images), torch.cat(labels), role,
                                  indices=list(map(int, indices)) if indices is not None else [], source=source))


def export_sanitizer(model, defense, validation, directory, targets, patterns, masks):
    from identification.fine_tune import create_color_batch
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    save_model(root / 'sanitized_base.pt', model, 'resnet18')
    export_dataset(defense, root/'defense.pt', 'defense', defense.dataset.indices)
    export_dataset(validation, root/'validation.pt', 'validation', validation.dataset.indices)
    bank_patterns, bank_masks, xs, ys = [], [], [], []
    for client, target in enumerate(targets):
        if not 0 <= int(target) < 10:
            raise ValueError('Invalid recovered target; do not export an incomplete Sanitizer run')
        pattern = patterns[client][target].detach().cpu()
        mask = masks[client][target].detach().cpu()
        bank_patterns.append(pattern)
        bank_masks.append(mask)
        bg = create_color_batch(client, 1)
        xs.append((1-mask)*bg + mask*pattern)
        ys.append(int(target))
    x, y = torch.cat(xs), torch.tensor(ys)
    # Original contribution verification uses fixed secret backgrounds. These two files
    # intentionally carry the same canonical keys; no independent-query claim is made.
    save_private(root/'contribution_train.pt', data_bundle(x, y, 'contribution_train'))
    save_private(root/'contribution_test.pt', data_bundle(x, y, 'contribution_test'))
    save_private(root/'original_triggers.pt', dict(patterns=torch.stack(bank_patterns),
                 masks=torch.stack(bank_masks), targets=torch.tensor(targets)))
    save_json(root/'export_info.json', {'arch': 'resnet18', 'dataset': 'cifar10',
              'note': 'SERVER ONLY. Export created AFTER harmless relearning. '
                      'Contribution files are canonical fixed keys, not independent held-out variants.'})
