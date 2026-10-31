"""Report correct/wrong/rejected attribution, clean utility and paired trigger harm."""
import copy
import csv
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from .common import load_data, load_model, load_tensors, predict, save_json, save_model, seed_all
from .keys import overlay
from .training import freeze_bn
from .verification import decide, load_manifest, outcome_counts, read_thresholds, score_manifest


def safety_metrics(model, reference, data, key, recipient, batch_size=64, device='cpu',
                   shifts=((0, 0), (2, 2), (-2, -2)), max_samples=0, original_bank=None):
    x, y = data.tensors
    if max_samples:
        x, y = x[:max_samples], y[:max_samples]
    clean = predict(model, x, batch_size, device)
    ref_clean = predict(reference, x, batch_size, device)
    rows = []
    bits = key['codebook'].shape[1]
    for bit in range(bits):
        # Unknown models are evaluated against both candidate labels; known copies use their code.
        targets = key['label_pairs'][bit].tolist() if recipient is None else [
            int(key['label_pairs'][bit, key['codebook'][recipient, bit]])]
        for shift in shifts:
            triggered = overlay(x, key, torch.full((len(x),), bit), shift=shift)
            pred = predict(model, triggered, batch_size, device)
            ref = predict(reference, triggered, batch_size, device)
            for target in targets:
                eligible = y != target
                denom = int(eligible.sum())
                rows.append(dict(bit=bit, shift=list(shift), target=target, eligible=denom,
                                 triggered_accuracy=(pred == y).float().mean().item(),
                                 reference_triggered_accuracy=(ref == y).float().mean().item(),
                                 target_asr=(pred[eligible] == target).float().mean().item() if denom else None,
                                 reference_target_asr=(ref[eligible] == target).float().mean().item() if denom else None,
                                 prediction_change=(pred != clean).float().mean().item()))
    old_rows = []
    if original_bank is not None:
        for i, (pattern, mask, target) in enumerate(zip(original_bank['patterns'], original_bank['masks'],
                                                       original_bank['targets'])):
            # Recovered upstream patterns already live in normalized model-input coordinates.
            triggered = (1-mask)*x + mask*pattern
            pred, ref = predict(model, triggered, batch_size, device), predict(reference, triggered, batch_size, device)
            target = int(target)
            eligible = y != target
            old_rows.append(dict(index=i, target=target, eligible=int(eligible.sum()),
                                 target_asr=(pred[eligible] == target).float().mean().item() if eligible.any() else None,
                                 reference_target_asr=(ref[eligible] == target).float().mean().item() if eligible.any() else None))
    valid = [r for r in rows if r['eligible']]
    return dict(samples=len(x), clean_accuracy=(clean == y).float().mean().item(),
                reference_clean_accuracy=(ref_clean == y).float().mean().item(),
                mean_target_asr=sum(r['target_asr'] for r in valid)/len(valid) if valid else None,
                mean_reference_target_asr=sum(r['reference_target_asr'] for r in valid)/len(valid) if valid else None,
                worst_target_asr=max((r['target_asr'] for r in valid), default=None),
                worst_asr_increase=max((r['target_asr']-r['reference_target_asr'] for r in valid), default=None),
                per_pattern=rows, original_triggers=old_rows)


def evaluate(manifest, key, thresholds_path, output, variants=4, arch='resnet18', device='cpu',
             batch_size=64, test_data=None, base_path=None, contribution_path=None,
             original_triggers=None, max_samples=0):
    thresholds = read_thresholds(thresholds_path, key, variants)
    report = score_manifest(manifest, key, 'test', variants, arch, device, batch_size)
    for row in report['rows']:
        if row['recipient'] is None and row['sha256'] in thresholds['calibration_unknown_hashes']:
            raise ValueError('Unknown test model was used for threshold calibration; use independent controls')
        row['decision'] = decide(row['scores'], thresholds['tau'], thresholds['delta'])
    report['counts'] = outcome_counts(report['rows'], thresholds['tau'], thresholds['delta'])
    counts = report['counts']
    report['rates'] = dict(correct_attribution=counts['correct']/counts['known'] if counts['known'] else None,
                           wrong_attribution=counts['wrong_known']/counts['known'] if counts['known'] else None,
                           known_rejection=counts['rejected_known']/counts['known'] if counts['known'] else None,
                           unknown_false_attribution=counts['false_attributed_unknown']/counts['unknown']
                           if counts['unknown'] else None)
    report['unknown_zero_error_upper95'] = (1-.05**(1/counts['unknown'])) if (
        counts['unknown'] and not counts['false_attributed_unknown']) else None
    report['statistical_note'] = ('Model-level counts. Binomial bound only meaningful for independent representative '
                                  'unknown models and fixed thresholds. Related checkpoints are correlated.')
    if test_data:
        if not base_path:
            raise ValueError('Safety evaluation requires a paired --base checkpoint')
        data, _ = load_data(test_data, 'test')
        reference = load_model(base_path, arch, device)
        bank = load_tensors(original_triggers) if original_triggers else None
        contribution = load_data(contribution_path, 'contribution_test')[0] if contribution_path else None
        for row, record in zip(report['rows'], load_manifest(manifest, key)):
            model = load_model(record['path'], arch, device)
            row['safety'] = safety_metrics(model, reference, data, key, row['recipient'], batch_size,
                                           device, max_samples=max_samples, original_bank=bank)
            if contribution:
                cx, cy = contribution.tensors
                row['contribution_verification_rate'] = (predict(model, cx, batch_size, device) == cy).float().mean().item()
                row['reference_contribution_verification_rate'] = (
                    predict(reference, cx, batch_size, device) == cy).float().mean().item()
            del model
    save_json(output, report)
    csv_path = Path(output).with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'true_recipient', 'predicted_recipient', 'status', *key['recipient_ids']])
        for row in report['rows']:
            writer.writerow([row['name'], row['recipient'], row['decision']['recipient'],
                             row['decision']['status'], *row['scores']])
    return report


def modify_model(checkpoint, output, method, arch='resnet18', device='cpu', amount=.2,
                 data_path=None, epochs=1, lr=.001, batch_size=64, seed=7):
    if not 0 <= amount < 1 or epochs < 1 or lr <= 0:
        raise ValueError('Invalid attack strength')
    seed_all(seed)
    model = load_model(checkpoint, arch, device)
    if method == 'prune':
        params = [(m, 'weight') for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
        for module, name in params:
            prune.remove(module, name)
    elif method == 'finetune':
        if not data_path:
            raise ValueError('Fine-tuning requires --attack-data with role attack, not test/defense')
        data, _ = load_data(data_path, 'attack')
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)
        model.train()
        freeze_bn(model)
        for _ in range(epochs):
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(x.to(device)), y.to(device)).backward()
                optimizer.step()
    elif method == 'weight-round':
        # Simulated int8 weight rounding, NOT deployment/backend int8 quantization.
        with torch.no_grad():
            for p in model.parameters():
                if p.ndim >= 2:
                    scale = p.abs().max()/127
                    if scale > 0:
                        p.copy_((p/scale).round().clamp(-127, 127)*scale)
    else:
        raise ValueError('Unknown modification')
    save_model(output, model, arch)
    return model
