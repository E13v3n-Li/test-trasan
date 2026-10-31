"""Label-only decoder, model-level calibration and optional PNG-over-HTTP queries."""
import base64
import io
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .common import denormalize, load_model, predict, read_json, save_json, sha256
from .keys import expected, queries


def validate_labels(labels, count):
    labels = torch.as_tensor(labels)
    if labels.shape != (count,) or labels.dtype == torch.bool or labels.is_floating_point():
        raise ValueError('Response must contain exactly N integer top-1 labels')
    if labels.numel() and (labels.min() < 0 or labels.max() >= 10):
        raise ValueError('Response labels must use the original CIFAR-10 indices 0..9')
    return labels.long()


def scores_for_labels(key, labels, bit_ids):
    labels = validate_labels(labels, len(bit_ids))
    return [(labels == expected(key, i, bit_ids)).double().mean().item()
            for i in range(len(key['recipient_ids']))]


def decide(scores, tau, delta):
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or len(s) < 2 or not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError('Expected at least two finite scores in [0,1]')
    if not (math.isfinite(tau) and 0 <= tau <= 1.000001 and math.isfinite(delta) and 0 < delta <= 1.000001):
        raise ValueError('Invalid calibrated thresholds')
    order = np.argsort(-s, kind='stable')
    first, second = order[:2]
    margin = float(s[first]-s[second])
    accepted = bool(s[first] >= tau and margin >= delta and margin > 0)
    return dict(status='attributed' if accepted else 'unattributable',
                recipient=int(first) if accepted else None,
                top_score=float(s[first]), margin=margin, scores=s.tolist())


def load_manifest(path, key):
    records = read_json(path)['records']
    if not records or len({r['name'] for r in records}) != len(records):
        raise ValueError('Manifest requires nonempty uniquely named records')
    for r in records:
        identity = r['recipient']
        if identity is not None and (type(identity) is not int or not 0 <= identity < len(key['recipient_ids'])):
            raise ValueError('recipient must be a zero-based identity or null for an unknown model')
        r['path'] = str((Path(path).resolve().parent / r['path']).resolve())
    hashes = [sha256(r['path']) for r in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate checkpoint bytes in manifest; do not inflate model counts')
    return records


def score_manifest(path, key, split, variants, arch='resnet18', device='cpu', batch_size=64):
    images, bits, digest = queries(key, split, variants)
    rows = []
    for r in load_manifest(path, key):
        model = load_model(r['path'], arch, device)
        labels = predict(model, images, batch_size, device)
        rows.append(dict(name=r['name'], recipient=r['recipient'], sha256=sha256(r['path']),
                         scores=scores_for_labels(key, labels, bits)))
        del model
    return dict(key_id=key['key_id'], split=split, variants=variants, query_sha256=digest, rows=rows)


def outcome_counts(rows, tau, delta):
    counts = dict(known=0, unknown=0, correct=0, wrong_known=0, rejected_known=0,
                  false_attributed_unknown=0, rejected_unknown=0)
    for row in rows:
        found = decide(row['scores'], tau, delta)['recipient']
        truth = row['recipient']
        if truth is None:
            counts['unknown'] += 1
            counts['rejected_unknown' if found is None else 'false_attributed_unknown'] += 1
        else:
            counts['known'] += 1
            counts['correct' if found == truth else ('rejected_known' if found is None else 'wrong_known')] += 1
    return counts


def calibrate(report, key, max_false_rate=.01, min_recall=.8):
    if report['split'] != 'calibration' or report['key_id'] != key['key_id']:
        raise ValueError('Thresholds must use this key and calibration queries')
    if not 0 <= max_false_rate <= 1 or not 0 <= min_recall <= 1:
        raise ValueError('Rates must be in [0,1]')
    rows = report['rows']
    known = [r for r in rows if r['recipient'] is not None]
    unknown = [r for r in rows if r['recipient'] is None]
    if not unknown or {r['recipient'] for r in known} != set(range(len(key['recipient_ids']))):
        raise ValueError('Calibration requires every enrolled identity and at least one unknown control')
    # Decisions change only at observed score/margin boundaries. Include rejection thresholds.
    tops = [max(r['scores']) for r in rows]
    margins = [sorted(r['scores'], reverse=True)[0]-sorted(r['scores'], reverse=True)[1] for r in rows]
    taus = sorted({0., 1.0000001, *tops, *(float(np.nextafter(x, np.inf)) for x in tops)})
    deltas = sorted({1e-9, 1.0000001, *(max(1e-9, x) for x in margins),
                     *(max(1e-9, float(np.nextafter(x, np.inf))) for x in margins)})
    best = None
    # Vectorized candidate testing over whole models; variants are NOT independent negatives.
    matrix = np.asarray([r['scores'] for r in rows])
    predicted = matrix.argmax(1)
    top = matrix.max(1)
    gap = np.sort(matrix, axis=1)[:, -1] - np.sort(matrix, axis=1)[:, -2]
    truth = np.asarray([-1 if r['recipient'] is None else r['recipient'] for r in rows])
    is_known = truth >= 0
    for tau in taus:
        for delta in deltas:
            accepted = (top >= tau) & (gap >= delta) & (gap > 0)
            false_unknown = np.count_nonzero(accepted & ~is_known) / len(unknown)
            wrong_known = np.count_nonzero(accepted & is_known & (predicted != truth)) / len(known)
            correct = np.count_nonzero(accepted & is_known & (predicted == truth))
            if false_unknown <= max_false_rate and wrong_known <= max_false_rate:
                rank = (correct, -false_unknown, -wrong_known, tau, delta)
                if best is None or rank > best[0]:
                    best = (rank, tau, delta)
    if best is None or best[0][0] / len(known) < min_recall:
        raise ValueError('No thresholds meet empirical false-attribution and recall constraints; improve embedding')
    _, tau, delta = best
    counts = outcome_counts(rows, tau, delta)
    return dict(version=1, key_id=key['key_id'], tau=tau, delta=delta,
                variants=report['variants'], query_sha256=report['query_sha256'],
                calibration_counts=counts, max_empirical_false_rate=max_false_rate,
                calibration_unknown_hashes=[r['sha256'] for r in unknown],
                note='Empirical calibration only. Threshold selection invalidates a formal held-out bound. '
                     'Validate on independent unknown models; queries/variants are correlated.')


def read_thresholds(path, key, variants):
    obj = read_json(path)
    if obj.get('version') != 1 or obj['key_id'] != key['key_id'] or obj['variants'] != variants:
        raise ValueError('Threshold key/version/query-budget mismatch')
    decide([0.] * len(key['recipient_ids']), obj['tau'], obj['delta'])
    return obj


def png_bytes(image):
    raw = denormalize(image).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    buffer = io.BytesIO()
    Image.fromarray(raw).save(buffer, format='PNG')
    return buffer.getvalue()


def http_labels(endpoint, images, batch_size=64, timeout=30, token=None):
    if not endpoint.startswith(('http://', 'https://')):
        raise ValueError('Endpoint must be HTTP(S)')
    labels = []
    for batch in images.split(batch_size):
        body = json.dumps({'images': [base64.b64encode(png_bytes(x)).decode('ascii') for x in batch]}).encode()
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        request = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        labels.extend(validate_labels(payload['labels'], len(batch)).tolist())
    return torch.tensor(labels, dtype=torch.long)


def export_queries(key, output_dir, variants=4):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    images, bits, digest = queries(key, 'test', variants)
    names = [f'query_{i:05d}.png' for i in range(len(images))]
    for name, image in zip(names, images):
        (output / name).write_bytes(png_bytes(image))
    save_json(output / 'request_manifest.json', dict(key_id=key['key_id'], split='test',
              variants=variants, query_sha256=digest, files=names))
    return dict(count=len(images), query_sha256=digest)
