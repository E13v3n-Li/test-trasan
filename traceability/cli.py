"""Command-line entry point. Run python trace_cli.py --help from the repository root."""
import argparse
import os
from pathlib import Path

import torch

from .common import load_model, predict, read_json, save_json, data_bundle, save_private, normalize
from .keys import read_key, write_key, queries
from .training import EmbedConfig, personalize
from .verification import (calibrate, decide, export_queries, http_labels, read_thresholds,
                           score_manifest, scores_for_labels, validate_labels)
from .evaluation import evaluate, modify_model


def model_args(p):
    p.add_argument('--arch', choices=['resnet18', 'tiny_smoke'], default='resnet18')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=64)


def key_args(p):
    p.add_argument('--key', required=True)
    p.add_argument('--variants', type=int, default=4)


def prepare_data(args):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    dataset = CIFAR10(args.data_root, train=args.role != 'test', download=args.download)
    if args.role == 'test':
        if args.indices:
            raise ValueError('Test export uses official CIFAR-10 test split; omit --indices')
        indices = list(range(len(dataset)))
    else:
        if not args.indices:
            raise ValueError('An explicit disjoint index list is required for defense/validation/attack')
        indices = read_json(args.indices)
    if not isinstance(indices, list) or not indices or len(set(indices)) != len(indices):
        raise ValueError('Indices must be a nonempty unique JSON list')
    if any(type(i) is not int or i < 0 or i >= len(dataset) for i in indices):
        raise ValueError('Invalid CIFAR sample index')
    if args.limit:
        indices = indices[:args.limit]
    convert = ToTensor()
    x = torch.stack([normalize(convert(dataset[i][0])) for i in indices])
    y = torch.tensor([dataset[i][1] for i in indices])
    save_private(args.output, data_bundle(x, y, args.role, indices=indices,
                 source='cifar10/test' if args.role == 'test' else 'cifar10/train'))
    return {'samples': len(indices), 'role': args.role}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=4)
    sub = p.add_subparsers(dest='command', required=True)
    k = sub.add_parser('keys', help='Create SERVER-ONLY private key; no overwrite')
    k.add_argument('--output', required=True)
    k.add_argument('--recipients', type=int, default=10)
    k.add_argument('--bits', type=int, default=64)
    k.add_argument('--distance', type=int, default=20)
    k.add_argument('--seed', type=int, help='Research reproducibility only; omit for a random secret')
    k.add_argument('--patch-size', type=int, default=4)
    d = sub.add_parser('prepare-data', help='Export normalized CIFAR tensors with explicit roles')
    d.add_argument('--data-root', default='./data/cifar')
    d.add_argument('--role', choices=['defense', 'validation', 'test', 'attack'], required=True)
    d.add_argument('--indices', help='JSON integer list; choose disjoint reserve indices BEFORE training')
    d.add_argument('--download', action='store_true')
    d.add_argument('--limit', type=int, default=0)
    d.add_argument('--output', required=True)
    e = sub.add_parser('embed', help='Independent recipient copies; D by default, E with --robust-weight')
    model_args(e)
    e.add_argument('--key', required=True)
    e.add_argument('--base', required=True)
    e.add_argument('--defense-data', required=True)
    e.add_argument('--validation-data', required=True)
    e.add_argument('--contribution-data')
    e.add_argument('--output-dir', required=True)
    e.add_argument('--recipients', type=int, nargs='+', help='Optional zero-based subset of enrolled recipients')
    defaults = EmbedConfig()
    for name in ['epochs', 'robust_every', 'train_variants', 'validation_variants', 'seed', 'max_steps']:
        e.add_argument('--'+name.replace('_', '-'), type=int, default=getattr(defaults, name))
    for name in ['lr', 'task_weight', 'fingerprint_weight', 'safety_weight', 'contribution_weight',
                 'robust_weight', 'inner_lr', 'max_accuracy_drop']:
        e.add_argument('--'+name.replace('_', '-'), type=float, default=getattr(defaults, name))
    c = sub.add_parser('calibrate', help='Fit score AND margin thresholds using whole-model controls')
    key_args(c)
    model_args(c)
    c.add_argument('--manifest', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--max-false-rate', type=float, default=.01)
    c.add_argument('--min-recall', type=float, default=.8)
    t = sub.add_parser('trace', help='Label-only verification: local, HTTP or offline returned labels')
    key_args(t)
    model_args(t)
    t.add_argument('--thresholds', required=True)
    source = t.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--endpoint')
    source.add_argument('--responses', help='JSON with key_id, query_sha256 and labels in query order')
    t.add_argument('--token-env', help='Name of environment variable containing optional API bearer token')
    t.add_argument('--timeout', type=float, default=30)
    t.add_argument('--output', required=True)
    q = sub.add_parser('export-queries', help='Write private test PNGs; only send to a suspect service for verification')
    key_args(q)
    q.add_argument('--output-dir', required=True)
    v = sub.add_parser('evaluate', help='Held-out trace matrix and paired safety metrics')
    key_args(v)
    model_args(v)
    v.add_argument('--manifest', required=True)
    v.add_argument('--thresholds', required=True)
    v.add_argument('--test-data')
    v.add_argument('--base')
    v.add_argument('--contribution-data')
    v.add_argument('--original-triggers')
    v.add_argument('--max-samples', type=int, default=0)
    v.add_argument('--output', required=True)
    a = sub.add_parser('attack', help='Generate modified copy for evaluation (does not overwrite source)')
    model_args(a)
    a.add_argument('--checkpoint', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--method', choices=['finetune', 'prune', 'weight-round'], required=True)
    a.add_argument('--attack-data')
    a.add_argument('--amount', type=float, default=.2)
    a.add_argument('--epochs', type=int, default=1)
    a.add_argument('--lr', type=float, default=.001)
    a.add_argument('--seed', type=int, default=7)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1 or getattr(args, 'batch_size', 1) < 1:
        raise ValueError('threads and batch-size must be positive')
    torch.set_num_threads(args.threads)
    if getattr(args, 'device', 'cpu').startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use --device cpu')
    if args.command == 'keys':
        key = write_key(args.output, recipients=args.recipients, bits=args.bits, distance=args.distance,
                        seed=args.seed, patch_size=args.patch_size)
        result = {'key_id': key['key_id'], 'min_distance': key['min_distance'], 'server_only': True}
    elif args.command == 'prepare-data':
        result = prepare_data(args)
    elif args.command == 'embed':
        cfg = EmbedConfig(**{name: getattr(args, name) for name in EmbedConfig.__dataclass_fields__})
        result = personalize(args.base, read_key(args.key), args.defense_data, args.validation_data,
                             args.output_dir, cfg, args.arch, args.device, args.contribution_data, args.recipients)
        result = {'copies_written': len(result['records']), 'server_registry': args.output_dir+'/server_private/registry.json'}
    elif args.command == 'calibrate':
        key = read_key(args.key)
        report = score_manifest(args.manifest, key, 'calibration', args.variants,
                                args.arch, args.device, args.batch_size)
        result = calibrate(report, key, args.max_false_rate, args.min_recall)
        save_json(args.output, result)
        save_json(str(Path(args.output).with_suffix('.scores.json')), report)
    elif args.command == 'trace':
        key = read_key(args.key)
        thresholds = read_thresholds(args.thresholds, key, args.variants)
        images, bits, digest = queries(key, 'test', args.variants)
        if args.checkpoint:
            labels = predict(load_model(args.checkpoint, args.arch, args.device), images, args.batch_size, args.device)
        elif args.endpoint:
            token = os.environ[args.token_env] if args.token_env else None
            labels = http_labels(args.endpoint, images, args.batch_size, args.timeout, token)
        else:
            response = read_json(args.responses)
            if response['key_id'] != key['key_id'] or response['query_sha256'] != digest:
                raise ValueError('Response belongs to different key, query order or query budget')
            labels = validate_labels(response['labels'], len(images))
        result = decide(scores_for_labels(key, labels, bits), thresholds['tau'], thresholds['delta'])
        result.update(key_id=key['key_id'], query_sha256=digest, queries=len(images))
        if result['recipient'] is not None:
            result['recipient_id'] = key['recipient_ids'][result['recipient']]
        save_json(args.output, result)
    elif args.command == 'export-queries':
        result = export_queries(read_key(args.key), args.output_dir, args.variants)
    elif args.command == 'evaluate':
        result = evaluate(args.manifest, read_key(args.key), args.thresholds, args.output, args.variants,
                          args.arch, args.device, args.batch_size, args.test_data, args.base,
                          args.contribution_data, args.original_triggers, args.max_samples)
        result = {'counts': result['counts'], 'rates': result['rates'], 'output': args.output}
    else:
        if Path(args.output).exists():
            raise ValueError('Attack output already exists; use a new path')
        modify_model(args.checkpoint, args.output, args.method, args.arch, args.device, args.amount,
                     args.attack_data, args.epochs, args.lr, args.batch_size, args.seed)
        result = {'method': args.method, 'output': args.output}
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result
