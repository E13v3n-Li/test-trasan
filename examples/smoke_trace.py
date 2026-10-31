"""Offline CPU plumbing demo on SYNTHETIC data. Not a Sanitizer/CIFAR experiment."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from traceability.common import (TinySmokeNet, data_bundle, load_model, normalize, save_json,
                                save_model, save_private, seed_all)
from traceability.keys import write_key
from traceability.training import EmbedConfig, personalize
from traceability.verification import score_manifest, calibrate
from traceability.evaluation import evaluate, modify_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', default='trace_runs/synthetic_smoke')
    args = p.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    seed_all(17)
    base = TinySmokeNet().eval()
    save_model(root/'base.pt', base, 'tiny_smoke')
    for role, count in [('defense', 128), ('validation', 32), ('test', 12), ('attack', 32)]:
        x = normalize(torch.rand(count, 3, 32, 32))
        with torch.no_grad():
            y = base(x).argmax(1)
        save_private(root/f'{role}.pt', data_bundle(x, y, role, source=f'synthetic/{role}'))
    key = write_key(root/'secret.pt', recipients=2, bits=8, distance=2, seed=19)
    cfg = EmbedConfig(epochs=35, batch_size=32, lr=.02, max_accuracy_drop=1.,
                      train_variants=8, validation_variants=4, safety_weight=1.)
    personalize(root/'base.pt', key, root/'defense.pt', root/'validation.pt', root/'copies',
                cfg, arch='tiny_smoke')
    known = [dict(name=f'copy_{i}', path=f'copies/distribution/copy_{i:03d}/model.pt', recipient=i)
             for i in range(2)]
    unknowns = []
    for i in range(4):
        seed_all(100+i)
        save_model(root/f'unknown_{i}.pt', TinySmokeNet(), 'tiny_smoke')
        unknowns.append(dict(name=f'unknown_{i}', path=f'unknown_{i}.pt', recipient=None))
    save_json(root/'calibration.json', dict(records=known+unknowns[:2]))
    save_json(root/'evaluation.json', dict(records=known+unknowns[2:]))
    calibration = score_manifest(root/'calibration.json', key, 'calibration', 4, 'tiny_smoke')
    thresholds = calibrate(calibration, key, max_false_rate=0, min_recall=1.)
    save_json(root/'thresholds.json', thresholds)
    report = evaluate(root/'evaluation.json', key, root/'thresholds.json', root/'result.json',
                      arch='tiny_smoke', test_data=root/'test.pt', base_path=root/'base.pt')
    assert report['counts']['correct'] == 2
    for method in ['finetune', 'prune', 'weight-round']:
        modify_model(root/'copies/distribution/copy_000/model.pt', root/f'{method}.pt',
                     method, arch='tiny_smoke', data_path=root/'attack.pt')
    save_json(root/'SMOKE_ONLY.json', dict(synthetic_pipeline_only=True, counts=report['counts'],
        warning='Random/synthetic controls are only plumbing tests. No empirical CIFAR harmlessness or robustness claim.'))
    print(f'Synthetic pipeline completed: {root / "SMOKE_ONLY.json"}')


if __name__ == '__main__':
    main()
