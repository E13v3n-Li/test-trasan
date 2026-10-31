"""Train an independent, unwatermarked CIFAR-10 control for OFFLINE research evaluation.

Uses the public CIFAR-10 training set; not an assertion that an FL server can access private data.
Use distinct seeds for calibration and final unknown controls. Report their clean accuracy.
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from traceability.common import MEAN, STD, build_model, save_model, save_json, seed_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='data/cifar')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=.1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--download', action='store_true')
    args = p.parse_args()
    if Path(args.output).exists():
        raise ValueError('Output already exists')
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('Invalid training parameters')
    seed_all(args.seed)
    torch.set_num_threads(4)
    augment = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train = datasets.CIFAR10(args.data_root, train=True, transform=augment, download=args.download)
    val = datasets.CIFAR10(args.data_root, train=True, transform=plain, download=False)
    perm = torch.randperm(len(train)).tolist()
    val = DataLoader(Subset(val, perm[:5000]), batch_size=args.batch_size)
    train = DataLoader(Subset(train, perm[5000:]), batch_size=args.batch_size, shuffle=True)
    model = build_model('resnet18').to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best, history, started = -1., [], time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x.to(args.device)), y.to(args.device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            for x, y in val:
                correct += (model(x.to(args.device)).argmax(1).cpu() == y).sum().item()
                total += len(y)
        accuracy = correct/total
        if accuracy > best:
            best = accuracy
            save_model(args.output, model, 'resnet18')
        history.append({'epoch': epoch+1, 'validation_accuracy': accuracy})
        print(history[-1], flush=True)
    save_json(str(args.output)+'.json', dict(seed=args.seed, history=history,
              seconds=time.perf_counter()-started, purpose='Independent unwatermarked public-CIFAR10 control'))


if __name__ == '__main__':
    main()
