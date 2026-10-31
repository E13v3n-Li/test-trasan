"""Private common probes and recipient codebook, separate from contribution keys."""
import hashlib
import secrets

import torch

from .common import normalize, load_tensors, save_private


def _generator(secret, purpose):
    digest = hashlib.sha256(f'{secret}:{purpose}'.encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], 'big') % (2**63 - 1))


def generate_key(recipients=10, bits=64, distance=20, seed=None, patch_size=4):
    if not (2 <= recipients <= 1000 and 2 <= bits <= 4096 and 1 <= distance <= bits):
        raise ValueError('Invalid recipient count, code length or minimum distance')
    if not 1 <= patch_size <= 8:
        raise ValueError('Patch size must be between 1 and 8')
    secret = secrets.token_hex(32) if seed is None else f'research-seed-{seed}'
    gen = _generator(secret, 'codebook')
    codes = []
    for _ in range(100000):
        candidate = torch.randint(2, (bits,), generator=gen)
        # Avoid low-entropy codewords; distinguish every recipient pair.
        if not bits * .25 <= candidate.sum() <= bits * .75:
            continue
        if all((candidate != c).sum() >= distance for c in codes):
            codes.append(candidate)
            if len(codes) == recipients:
                break
    if len(codes) != recipients:
        raise ValueError('Cannot construct this codebook; increase bits or reduce distance/recipients')
    codes = torch.stack(codes)
    # Permuting columns preserves distances; avoid common output-class shortcuts.
    gen = _generator(secret, 'probes')
    pairs = torch.stack([torch.randperm(10, generator=gen)[:2] for _ in range(bits)])
    key = dict(version=1, secret=secret, codebook=codes, label_pairs=pairs,
               recipient_ids=[f'client_{i:03d}' for i in range(recipients)],
               backgrounds=.15 + .7 * torch.rand(bits, 3, 1, 1, generator=gen),
               patches=torch.randint(2, (bits, 3, patch_size, patch_size), generator=gen).float(),
               positions=torch.randint(1, 32-patch_size, (bits, 2), generator=gen),
               min_distance=int(min((codes[i] != codes[j]).sum().item()
                                    for i in range(recipients) for j in range(i))),
               research_seed=seed)
    key['key_id'] = hashlib.sha256(secret.encode()).hexdigest()[:24]
    return key


def write_key(path, **kwargs):
    key = generate_key(**kwargs)
    save_private(path, key)
    return key


def read_key(path):
    key = load_tensors(path)
    if key.get('version') != 1:
        raise ValueError('Unsupported key format')
    return key


def overlay(images, key, bit_ids, shift=(0, 0), opacity=1.):
    """Apply raw-RGB patches to normalized natural or artificial images."""
    result = images.clone()
    for row, bit in enumerate(bit_ids.tolist()):
        patch = normalize(key['patches'][bit]).to(images.device)
        size = patch.shape[-1]
        top, left = key['positions'][bit].tolist()
        top = max(0, min(32-size, top+shift[0]))
        left = max(0, min(32-size, left+shift[1]))
        region = result[row, :, top:top+size, left:left+size]
        result[row, :, top:top+size, left:left+size] = (1-opacity)*region + opacity*patch
    return result


def queries(key, split='test', variants=4):
    if split not in {'train', 'calibration', 'test'} or variants < 1:
        raise ValueError('Invalid query split or variants')
    bits = key['codebook'].shape[1]
    images, bit_ids = [], []
    for bit in range(bits):
        for variant in range(variants):
            gen = _generator(key['secret'], f'{split}/{bit}/{variant}')
            color = key['backgrounds'][bit] + .04 * torch.randn(3, 1, 1, generator=gen)
            raw = (color + .015 * torch.randn(3, 32, 32, generator=gen)).clamp(0, 1)
            # Same pixels locally and over the PNG HTTP transport.
            raw = raw.mul(255).round().div(255)
            images.append(normalize(raw))
            bit_ids.append(bit)
    bit_ids = torch.tensor(bit_ids)
    images = overlay(torch.stack(images), key, bit_ids)
    digest = hashlib.sha256(images.numpy().tobytes()).hexdigest()
    return images, bit_ids, digest


def expected(key, recipient, bit_ids):
    codes = key['codebook'][recipient, bit_ids]
    return key['label_pairs'][bit_ids, codes]
