import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
import torch
from PIL import Image

from traceability.common import (TinySmokeNet, build_model, data_bundle, load_data, load_model,
                                normalize, save_json, save_model, save_private)
from traceability.keys import expected, generate_key, overlay, queries, write_key
from traceability.training import one_step_robust_loss
from traceability.verification import (calibrate, decide, http_labels, png_bytes,
                                      scores_for_labels, validate_labels)


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_default_codebook_and_secret_splits():
    key = generate_key(seed=11)
    assert key['min_distance'] >= 20
    q, bits, digest = queries(key, 'test', 2)
    q2, bits2, digest2 = queries(key, 'test', 2)
    train, _, train_digest = queries(key, 'train', 2)
    cal, _, cal_digest = queries(key, 'calibration', 2)
    assert torch.equal(q, q2) and torch.equal(bits, bits2) and digest == digest2
    assert len({digest, train_digest, cal_digest}) == 3
    assert not torch.equal(q, train) and not torch.equal(q, cal)
    for i in range(10):
        scores = scores_for_labels(key, expected(key, i, bits), bits)
        assert scores[i] == 1
        assert max(s for j, s in enumerate(scores) if j != i) <= 1-20/64


def test_secret_not_overwritten_and_keys_random(tmp_path):
    write_key(tmp_path/'key.pt')
    with pytest.raises(FileExistsError):
        write_key(tmp_path/'key.pt')
    assert generate_key()['key_id'] != generate_key()['key_id']


def test_png_transport_preserves_query_pixels():
    key = generate_key(2, 8, 2, seed=3)
    q, _, _ = queries(key, 'test', 1)
    image = Image.open(io.BytesIO(png_bytes(q[0]))).convert('RGB')
    restored = normalize(torch.from_numpy(np.array(image)).permute(2, 0, 1).float()/255)
    assert torch.allclose(restored, q[0], atol=1e-6)


def test_decoder_rejects_ties_and_unrelated_classes():
    assert decide([.9, .9], .8, .01)['recipient'] is None
    assert decide([.6, .2], .8, .1)['recipient'] is None
    assert decide([.95, .6], .8, .1)['recipient'] == 0
    key = generate_key(2, 8, 2, seed=4)
    _, bits, _ = queries(key, variants=1)
    other = torch.tensor([next(k for k in range(10) if k not in pair) for pair in key['label_pairs']])
    assert scores_for_labels(key, other, bits) == [0., 0.]
    for labels in [[0.1], [10], [-1], [True], [1, 2]]:
        with pytest.raises(ValueError):
            validate_labels(labels, 1)


def test_calibration_requires_controls_and_all_recipients():
    key = generate_key(2, 8, 2, seed=3)
    report = dict(key_id=key['key_id'], split='calibration', variants=4, query_sha256='test', rows=[
        dict(recipient=0, scores=[.95, .45], sha256='a'),
        dict(recipient=1, scores=[.4, .95], sha256='b'),
        dict(recipient=None, scores=[.7, .5], sha256='c')])
    c = calibrate(report, key, max_false_rate=0, min_recall=1)
    assert c['calibration_counts']['correct'] == 2
    assert c['calibration_counts']['false_attributed_unknown'] == 0
    assert decide([.7, .5], c['tau'], c['delta'])['recipient'] is None
    report['rows'].pop()
    with pytest.raises(ValueError, match='unknown control'):
        calibrate(report, key)


def test_tensor_roles_and_original_checkpoint_compatibility(tmp_path):
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
    save_private(tmp_path/'data.pt', data_bundle(x, y, 'test'))
    assert len(load_data(tmp_path/'data.pt', 'test')[0]) == 2
    with pytest.raises(ValueError):
        load_data(tmp_path/'data.pt', 'defense')
    model = build_model('resnet18').eval()
    torch.save(model.state_dict(), tmp_path/'upstream.pth')
    restored = load_model(tmp_path/'upstream.pth')
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))


def test_robust_loss_backpropagates_and_does_not_mutate_model():
    model = TinySmokeNet()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(3, 3, 32, 32)
    loss = one_step_robust_loss(model, x, torch.tensor([0, 1, 2]), x*.5, torch.tensor([3, 4, 5]), .01)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_http_label_only_protocol():
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            import base64
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            for value in payload['images']:
                assert Image.open(io.BytesIO(base64.b64decode(value))).size == (32, 32)
                seen.append(value)
            body = json.dumps({'labels': [3]*len(payload['images'])}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        q, _, _ = queries(generate_key(2, 8, 2, seed=1), variants=1)
        labels = http_labels(f'http://127.0.0.1:{server.server_port}', q, batch_size=3)
        assert labels.tolist() == [3]*8 and len(seen) == 8
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
