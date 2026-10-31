"""Local protocol example, not a production web server. Only needs a delivered model, never the key."""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from traceability.common import load_model, normalize, predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--arch', choices=['resnet18', 'tiny_smoke'], default='resnet18')
    p.add_argument('--device', default='cpu')
    p.add_argument('--port', type=int, default=8765)
    args = p.parse_args()
    torch.set_num_threads(2)
    model = load_model(args.checkpoint, args.arch, args.device)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != '/predict':
                self.send_error(404)
                return
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8*1024*1024:
                self.send_error(413)
                return
            try:
                values = json.loads(self.rfile.read(length))['images']
                if not isinstance(values, list) or not 1 <= len(values) <= 256:
                    raise ValueError('Batch must contain 1..256 images')
                images = []
                for encoded in values:
                    image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
                    if image.size != (32, 32):
                        raise ValueError('Expected 32x32 images')
                    images.append(normalize(torch.from_numpy(np.array(image.convert('RGB'))).permute(2, 0, 1).float()/255))
                labels = predict(model, torch.stack(images), device=args.device)
                body = json.dumps({'labels': labels.tolist()}).encode()
            except (ValueError, KeyError, TypeError, OSError):
                self.send_error(400, 'Invalid query payload')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Local label API: http://127.0.0.1:{args.port}/predict', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
