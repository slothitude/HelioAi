#!/usr/bin/env python3
"""Hunyuan3D-2.1 minimal API server for RTX 3060 6GB.

Uses CPU offloading to fit in limited VRAM.
Shape generation only by default (texture needs too much VRAM).

Run from C:\AI\Hunyuan3D-2.1 with the venv python:
  venv\Scripts\python.exe hy3d_server.py
"""

import os
import sys
import json
import uuid
import time
import base64
import argparse
import tempfile
import threading
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler

# Must run from repo root
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "hy3dshape"))
sys.path.insert(0, os.path.join(REPO_DIR, "hy3dpaint"))

os.environ["HF_HUB_CACHE"] = r"C:\AI\hf_cache"
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import trimesh
from PIL import Image

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover

PORT = int(os.environ.get("PORT", 8081))

# ── Model Management ──────────────────────────────────────────────────────────

model_lock = threading.Lock()
_pipeline = None
_rembg = None


def load_model():
    global _pipeline, _rembg
    if _pipeline is not None:
        return _pipeline, _rembg

    print("Loading Hunyuan3D-2.1 shape pipeline...", flush=True)

    # Download model files first via huggingface_hub
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(
        "tencent/Hunyuan3D-2.1",
        allow_patterns=["hunyuan3d-dit-v2-1/*"],
        cache_dir=os.environ.get("HF_HUB_CACHE", None),
    )
    print(f"Model downloaded to {model_dir}", flush=True)

    _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1",
        subfolder="hunyuan3d-dit-v2-1",
    )
    # Enable CPU offload to fit in 6GB VRAM
    _pipeline.enable_model_cpu_offload()
    print("Shape pipeline loaded (CPU offload mode)", flush=True)

    print("Loading background remover...", flush=True)
    _rembg = BackgroundRemover()
    print("Ready!", flush=True)
    return _pipeline, _rembg


# ── Helpers ────────────────────────────────────────────────────────────────────

def _generate(image_bytes, seed=1234, steps=5, octree_res=256,
              guidance=5.0, num_chunks=8000, remove_bg=True, gen_texture=False):
    """Generate a 3D GLB from image bytes. Returns GLB bytes."""
    pipeline, rembg = load_model()

    # Load image
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")

    # Remove background if needed
    if remove_bg and image.mode != "RGBA":
        image = rembg(image)

    # Generate mesh
    print(f"Generating mesh (steps={steps}, octree={octree_res})...", flush=True)
    t0 = time.time()
    mesh = pipeline(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        octree_resolution=octree_res,
        num_chunks=num_chunks,
        output_type="trimesh",
    )[0]
    print(f"Mesh generated in {time.time()-t0:.1f}s", flush=True)

    # Export to GLB
    buf = BytesIO()
    mesh.export(buf, file_type="glb")
    buf.seek(0)
    torch.cuda.empty_cache()
    return buf.read()


# ── HTTP Handler ───────────────────────────────────────────────────────────────

class HY3DHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            self._json(200, {"status": "ok", "model": "hunyuan3d-2.1"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        ct = self.headers.get("Content-Type", "")
        path = self.path.split("?")[0]

        if path == "/generate":
            self._handle_generate(body)
        else:
            self._json(404, {"error": "not found"})

    def _handle_generate(self, body):
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        image_b64 = data.get("image")
        if not image_b64:
            return self._json(400, {"error": "image (base64) is required"})

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return self._json(400, {"error": "invalid base64 image"})

        seed = data.get("seed", 1234)
        steps = data.get("num_inference_steps", 5)
        octree_res = data.get("octree_resolution", 256)
        guidance = data.get("guidance_scale", 5.0)
        num_chunks = data.get("num_chunks", 8000)
        remove_bg = data.get("remove_background", True)
        gen_texture = data.get("texture", False)

        with model_lock:
            try:
                glb_bytes = _generate(
                    image_bytes, seed=seed, steps=steps, octree_res=octree_res,
                    guidance=guidance, num_chunks=num_chunks,
                    remove_bg=remove_bg, gen_texture=gen_texture,
                )
                self.send_response(200)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header("Content-Length", str(len(glb_bytes)))
                self.end_headers()
                self.wfile.write(glb_bytes)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json(500, {"error": str(e)})

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        pass


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    # Pre-load model at startup
    load_model()

    print(f"Hunyuan3D-2.1 server on :{args.port}", flush=True)
    HTTPServer(("0.0.0.0", args.port), HY3DHandler).serve_forever()
