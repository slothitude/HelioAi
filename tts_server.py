#!/usr/bin/env python3
"""CosyVoice2 TTS REST Server.

Two modes with lazy model loading (~3.5GB VRAM):
  - Zero-shot voice cloning: clone from reference audio
  - Cross-lingual TTS: text-to-speech with a reference voice

First request loads the model (~10s). Subsequent requests are fast.
"""

import os
import sys
import json
import uuid
import tempfile
import threading
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Add CosyVoice to path
sys.path.insert(0, r"C:\Users\aaron\hotswap\CosyVoice")

PORT = int(os.environ.get("PORT", 8006))

MODEL_NAME = "CosyVoice2-0.5B"
MODEL_DIR = Path(r"C:\Users\aaron\hotswap\models\iic\CosyVoice2-0___5B")

REF_DIR = Path(r"C:\Users\aaron\hotswap\tts_references")
REF_DIR.mkdir(parents=True, exist_ok=True)

SPEAKERS = {
    "Vivian":   "Warm, friendly female voice",
    "Serena":   "Clear, professional female voice",
    "Uncle_Fu": "Deep, wise male voice",
    "Dylan":    "Casual, energetic male voice",
    "Eric":     "Calm, measured male voice",
    "Ryan":     "Confident, articulate male voice",
    "Aiden":    "Young, enthusiastic male voice",
    "Ono_Anna": "Soft, gentle Japanese female voice",
    "Sohee":    "Bright, melodic Korean female voice",
}

LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto",
]

# ── Model Management ───────────────────────────────────────────────────────────

model_lock = threading.Lock()
model_instance = None


def _load_model():
    """Load CosyVoice2 model."""
    global model_instance
    with model_lock:
        if model_instance is not None:
            return model_instance

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
            model_path = str(MODEL_DIR)
            print(f"Loading CosyVoice2 from {model_path}...", flush=True)
            model_instance = CosyVoice2(model_path)
            print("CosyVoice2 loaded.", flush=True)
            return model_instance
        except ImportError:
            raise RuntimeError(
                "cosyvoice not installed. Clone https://github.com/FunAudioLLM/CosyVoice "
                "and install: pip install -r requirements.txt"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")


def _synthesize(model, text, ref_audio=None, ref_text=""):
    """Run TTS synthesis using CosyVoice2. Returns audio tensor."""
    import torch

    if ref_audio:
        # Zero-shot voice cloning
        gen = model.inference_zero_shot(
            tts_text=text,
            prompt_text=ref_text,
            prompt_wav=ref_audio,
        )
    else:
        # Cross-lingual mode - needs a reference wav
        # Use the first available reference or generate without reference
        # For now, use zero-shot with empty prompt
        gen = model.inference_zero_shot(
            tts_text=text,
            prompt_text="",
            prompt_wav=ref_audio,
        )

    # Collect all chunks
    results = list(gen)
    if not results:
        raise RuntimeError("No audio generated")

    # Concatenate if multiple chunks
    speeches = [r["tts_speech"] for r in results]
    full_speech = torch.cat(speeches, dim=1)
    return full_speech


# ── HTTP Handler ────────────────────────────────────────────────────────────────

class TTSHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            self._json(200, {"status": "ok", "loaded_model": MODEL_NAME if model_instance else None})
        elif path == "/speakers":
            self._json(200, {"speakers": SPEAKERS})
        elif path == "/languages":
            self._json(200, {"languages": LANGUAGES})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        ct = self.headers.get("Content-Type", "")
        path = self.path.split("?")[0]

        if path == "/tts":
            self._handle_tts(body)
        elif path == "/tts/clone":
            self._handle_clone(body, ct)
        elif path == "/tts/clone-url":
            self._handle_clone_url(body)
        elif path == "/upload-reference":
            self._handle_upload_ref(body, ct)
        else:
            self._json(404, {"error": "not found"})

    # ── TTS (requires reference audio for zero-shot) ────────────────────────

    def _handle_tts(self, body):
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        text = data.get("text", "")
        speaker = data.get("speaker", "")
        language = data.get("language", "Auto")

        if not text:
            return self._json(400, {"error": "text is required"})

        # CosyVoice2 requires a reference audio for zero-shot
        # Check if speaker matches an uploaded reference
        ref_path = REF_DIR / f"{speaker}.wav" if speaker else None
        if ref_path and not ref_path.exists():
            # No reference audio for this speaker
            return self._json(400, {
                "error": f"No reference audio for speaker '{speaker}'. "
                         f"Upload one via /upload-reference first, or use /tts/clone."
            })

        try:
            model = _load_model()
            audio = _synthesize(model, text, ref_audio=str(ref_path) if ref_path else None)
            self._send_audio(audio)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    # ── Voice Clone ────────────────────────────────────────────────────────

    def _handle_clone(self, body, ct):
        fields = self._parse_multipart(body, ct)
        text = self._decode_field(fields.get("text", ""))
        language = self._decode_field(fields.get("language", "Auto"))
        ref_text = self._decode_field(fields.get("ref_text", ""))
        audio_data = fields.get("file")

        if not text or not audio_data:
            return self._json(400, {"error": "text and file are required"})

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            ref_path = f.name

        try:
            model = _load_model()
            audio = _synthesize(model, text, ref_audio=ref_path, ref_text=ref_text)
            self._send_audio(audio)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json(500, {"error": str(e)})
        finally:
            os.unlink(ref_path)

    def _handle_clone_url(self, body):
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        text = data.get("text", "")
        language = data.get("language", "Auto")
        ref_url = data.get("ref_audio_url", "")
        ref_text = data.get("ref_text", "")

        if not text or not ref_url:
            return self._json(400, {"error": "text and ref_audio_url are required"})

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            try:
                with urllib.request.urlopen(ref_url, timeout=30) as r:
                    f.write(r.read())
                ref_path = f.name
            except Exception as e:
                return self._json(400,
                                  {"error": f"download ref audio failed: {e}"})

        try:
            model = _load_model()
            audio = _synthesize(model, text, ref_audio=ref_path, ref_text=ref_text)
            self._send_audio(audio)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json(500, {"error": str(e)})
        finally:
            os.unlink(ref_path)

    # ── Upload Reference ───────────────────────────────────────────────────

    def _handle_upload_ref(self, body, ct):
        fields = self._parse_multipart(body, ct)
        audio_data = fields.get("file")
        if not audio_data:
            return self._json(400, {"error": "file is required"})

        speaker_id = uuid.uuid4().hex[:12]
        ref_path = REF_DIR / f"{speaker_id}.wav"
        with open(ref_path, "wb") as f:
            f.write(audio_data)

        self._json(200, {"speaker_id": speaker_id, "path": str(ref_path)})

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_field(val):
        """Decode bytes to str for multipart text fields."""
        if isinstance(val, bytes):
            return val.decode(errors="replace")
        return val

    @staticmethod
    def _parse_multipart(body, content_type):
        """Simple multipart form data parser."""
        fields = {}
        if "boundary=" not in content_type:
            return fields
        boundary = content_type.split("boundary=")[-1].strip()
        if boundary.startswith('"') and boundary.endswith('"'):
            boundary = boundary[1:-1]
        boundary = boundary.encode()

        parts = body.split(b"--" + boundary)
        for part in parts:
            if not part or part.strip() in (b"", b"--", b"--\r\n"):
                continue
            header_end = part.find(b"\r\n\r\n")
            if header_end == -1:
                continue
            headers_raw = part[:header_end].decode(errors="replace")
            part_body = part[header_end + 4:]
            if part_body.endswith(b"\r\n"):
                part_body = part_body[:-2]

            name = None
            for line in headers_raw.split("\r\n"):
                if "name=" in line:
                    for seg in line.split(";"):
                        seg = seg.strip()
                        if seg.startswith("name="):
                            name = seg.split("=", 1)[1].strip('"')
                            break
            if name:
                fields[name] = part_body
        return fields

    def _send_audio(self, audio):
        """Send audio as WAV response using soundfile."""
        import soundfile as sf
        import io as _io
        import numpy as np

        buf = _io.BytesIO()
        audio_np = audio.squeeze().cpu().numpy()
        sf.write(buf, audio_np, 24000, format='WAV')
        buf.seek(0)
        data = buf.read()

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        pass


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"TTS Server on :{PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), TTSHandler).serve_forever()
