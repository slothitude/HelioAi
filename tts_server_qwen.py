#!/usr/bin/env python3
"""Qwen3-TTS Server - Text-to-speech with predefined speakers and voice cloning.

Runs on lappy-server (Windows, RTX 3060 6GB).
Two models, swapped dynamically (only one in VRAM at a time):
  - CustomVoice: predefined speakers (Vivian, Ryan, etc.) for /tts
  - Base: voice cloning from reference audio for /tts/clone, /tts/clone-url
"""

import os
import re
import sys
import json
import struct
import uuid
import tempfile
import threading
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_CACHE"] = r"C:\Users\aaron\hotswap\hf_cache"
import warnings
warnings.filterwarnings("ignore")

PORT = int(os.environ.get("PORT", 8006))

CUSTOM_VOICE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_SPEAKER = "Ryan"
DEFAULT_LANGUAGE = "English"

SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]
LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean",
    "Spanish", "French", "German", "Portuguese",
    "Russian", "Arabic", "Auto",
]

# ── Model Management ───────────────────────────────────────────────────────────

model_lock = threading.Lock()
_current_model = None
_current_type = None  # "custom" or "base"


def _unload():
    global _current_model, _current_type
    if _current_model is not None:
        del _current_model
        _current_model = None
        import torch
        torch.cuda.empty_cache()
        _current_type = None


def _load(model_type):
    global _current_model, _current_type
    if _current_type == model_type:
        return _current_model
    _unload()
    import torch
    from qwen_tts import Qwen3TTSModel

    name = CUSTOM_VOICE_MODEL if model_type == "custom" else BASE_MODEL
    print(f"Loading {name}...", flush=True)
    _current_model = Qwen3TTSModel.from_pretrained(
        name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    _current_type = model_type
    print(f"Loaded {name}", flush=True)
    return _current_model


def ensure_custom():
    with model_lock:
        return _load("custom")


def ensure_base():
    with model_lock:
        return _load("base")


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _wav_bytes(audio_np, sr):
    import io, soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _parse_multipart(body, content_type):
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


def _decode(val):
    return val.decode(errors="replace") if isinstance(val, bytes) else val


def _split_sentences(text, group_size=3):
    """Split text into sentences at . ! ? and newlines, then group into chunks."""
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    groups = []
    for i in range(0, len(parts), group_size):
        groups.append(" ".join(parts[i:i + group_size]))
    return groups


# ── Voice clone prompt cache (avoids re-processing ref audio) ────────────────

_prompt_cache = {}  # key: hash of audio bytes -> prompt items


def _get_or_create_prompt(model, ref_audio_path, ref_text, xvec_only):
    with open(ref_audio_path, "rb") as f:
        key = hash(f.read())
    if key not in _prompt_cache:
        _prompt_cache.clear()  # keep only 1 entry (save RAM)
        _prompt_cache[key] = model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            x_vector_only_mode=xvec_only,
        )
    return _prompt_cache[key]


# ── HTTP Handler ────────────────────────────────────────────────────────────────

class TTSHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            self._json(200, {
                "status": "ok",
                "model": _current_type,
            })
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
        elif path == "/tts/clone-stream":
            self._handle_clone_stream(body, ct)
        elif path == "/tts/clone-url":
            self._handle_clone_url(body)
        else:
            self._json(404, {"error": "not found"})

    # ── Predefined speaker TTS ─────────────────────────────────────────────

    def _handle_tts(self, body):
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        text = data.get("text", "")
        if not text:
            return self._json(400, {"error": "text is required"})

        speaker = data.get("speaker", DEFAULT_SPEAKER)
        language = data.get("language", DEFAULT_LANGUAGE)

        try:
            model = ensure_custom()
            wavs, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
            )
            self._send_audio(wavs[0], sr)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    # ── Voice Clone (file upload) ───────────────────────────────────────────

    def _handle_clone(self, body, ct):
        fields = _parse_multipart(body, ct)
        text = _decode(fields.get("text", ""))
        if not text:
            return self._json(400, {"error": "text is required"})

        audio_data = fields.get("file")
        if not audio_data:
            return self._json(400, {"error": "file is required"})

        language = _decode(fields.get("language", DEFAULT_LANGUAGE))
        ref_text = _decode(fields.get("ref_text", ""))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            ref_path = f.name

        try:
            model = ensure_base()
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_path,
                ref_text=ref_text if ref_text else None,
                x_vector_only_mode=(not ref_text),
            )
            self._send_audio(wavs[0], sr)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json(500, {"error": str(e)})
        finally:
            try:
                os.unlink(ref_path)
            except OSError:
                pass

    # ── Voice Clone (URL) ───────────────────────────────────────────────────

    def _handle_clone_url(self, body):
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        text = data.get("text", "")
        ref_url = data.get("ref_audio_url", "")
        if not text or not ref_url:
            return self._json(400, {"error": "text and ref_audio_url are required"})

        language = data.get("language", DEFAULT_LANGUAGE)
        ref_text = data.get("ref_text", "")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            try:
                with urllib.request.urlopen(ref_url, timeout=30) as r:
                    f.write(r.read())
                ref_path = f.name
            except Exception as e:
                return self._json(400, {"error": f"download failed: {e}"})

        try:
            model = ensure_base()
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_path,
                ref_text=ref_text if ref_text else None,
                x_vector_only_mode=(not ref_text),
            )
            self._send_audio(wavs[0], sr)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json(500, {"error": str(e)})
        finally:
            try:
                os.unlink(ref_path)
            except OSError:
                pass

    # ── Voice Clone Streaming (sentence-level chunks) ──────────────────────

    def _handle_clone_stream(self, body, ct):
        fields = _parse_multipart(body, ct)
        text = _decode(fields.get("text", ""))
        if not text:
            return self._json(400, {"error": "text is required"})

        audio_data = fields.get("file")
        if not audio_data:
            return self._json(400, {"error": "file is required"})

        language = _decode(fields.get("language", DEFAULT_LANGUAGE))
        ref_text = _decode(fields.get("ref_text", ""))
        xvec_only = not ref_text

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            ref_path = f.name

        try:
            model = ensure_base()
            sentences = _split_sentences(text)
            if not sentences:
                return self._json(400, {"error": "no sentences found in text"})

            prompt = _get_or_create_prompt(model, ref_path, ref_text, xvec_only)

            # Send streaming response headers
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            for sentence in sentences:
                wavs, sr = model.generate_voice_clone(
                    text=sentence,
                    language=language,
                    voice_clone_prompt=prompt,
                )
                wav_bytes = _wav_bytes(wavs[0], sr)
                self.wfile.write(struct.pack('<I', len(wav_bytes)))
                self.wfile.write(wav_bytes)
                self.wfile.flush()

            # End marker
            self.wfile.write(struct.pack('<I', 0))
            self.wfile.flush()
        except Exception as e:
            import traceback
            traceback.print_exc()
            # If headers already sent, we can't send a JSON error
            try:
                self._json(500, {"error": str(e)})
            except Exception:
                pass
        finally:
            try:
                os.unlink(ref_path)
            except OSError:
                pass

    # ── Audio response ──────────────────────────────────────────────────────

    def _send_audio(self, audio_np, sr):
        data = _wav_bytes(audio_np, sr)
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
    print(f"Qwen3-TTS Server on :{PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), TTSHandler).serve_forever()
