#!/usr/bin/env python3
"""Qwen3-TTS REST Server.

Two modes with lazy model swapping (~3.5GB VRAM each):
  - Custom Voice (default): preset speakers with optional instruct control
  - Voice Clone: clone from reference audio

Only one model in VRAM at a time. First request after a model swap
takes ~15s to load. Subsequent requests are fast.
"""

import os
import json
import uuid
import tempfile
import threading
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PORT = int(os.environ.get("PORT", 8006))

MODEL_CUSTOM = "Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_BASE = "Qwen3-TTS-12Hz-1.7B-Base"

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
loaded_model_name = None
model_instance = None


def _load_model(name):
    """Load a TTS model, swapping out the current one if needed."""
    global loaded_model_name, model_instance
    with model_lock:
        if loaded_model_name == name and model_instance is not None:
            return model_instance

        # Unload current
        model_instance = None
        loaded_model_name = None

        try:
            from qwen_tts import QwenTTS
            model_instance = QwenTTS.from_pretrained(name)
            loaded_model_name = name
            return model_instance
        except ImportError:
            pass

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                name, trust_remote_code=True,
                torch_dtype=torch.float16, device_map="auto",
            )
            model_instance = {"model": model, "tokenizer": tokenizer}
            loaded_model_name = name
            return model_instance
        except Exception as e:
            raise RuntimeError(f"Failed to load model {name}: {e}")


def _synthesize(model, text, speaker=None, language="Auto", instruct=None,
                ref_audio=None, ref_text=None):
    """Run TTS synthesis. Returns audio bytes (WAV) or numpy array."""
    # Try qwen_tts API
    if hasattr(model, "synthesize"):
        return model.synthesize(
            text=text, speaker=speaker, language=language,
            instruct=instruct, ref_audio=ref_audio, ref_text=ref_text,
        )
    if callable(model) and not isinstance(model, dict):
        return model(text=text, speaker=speaker, language=language)
    if isinstance(model, dict):
        raise RuntimeError(
            "Transformers-based synthesis not yet implemented. "
            "Install qwen_tts: pip install qwen-tts"
        )
    raise RuntimeError("No valid model loaded")


# ── HTTP Handler ────────────────────────────────────────────────────────────────

class TTSHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            self._json(200, {"status": "ok", "loaded_model": loaded_model_name})
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

    # ── Custom Voice TTS ───────────────────────────────────────────────────

    def _handle_tts(self, body):
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        text = data.get("text", "")
        speaker = data.get("speaker", "Ryan")
        language = data.get("language", "Auto")
        instruct = data.get("instruct")

        if not text:
            return self._json(400, {"error": "text is required"})

        try:
            model = _load_model(MODEL_CUSTOM)
            audio = _synthesize(model, text, speaker=speaker,
                                language=language, instruct=instruct)
            self._send_audio(audio)
        except Exception as e:
            self._json(500, {"error": str(e)})

    # ── Voice Clone ────────────────────────────────────────────────────────

    def _handle_clone(self, body, ct):
        fields = self._parse_multipart(body, ct)
        text = fields.get("text", "")
        language = fields.get("language", "Auto")
        ref_text = fields.get("ref_text", "")
        audio_data = fields.get("file")

        if not text or not audio_data:
            return self._json(400, {"error": "text and file are required"})

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            ref_path = f.name

        try:
            model = _load_model(MODEL_BASE)
            audio = _synthesize(model, text, language=language,
                                ref_audio=ref_path, ref_text=ref_text)
            self._send_audio(audio)
        except Exception as e:
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
            model = _load_model(MODEL_BASE)
            audio = _synthesize(model, text, language=language,
                                ref_audio=ref_path, ref_text=ref_text)
            self._send_audio(audio)
        except Exception as e:
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
        """Send audio as WAV response."""
        if isinstance(audio, bytes):
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)
        else:
            import soundfile as sf
            import io as _io
            buf = _io.BytesIO()
            sf.write(buf, audio, samplerate=24000, format="WAV")
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
