#!/usr/bin/env python3
"""Simple Whisper STT server using faster-whisper.

Endpoints:
  GET  /health         - health check
  POST /v1/audio/transcriptions - multipart file upload, returns transcription
"""

import os
import io
import json
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = int(os.environ.get("PORT", 8005))
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# Lazy-load model
_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...", flush=True)
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Whisper model loaded.", flush=True)
    return _model


class WhisperHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/health":
            loaded = _model is not None
            self._json(200, {
                "status": "ok",
                "model": MODEL_SIZE,
                "loaded": loaded,
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/audio/transcriptions":
            self._transcribe()
        else:
            self._json(404, {"error": "not found"})

    def _transcribe(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        ct = self.headers.get("Content-Type", "")

        # Parse multipart
        fields = self._parse_multipart(body, ct)
        audio_data = fields.get("file")

        if not audio_data:
            return self._json(400, {"error": "no file uploaded"})

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name

        try:
            model = get_model()
            segments, info = model.transcribe(tmp_path, beam_size=5)
            text = " ".join(seg.text for seg in segments).strip()
            self._json(200, {"text": text})
        except Exception as e:
            self._json(500, {"error": str(e)})
        finally:
            os.unlink(tmp_path)

    @staticmethod
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

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print(f"Whisper STT server on :{PORT} (model={MODEL_SIZE}, device={DEVICE})", flush=True)
    HTTPServer(("0.0.0.0", PORT), WhisperHandler).serve_forever()
