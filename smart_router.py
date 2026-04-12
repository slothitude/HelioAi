#!/usr/bin/env python3
"""Smart Router Gateway - Unified AI Services Proxy.

Routes requests to backend services with automatic GPU swapping.
Runs on lappy-server (Windows, RTX 3060 6GB).

GPU modes:
  text   -> llama.cpp  (port 8201, ~5GB VRAM)
  imggen -> ComfyUI    (port 8202, ~5GB VRAM)
  tts    -> Qwen3-TTS  (port 8006, ~3.5GB VRAM)
  Whisper runs on CPU (port 8005, no swap needed)
"""

import os
import sys
import json
import uuid
import time
import threading
import subprocess
import socket
import base64
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

PORT = int(os.environ.get("PORT", 4000))

BACKENDS = {
    "litellm":  os.environ.get("LITELLM_URL",  "http://localhost:4001"),
    "llamacpp": os.environ.get("LLAMACPP_URL", "http://localhost:8201"),
    "whisper":  os.environ.get("WHISPER_URL",  "http://localhost:8005"),
    "tts":      os.environ.get("TTS_URL",      "http://localhost:8006"),
    "comfyui":  os.environ.get("COMFYUI_URL",  "http://localhost:8202"),
}

BASE_DIR = Path(r"C:\Users\aaron\hotswap")
LLAMA_EXE = str(BASE_DIR / "llama-server.exe")
LLAMA_MODEL = str(Path(
    r"C:\Users\aaron\.ollama\models\blobs"
    r"\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
))
COMFYUI_DIR = str(BASE_DIR / "ComfyUI")
COMFYUI_PYTHON = str(Path(COMFYUI_DIR) / "venv" / "Scripts" / "python.exe")
TTS_PYTHON = str(BASE_DIR / "tts-venv" / "Scripts" / "python.exe")
TTS_SCRIPT = str(BASE_DIR / "tts_server.py")
WORKFLOWS_DIR = str(BASE_DIR / "workflows")

GPU_MODES = {
    "text": {
        "port": 8201,
        "health_path": "/health",
        "start_cmd": [
            LLAMA_EXE, "-m", LLAMA_MODEL, "-ngl", "99", "-c", "2048",
            "-t", "8", "--flash-attn", "auto", "--host", "0.0.0.0", "--port", "8201",
        ],
        "stop_ps": "Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue",
        "timeout": 90,
    },
    "imggen": {
        "port": 8202,
        "health_path": "/system_stats",
        "start_cmd": [
            COMFYUI_PYTHON, f"{COMFYUI_DIR}\\main.py",
            "--listen", "0.0.0.0", "--port", "8202",
        ],
        "stop_ps": (
            "Get-Process python -ErrorAction SilentlyContinue "
            "| Where-Object {$_.CommandLine -like '*ComfyUI*'} "
            "| Stop-Process -Force"
        ),
        "timeout": 120,
    },
    "tts": {
        "port": 8006,
        "health_path": "/health",
        "start_cmd": [TTS_PYTHON, TTS_SCRIPT],
        "stop_ps": (
            "Get-Process python -ErrorAction SilentlyContinue "
            "| Where-Object {$_.CommandLine -like '*tts_server*'} "
            "| Stop-Process -Force"
        ),
        "timeout": 60,
    },
}

# ── GPU State Management ───────────────────────────────────────────────────────

gpu_lock = threading.Lock()
current_gpu_mode = None


def _is_port_up(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(2)
        s.connect(("127.0.0.1", port))
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False
    finally:
        s.close()


def _http_get(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _detect_gpu_mode():
    for mode, cfg in GPU_MODES.items():
        if _http_get(f"http://localhost:{cfg['port']}{cfg['health_path']}"):
            return mode
    return None


def _stop_service(mode):
    cfg = GPU_MODES[mode]
    subprocess.run(
        ["powershell", "-Command", cfg["stop_ps"]],
        capture_output=True, timeout=15,
    )
    for _ in range(30):
        if not _is_port_up(cfg["port"]):
            return True
        time.sleep(1)
    return False


def _start_service(mode):
    cfg = GPU_MODES[mode]
    DETACHED = 0x00000008 | 0x00000200
    subprocess.Popen(
        cfg["start_cmd"], creationflags=DETACHED,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(cfg["timeout"]):
        if _http_get(f"http://localhost:{cfg['port']}{cfg['health_path']}"):
            return True
        time.sleep(1)
    return False


def ensure_gpu(mode):
    global current_gpu_mode
    with gpu_lock:
        cfg = GPU_MODES[mode]
        if _http_get(f"http://localhost:{cfg['port']}{cfg['health_path']}"):
            current_gpu_mode = mode
            return True
        if current_gpu_mode:
            _stop_service(current_gpu_mode)
        if _start_service(mode):
            current_gpu_mode = mode
            return True
        return False


# ── API Documentation ──────────────────────────────────────────────────────────

API_DOCS = {
    "name": "Smart Router Gateway",
    "version": "1.0.0",
    "description": (
        "Unified AI services proxy with automatic GPU management. "
        "Supports OpenAI-compatible chat, Anthropic Messages, Whisper STT, "
        "Qwen3-TTS, and FLUX image generation. The router auto-swaps the GPU "
        "between services as needed (takes ~30-90s on first request to a new mode)."
    ),
    "base_url": "http://192.168.0.18:4000",
    "gpu_modes": {
        "text": "LLM chat via llama.cpp (llama3.1 8B, ~48 tok/s)",
        "imggen": "Image generation via ComfyUI + FLUX (schnell/dev)",
        "tts": "Text-to-speech via Qwen3-TTS (custom voice / voice clone)",
    },
    "note": (
        "GPU auto-swap: the RTX 3060 (6GB) can only run ONE GPU service at a time. "
        "The router will automatically stop the current GPU service and start the "
        "needed one when you request a different service type. First request after "
        "a swap may take 30-90 seconds. Whisper STT runs on CPU and is always available."
    ),
    "endpoints": {
        "GET /health": {
            "description": "Health check",
            "response": {"status": "ok"},
        },
        "GET /status": {
            "description": "Full service status dashboard with GPU mode and VRAM",
            "response_example": {
                "services": {
                    "litellm": {"status": "up", "url": "http://localhost:4001"},
                    "llamacpp": {"status": "down", "url": "http://localhost:8201"},
                    "whisper": {"status": "up", "url": "http://localhost:8005"},
                    "tts": {"status": "down", "url": "http://localhost:8006"},
                    "comfyui": {"status": "up", "url": "http://localhost:8202"},
                },
                "gpu_mode": "imggen",
                "vram": "5234 MiB / 6144 MiB",
            },
        },
        "GET /v1/models": {
            "description": "List available models (OpenAI-compatible)",
            "response_example": {
                "object": "list",
                "data": [{"id": "llama3.1", "object": "model", "owned_by": "local"}],
            },
        },
        "POST /v1/chat/completions": {
            "description": "OpenAI-compatible chat completions. Auto-starts llama.cpp if needed.",
            "supports_streaming": True,
            "request_example": {
                "model": "llama3.1",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Explain quantum computing in one paragraph."},
                ],
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": False,
            },
            "response_example": {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "model": "llama3.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "..."},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 100, "total_tokens": 120},
            },
            "streaming": (
                "Set stream=true. Returns SSE events: data: {choices:[{delta:{content:'text'}}]}. "
                "Terminates with data: [DONE]."
            ),
        },
        "POST /v1/messages": {
            "description": (
                "Anthropic Messages API compatible endpoint. "
                "Translates to OpenAI format internally. Auto-starts llama.cpp."
            ),
            "supports_streaming": True,
            "request_example": {
                "model": "llama3.1",
                "max_tokens": 1024,
                "system": "You are a helpful assistant.",
                "messages": [
                    {"role": "user", "content": "Hello!"},
                ],
                "stream": False,
            },
            "response_example": {
                "id": "msg_abc123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello! How can I help you?"}],
                "model": "llama3.1",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 8},
            },
            "streaming": (
                "Set stream=true. Returns Anthropic SSE events: "
                "message_start, content_block_start, content_block_delta, "
                "content_block_stop, message_delta, message_stop."
            ),
        },
        "POST /v1/completions": {
            "description": "OpenAI legacy completions endpoint. Auto-starts llama.cpp.",
            "request_example": {
                "model": "llama3.1",
                "prompt": "Once upon a time",
                "max_tokens": 100,
            },
        },
        "POST /v1/audio/transcriptions": {
            "description": (
                "Whisper STT - Speech to text. Accepts multipart/form-data. "
                "Always available (runs on CPU)."
            ),
            "content_type": "multipart/form-data",
            "request_example": (
                "curl -X POST http://192.168.0.18:4000/v1/audio/transcriptions "
                '-F file=@recording.wav -F model=whisper-1'
            ),
            "response_example": {"text": "Hello, this is a transcription."},
            "supported_formats": "wav, mp3, flac, ogg, m4a",
        },
        "POST /v1/audio/speech": {
            "description": (
                "Text-to-speech via Qwen3-TTS. Auto-starts TTS service (GPU swap). "
                "Returns audio/wav bytes."
            ),
            "request_example": {
                "model": "tts-1",
                "input": "Hello, how are you today?",
                "voice": "Ryan",
            },
            "voices": [
                "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
                "Ryan", "Aiden", "Ono_Anna", "Sohee",
            ],
            "response": "Binary audio/wav data",
            "usage_example": (
                "curl -X POST http://192.168.0.18:4000/v1/audio/speech "
                '-H "Content-Type: application/json" '
                "-d '{\"input\":\"Hello world\",\"voice\":\"Ryan\"}' "
                "-o output.wav"
            ),
        },
        "POST /v1/audio/speech/clone": {
            "description": (
                "Voice cloning from reference audio. Upload a short clip of a voice "
                "and generate speech in that voice. Auto-starts TTS (GPU swap). "
                "Returns audio/wav bytes."
            ),
            "content_type": "multipart/form-data",
            "fields": {
                "file": "Audio file (wav/mp3) of the reference voice (required)",
                "text": "Text to speak in the cloned voice (required)",
                "language": "Language code (default: Auto)",
                "ref_text": "Transcript of the reference audio (optional, improves quality)",
            },
            "usage_example": (
                "curl -X POST http://192.168.0.18:4000/v1/audio/speech/clone "
                "-F file=@voice.wav "
                "-F text='Hello, this sounds like the original speaker' "
                "-F language=English "
                "-o cloned.wav"
            ),
        },
        "POST /v1/audio/speech/clone-url": {
            "description": (
                "Voice cloning from a reference audio URL. Same as /clone but takes "
                "a URL instead of file upload. Auto-starts TTS (GPU swap). "
                "Returns audio/wav bytes."
            ),
            "request_example": {
                "text": "Hello, this is a cloned voice",
                "language": "English",
                "ref_audio_url": "https://example.com/voice.wav",
                "ref_text": "Transcript of the reference audio",
            },
            "response": "Binary audio/wav data",
        },
        "POST /v1/images/generations": {
            "description": (
                "Image generation via ComfyUI + FLUX. Auto-starts ComfyUI (GPU swap). "
                "Returns base64-encoded PNG."
            ),
            "request_example": {
                "prompt": "a cat sitting on a windowsill, golden hour lighting",
                "model": "schnell",
                "size": "1024x1024",
            },
            "models": {
                "schnell": "FLUX.1-schnell (4 steps, ~39s, fast)",
                "dev": "FLUX.1-dev (28 steps, higher quality, slower)",
            },
            "response_example": {
                "created": 1712345678,
                "data": [{"b64_json": "...(base64 encoded PNG)..."}],
            },
        },
        "GET /api-docs": {
            "description": "This documentation. Read this to understand all available endpoints.",
        },
    },
    "quick_start": {
        "chat_python": (
            "from openai import OpenAI\n"
            "client = OpenAI(base_url='http://192.168.0.18:4000/v1', api_key='unused')\n"
            "resp = client.chat.completions.create(\n"
            "    model='llama3.1',\n"
            "    messages=[{'role':'user','content':'Hello!'}]\n"
            ")\n"
            "print(resp.choices[0].message.content)"
        ),
        "chat_anthropic_sdk": (
            "from anthropic import Anthropic\n"
            "client = Anthropic(base_url='http://192.168.0.18:4000', api_key='unused')\n"
            "msg = client.messages.create(\n"
            "    model='llama3.1',\n"
            "    max_tokens=1024,\n"
            "    messages=[{'role':'user','content':'Hello!'}]\n"
            ")\n"
            "print(msg.content[0].text)"
        ),
        "chat_curl": (
            'curl http://192.168.0.18:4000/v1/chat/completions '
            '-H "Content-Type: application/json" '
            "-d '{"
            '"model":"llama3.1",'
            '"messages":[{"role":"user","content":"hello"}]'
            "}'"
        ),
        "stt_curl": (
            "curl http://192.168.0.18:4000/v1/audio/transcriptions "
            "-F file=@recording.wav"
        ),
        "tts_curl": (
            "curl http://192.168.0.18:4000/v1/audio/speech "
            '-H "Content-Type: application/json" '
            "-d '{"
            '"input":"Hello world",'
            '"voice":"Ryan"'
            "}' "
            "-o output.wav"
        ),
        "image_curl": (
            "curl http://192.168.0.18:4000/v1/images/generations "
            '-H "Content-Type: application/json" '
            "-d '{"
            '"prompt":"a cat on a windowsill",'
            '"model":"schnell"'
            "}' "
            "-o image_response.json"
        ),
        "status_curl": "curl http://192.168.0.18:4000/status",
    },
    "supported_client_libraries": {
        "openai_python": (
            "from openai import OpenAI\n"
            "client = OpenAI(base_url='http://192.168.0.18:4000/v1', api_key='unused')\n"
            "# Then use client.chat.completions.create(), client.audio.transcriptions.create(), etc."
        ),
        "anthropic_python": (
            "from anthropic import Anthropic\n"
            "client = Anthropic(base_url='http://192.168.0.18:4000', api_key='unused')\n"
            "# Then use client.messages.create()"
        ),
        "litellm_python": (
            "import litellm\n"
            "litellm.api_base = 'http://192.168.0.18:4000/v1'\n"
            "litellm.api_key = 'unused'\n"
            "resp = litellm.completion(model='llama3.1', messages=[...])"
        ),
    },
}

# ── Markdown API Docs ──────────────────────────────────────────────────────────

MARKDOWN_DOCS = r"""# Smart Router Gateway — API Documentation

> **Base URL:** `http://192.168.0.18:4000`
> **Tailscale:** `http://100.84.161.63:4000`

Unified AI services proxy with automatic GPU management. Supports OpenAI-compatible chat, Anthropic Messages, Whisper STT, Qwen3-TTS, and FLUX image generation.

## GPU Auto-Swap

The RTX 3060 (6GB VRAM) can only run **one** GPU service at a time. The router automatically stops the current GPU service and starts the needed one when you request a different service type.

| Mode | Service | Port | VRAM |
|------|---------|------|------|
| `text` | llama.cpp (llama3.1 8B, ~48 tok/s) | 8201 | ~5GB |
| `imggen` | ComfyUI + FLUX | 8202 | ~5GB |
| `tts` | Qwen3-TTS | 8006 | ~3.5GB |

Whisper STT runs on **CPU** (port 8005) and is always available.

**First request after a GPU swap takes 30–90 seconds.** Subsequent requests to the same service are instant.

---

## Endpoints

### `GET /health`

Health check.

```json
{"status": "ok"}
```

### `GET /status`

Full service status dashboard.

```json
{
  "services": {
    "litellm": {"status": "down", "url": "http://localhost:4001"},
    "llamacpp": {"status": "up", "url": "http://localhost:8201"},
    "whisper": {"status": "up", "url": "http://localhost:8005"},
    "tts": {"status": "down", "url": "http://localhost:8006"},
    "comfyui": {"status": "down", "url": "http://localhost:8202"}
  },
  "gpu_mode": "text",
  "vram": "5161 MiB, 6144 MiB"
}
```

### `GET /v1/models`

List available models (OpenAI-compatible).

```json
{
  "object": "list",
  "data": [{"id": "llama3.1", "object": "model", "owned_by": "local"}]
}
```

### `POST /v1/chat/completions`

OpenAI-compatible chat completions. **Auto-starts llama.cpp (GPU swap).**

**Request:**

```json
{
  "model": "llama3.1",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing briefly."}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "llama3.1",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 20, "completion_tokens": 100, "total_tokens": 120}
}
```

**Streaming:** Set `"stream": true`. Returns SSE events: `data: {"choices":[{"delta":{"content":"text"}}]}`. Terminates with `data: [DONE]`.

### `POST /v1/messages`

Anthropic Messages API compatible endpoint. **Auto-starts llama.cpp (GPU swap).**

**Request:**

```json
{
  "model": "llama3.1",
  "max_tokens": 1024,
  "system": "You are a helpful assistant.",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false
}
```

**Response:**

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "Hello! How can I help you?"}],
  "model": "llama3.1",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 10, "output_tokens": 8}
}
```

**Streaming:** Set `"stream": true`. Returns Anthropic SSE events: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`.

### `POST /v1/completions`

OpenAI legacy completions endpoint. **Auto-starts llama.cpp (GPU swap).**

```json
{
  "model": "llama3.1",
  "prompt": "Once upon a time",
  "max_tokens": 100
}
```

### `POST /v1/audio/transcriptions`

Whisper STT — speech to text. **Always available (CPU).**

`Content-Type: multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Audio file (wav, mp3, flac, ogg, m4a) |
| `model` | string | No | Model name (ignored, uses `base`) |

**Example:**

```bash
curl -X POST http://192.168.0.18:4000/v1/audio/transcriptions \
  -F file=@recording.wav
```

**Response:**

```json
{"text": "Hello, this is a transcription."}
```

### `POST /v1/audio/speech`

Text-to-speech via Qwen3-TTS. **Auto-starts TTS service (GPU swap).**

**Request:**

```json
{
  "model": "tts-1",
  "input": "Hello, how are you today?",
  "voice": "Ryan"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | — | Text to speak (required) |
| `voice` | string | `"Ryan"` | Speaker voice |

**Available voices:** `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`

**Response:** Binary `audio/wav` data.

**Example:**

```bash
curl -X POST http://192.168.0.18:4000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","voice":"Ryan"}' \
  -o output.wav
```

### `POST /v1/audio/speech/clone`

Voice cloning from reference audio. Upload a short clip of a voice and generate speech in that voice. **Auto-starts TTS (GPU swap).**

`Content-Type: multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Audio clip of the reference voice (wav, mp3) |
| `text` | string | Yes | Text to speak in the cloned voice |
| `language` | string | No | Language (default: `"Auto"`) |
| `ref_text` | string | No | Transcript of the reference audio (improves quality) |

**Response:** Binary `audio/wav` data.

**Example:**

```bash
curl -X POST http://192.168.0.18:4000/v1/audio/speech/clone \
  -F file=@voice.wav \
  -F text="Hello, this sounds like the original speaker" \
  -F language=English \
  -o cloned.wav
```

### `POST /v1/audio/speech/clone-url`

Voice cloning from a reference audio URL. Same as `/clone` but takes a URL. **Auto-starts TTS (GPU swap).**

**Request:**

```json
{
  "text": "Hello, this is a cloned voice",
  "language": "English",
  "ref_audio_url": "https://example.com/voice.wav",
  "ref_text": "Transcript of the reference audio"
}
```

**Response:** Binary `audio/wav` data.

### `POST /v1/images/generations`

Image generation via ComfyUI + FLUX. **Auto-starts ComfyUI (GPU swap).**

**Request:**

```json
{
  "prompt": "a cat sitting on a windowsill, golden hour lighting",
  "model": "schnell",
  "size": "1024x1024"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | — | Image description (required) |
| `model` | string | `"schnell"` | `schnell` (fast, 4 steps) or `dev` (quality, 28 steps) |
| `size` | string | `"1024x1024"` | Output size |

**Response:**

```json
{
  "created": 1712345678,
  "data": [{"b64_json": "...(base64 encoded PNG)..."}]
}
```

### `GET /api-docs`

This documentation as structured JSON (machine-readable).

### `GET /api-docs.md`

This documentation as Markdown (human-readable).

---

## Quick Start

### Python — OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.0.18:4000/v1", api_key="unused")

# Chat
resp = client.chat.completions.create(
    model="llama3.1",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(resp.choices[0].message.content)

# Transcription
with open("recording.wav", "rb") as f:
    text = client.audio.transcriptions.create(model="whisper-1", file=f)
    print(text.text)

# TTS
response = client.audio.speech.create(
    model="tts-1", voice="Ryan", input="Hello world"
)
response.stream_to_file("output.wav")

# Image
response = client.images.generate(
    model="schnell", prompt="a cat on a windowsill", size="1024x1024"
)
```

### Python — Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://192.168.0.18:4000", api_key="unused")

msg = client.messages.create(
    model="llama3.1",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(msg.content[0].text)
```

### curl

```bash
# Chat
curl http://192.168.0.18:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.1","messages":[{"role":"user","content":"hello"}]}'

# Transcription
curl http://192.168.0.18:4000/v1/audio/transcriptions \
  -F file=@recording.wav

# TTS
curl http://192.168.0.18:4000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","voice":"Ryan"}' \
  -o output.wav

# Image
curl http://192.168.0.18:4000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cat on a windowsill","model":"schnell"}' \
  -o image_response.json

# Status
curl http://192.168.0.18:4000/status
```

---

## Service Architecture

```
Client -> Smart Router (port 4000)
              |-- /v1/chat/*, /v1/messages  ->  llama.cpp (port 8201, GPU)
              |-- /v1/audio/transcriptions   ->  Whisper (port 8005, CPU)
              |-- /v1/audio/speech           ->  Qwen3-TTS (port 8006, GPU)
              |-- /v1/images/generations     ->  ComfyUI (port 8202, GPU)
              |-- /status                    ->  health dashboard
              +-- GPU swap logic             ->  stops/starts services as needed
```
"""


# ── HTTP Handler ────────────────────────────────────────────────────────────────

class RouterHandler(BaseHTTPRequestHandler):

    # ── GET routes ─────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?")[0]  # strip query string
        if path == "/health":
            self._json(200, {"status": "ok"})
        elif path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "llama3.1", "object": "model", "owned_by": "local"},
            ]})
        elif path == "/status":
            self._status()
        elif path == "/api-docs":
            self._json(200, API_DOCS)
        elif path == "/api-docs.md":
            self._markdown_docs()
        else:
            self._json(404, {"error": "not found"})

    # ── POST routes ────────────────────────────────────────────────────────

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        ct = self.headers.get("Content-Type", "")

        path = self.path.split("?")[0]
        route = {
            "/v1/chat/completions":    lambda: self._chat(body),
            "/v1/messages":            lambda: self._anthropic(body),
            "/v1/completions":         lambda: self._completions(body),
            "/v1/audio/transcriptions": lambda: self._stt(body, ct),
            "/v1/audio/speech":        lambda: self._speech(body),
            "/v1/audio/speech/clone":  lambda: self._clone(body, ct),
            "/v1/audio/speech/clone-url": lambda: self._clone_url(body),
            "/v1/images/generations":  lambda: self._image(body),
        }.get(path)

        if route:
            route()
        else:
            self._json(404, {"error": "not found"})

    # ── Chat routing ───────────────────────────────────────────────────────

    def _chat_backend(self):
        """Prefer LiteLLM, fall back to direct llama.cpp."""
        if _http_get(BACKENDS["litellm"] + "/health"):
            return BACKENDS["litellm"]
        return BACKENDS["llamacpp"]

    def _chat(self, body):
        if not ensure_gpu("text"):
            return self._json(503, {"error": "text service unavailable"})
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        target = self._chat_backend()
        url = target + "/v1/chat/completions"
        if data.get("stream"):
            self._stream_sse(url, body)
        else:
            self._proxy_json(url, body)

    def _completions(self, body):
        if not ensure_gpu("text"):
            return self._json(503, {"error": "text service unavailable"})
        target = self._chat_backend()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        url = target + "/v1/completions"
        if data.get("stream"):
            self._stream_sse(url, body)
        else:
            self._proxy_json(url, body)

    def _anthropic(self, body):
        if not ensure_gpu("text"):
            return self._json(503, {"error": "text service unavailable"})
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        oa = self._anthropic_to_openai(data)
        target = self._chat_backend()
        model = data.get("model", "llama3.1")

        if data.get("stream"):
            self._stream_anthropic(target, oa, model)
        else:
            self._forward_anthropic(target, oa, model)

    # ── STT routing ────────────────────────────────────────────────────────

    def _stt(self, body, ct):
        self._proxy_raw(BACKENDS["whisper"] + "/v1/audio/transcriptions", body, ct)

    # ── TTS routing ────────────────────────────────────────────────────────

    def _speech(self, body):
        if not ensure_gpu("tts"):
            return self._json(503, {"error": "TTS service unavailable"})
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        tts_req = json.dumps({
            "text": data.get("input", data.get("text", "")),
            "speaker": data.get("voice", "Ryan"),
            "language": "Auto",
        }).encode()
        self._proxy_raw(
            BACKENDS["tts"] + "/tts", tts_req,
            "application/json", passthrough=True,
        )

    def _clone(self, body, ct):
        """Forward multipart voice clone to TTS server."""
        if not ensure_gpu("tts"):
            return self._json(503, {"error": "TTS service unavailable"})
        self._proxy_raw(
            BACKENDS["tts"] + "/tts/clone", body, ct, passthrough=True,
        )

    def _clone_url(self, body):
        """Forward URL-based voice clone to TTS server."""
        if not ensure_gpu("tts"):
            return self._json(503, {"error": "TTS service unavailable"})
        self._proxy_json(BACKENDS["tts"] + "/tts/clone-url", body)

    # ── Image routing ──────────────────────────────────────────────────────

    def _image(self, body):
        if not ensure_gpu("imggen"):
            return self._json(503, {"error": "image service unavailable"})
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})

        prompt = data.get("prompt", "")
        model_name = data.get("model", "schnell")

        wf_path = Path(WORKFLOWS_DIR) / f"flux_{model_name}.json"
        if not wf_path.exists():
            wf_path = Path(WORKFLOWS_DIR) / "flux_schnell.json"
        if not wf_path.exists():
            return self._json(500, {
                "error": f"workflow not found: {wf_path}. "
                         f"Deploy workflows with: python _deploy_router.py",
            })
        with open(wf_path) as f:
            workflow = json.load(f)

        if "3" in workflow:
            workflow["3"]["inputs"]["text"] = prompt

        try:
            result = self._api_post(BACKENDS["comfyui"] + "/prompt",
                                    {"prompt": workflow})
        except Exception as e:
            return self._json(502, {"error": f"ComfyUI submit failed: {e}"})

        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return self._json(502, {"error": "ComfyUI rejected", "details": result})

        for _ in range(300):
            time.sleep(2)
            hist = self._api_get(f"{BACKENDS['comfyui']}/history/{prompt_id}")
            if not hist or prompt_id not in hist:
                continue
            st = hist[prompt_id].get("status", {})
            if st.get("completed") or st.get("status_str") == "success":
                break
            if st.get("status_str") == "error":
                return self._json(502, {"error": "ComfyUI generation failed"})
        else:
            return self._json(504, {"error": "generation timed out"})

        outputs = hist[prompt_id].get("outputs", {})
        img_info = None
        for nid, nout in outputs.items():
            if "images" in nout:
                img_info = nout["images"][0]
                break
        if not img_info:
            return self._json(502, {"error": "no image in output"})

        fname = img_info["filename"]
        sub = img_info.get("subfolder", "")
        url = (f"{BACKENDS['comfyui']}/view?filename={fname}"
               f"&subfolder={sub}&type=output")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                img_data = r.read()
        except Exception as e:
            return self._json(502, {"error": f"image fetch failed: {e}"})

        self._json(200, {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(img_data).decode()}],
        })

    # ── Status dashboard ───────────────────────────────────────────────────

    def _status(self):
        services = {}
        for name, url in BACKENDS.items():
            hp = "/system_stats" if name == "comfyui" else "/health"
            healthy = _http_get(f"{url}{hp}")
            services[name] = {"status": "up" if healthy else "down", "url": url}

        vram = "unknown"
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            vram = r.stdout.strip()
        except Exception:
            pass

        self._json(200, {
            "services": services,
            "gpu_mode": current_gpu_mode,
            "vram": vram,
        })

    def _markdown_docs(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.end_headers()
        self.wfile.write(MARKDOWN_DOCS.encode("utf-8"))

    # ── Anthropic <-> OpenAI translation ───────────────────────────────────

    @staticmethod
    def _anthropic_to_openai(data):
        messages = data.get("messages", [])
        system = data.get("system", "")
        oa = []
        if system:
            if isinstance(system, list):
                system = "\n".join(
                    b["text"] for b in system
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            oa.append({"role": "system", "content": system})
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                parts = []
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "text":
                            parts.append(b["text"])
                        elif b.get("type") == "tool_result":
                            parts.append(json.dumps(b.get("content", "")))
                content = "\n".join(parts)
            oa.append({"role": m.get("role", "user"), "content": content})
        req = {
            "model": "llama3.1",
            "messages": oa,
            "max_tokens": data.get("max_tokens", 1024),
            "stream": data.get("stream", False),
        }
        if "temperature" in data:
            req["temperature"] = data["temperature"]
        return req

    def _forward_anthropic(self, target, oa_req, model):
        url = target + "/v1/chat/completions"
        raw = json.dumps(oa_req).encode()
        req = urllib.request.Request(url, data=raw,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                result = json.loads(r.read())
        except Exception as e:
            return self._json(502, {
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            })

        ch = result.get("choices", [{}])[0]
        usage = result.get("usage", {})
        self._json(200, {
            "id": "msg_" + uuid.uuid4().hex[:24],
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text",
                         "text": ch.get("message", {}).get("content", "")}],
            "model": model,
            "stop_reason": ("end_turn" if ch.get("finish_reason") == "stop"
                            else ch.get("finish_reason")),
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        })

    def _stream_anthropic(self, target, oa_req, model):
        url = target + "/v1/chat/completions"
        oa_req["stream"] = True
        raw = json.dumps(oa_req).encode()
        req = urllib.request.Request(url, data=raw,
                                     headers={"Content-Type": "application/json"})
        mid = "msg_" + uuid.uuid4().hex[:24]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def sse(evt, d):
            self.wfile.write(
                f"event: {evt}\ndata: {json.dumps(d)}\n\n".encode())
            self.wfile.flush()

        sse("message_start", {"type": "message_start", "message": {
            "id": mid, "type": "message", "role": "assistant",
            "model": model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0,
                      "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 0},
        }})
        sse("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        sse("ping", {"type": "ping"})

        tokens = 0
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    dec = line.decode(errors="replace").strip()
                    if not dec.startswith("data: "):
                        continue
                    payload = dec[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    text = (chunk.get("choices", [{}])[0]
                            .get("delta", {}).get("content", ""))
                    if text:
                        tokens += 1
                        sse("content_block_delta", {
                            "type": "content_block_delta", "index": 0,
                            "delta": {"type": "text_delta", "text": text},
                        })
        except Exception as e:
            sse("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta",
                          "text": f"\n[stream error: {e}]"},
            })

        sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": tokens},
        })
        sse("message_stop", {"type": "message_stop"})

    # ── Proxy helpers ──────────────────────────────────────────────────────

    def _proxy_json(self, url, body):
        if isinstance(body, str):
            body = body.encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                self._json(200, json.loads(r.read()))
        except urllib.error.URLError as e:
            self._json(502, {"error": str(e)})

    def _proxy_raw(self, url, body, ct, passthrough=False):
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": ct})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp_body = r.read()
                self.send_response(200)
                if passthrough:
                    self.send_header("Content-Type",
                                     r.headers.get("Content-Type",
                                                   "application/octet-stream"))
                else:
                    self.send_header("Content-Type",
                                     r.headers.get("Content-Type",
                                                   "application/json"))
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except Exception as e:
            self._json(502, {"error": str(e)})

    def _stream_sse(self, url, body):
        if isinstance(body, str):
            body = body.encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                for line in r:
                    self.wfile.write(line)
                    self.wfile.flush()
        except Exception:
            pass

    @staticmethod
    def _api_get(url):
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                return json.loads(r.read())
        except Exception:
            return None

    @staticmethod
    def _api_post(url, data):
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        pass


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    current_gpu_mode = _detect_gpu_mode()
    print(f"Smart Router on :{PORT}  |  GPU mode: {current_gpu_mode or 'none'}",
          flush=True)
    HTTPServer(("0.0.0.0", PORT), RouterHandler).serve_forever()
