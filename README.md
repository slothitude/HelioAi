# Hotswap — LLM Inference & Image Gen on RTX 3060

A home inference stack running on a laptop with an RTX 3060 6GB VRAM, controlled remotely via SSH from a desktop. Two GPU-exclusive services share the hardware: text generation (llama.cpp) and image generation (ComfyUI).

## Hardware

| | |
|---|---|
| **GPU** | NVIDIA RTX 3060 Laptop, 6GB VRAM, CUDA 13.2 |
| **RAM** | 40 GB |
| **Disk** | ~60 GB free |
| **OS** | Windows 11 |

## Services

### Text Generation — llama.cpp (Port 8201)

- **Model**: Llama 3.1 8B (BF16, 4.9GB), all 33/33 layers on GPU
- **Performance**: ~48 tok/s generation, ~218 tok/s prompt
- **VRAM**: 5159/6144 MiB
- **Endpoint**: `http://192.168.0.18:8201`

### Image Generation — ComfyUI (Port 8202)

- **Models**:
  - **FLUX.1-schnell** Q3_K_S (5.2GB) — 4 steps, ~29s/image
  - **FLUX.1-dev** Q3_K_S (5.2GB) — 28 steps, ~6min/image
  - **FLUX.1-Kontext-dev** Q3_K_S (5.2GB) — instruction-based image editing
- **Text encoders**: CLIP-L (246MB) + T5-XXL Q4_K_S (2.7GB, CPU offload)
- **VAE**: ae.safetensors (335MB)
- **Endpoint**: `http://192.168.0.18:8202`

The two services are GPU-exclusive — only one runs at a time. Use `_gpu_swap.py` to toggle.

## Quick Start

```bash
# Switch to image generation
python _gpu_swap.py --to imggen

# Generate an image (schnell, ~29s)
python _test_imggen.py --model schnell --prompt "a wolf howling at the moon"

# Generate an image (dev, higher quality, ~6min)
python _test_imggen.py --model dev --prompt "a cyberpunk city at night"

# Edit an image with Kontext
python _test_kontext.py --image output_wolf.png --instruction "make it a watercolor painting"

# Switch back to text generation
python _gpu_swap.py --to text
```

## Scripts

| Script | Purpose |
|--------|---------|
| `_deploy_final.py` | Deploy llama.cpp with Llama 3.1 all-GPU |
| `_deploy_comfyui.py` | Full ComfyUI setup (install, download models, start) |
| `_test_imggen.py` | Text-to-image generation + benchmark |
| `_test_kontext.py` | Image editing with Kontext (upload + instruction) |
| `_gpu_swap.py` | Toggle GPU between llama.cpp and ComfyUI |
| `_update_llama.py` | Download new llama.cpp release to laptop |
| `_speed_test.py` | Benchmark text inference speed |

All `_*.py` scripts use paramiko SSH to `192.168.0.18`.

## Workflows

ComfyUI API-format workflow JSONs in `workflows/`:

| File | Model | Steps | CFG | Use |
|------|-------|-------|-----|-----|
| `flux_schnell.json` | FLUX.1-schnell Q3_K_S | 4 | 1.0 | Fast generation |
| `flux_dev.json` | FLUX.1-dev Q3_K_S | 28 | 3.5 | High quality |
| `flux_kontext.json` | FLUX.1-Kontext-dev Q3_K_S | 28 | 2.5 | Image editing |

## Network

- **Desktop (rog)**: `100.107.34.12` (Tailscale)
- **Laptop (lappy-server)**: `192.168.0.18` (LAN), `100.84.161.63` (Tailscale)
- All scripts connect via LAN SSH

## Environment Variables

- `HF_TOKEN` — HuggingFace access token (required for gated model downloads like the FLUX VAE)

## Notes

- ComfyUI's `app/logger.py` is patched with `try/except OSError` on `write()` and `flush()` to fix a Windows tqdm bug. Re-patch after ComfyUI updates.
- PyTorch must be installed with `--index-url https://download.pytorch.org/whl/cu124` (default pip installs CPU-only).
- The ComfyUI-GGUF node types are `UnetLoaderGGUF` and `DualCLIPLoaderGGUF` (not `GGUFLoader`).
- Windows firewall rules are added for ports 8201 and 8202.
