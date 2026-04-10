import paramiko, time, socket, sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "192.168.0.18"
USER = "aaron"
PASS = "T0b1@n7243"
COMFYUI_DIR = r"C:\Users\aaron\hotswap\ComfyUI"
COMFYUI_PORT = 8202
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODEL = "schnell"
SKIP_INSTALL = False
for arg in sys.argv[1:]:
    if arg in ("schnell", "dev"):
        MODEL = arg
    elif arg == "--skip-install":
        SKIP_INSTALL = True

print(f"=== ComfyUI Deploy (model: {MODEL}) ===")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS, timeout=15)
print("Connected to lappy-server")


def run(cmd, timeout=120):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    return stdout.read().decode(errors='replace').strip(), stderr.read().decode(errors='replace').strip()


def run_bg(cmd):
    transport = ssh.get_transport()
    channel = transport.open_session()
    channel.settimeout(2)
    channel.exec_command(cmd)
    return channel


def patch_logger():
    """Patch ComfyUI logger.py to handle OSError in write/flush (Windows tqdm fix)."""
    logger_path = f"{COMFYUI_DIR}/app/logger.py"
    sftp = ssh.open_sftp()
    try:
        with sftp.open(logger_path, 'rb') as f:
            content = f.read().decode()

        patched = False
        # Patch write method
        old_write = '        super().write(data)\r\n'
        new_write = '        try:\r\n            super().write(data)\r\n        except OSError:\r\n            pass\r\n'
        if old_write in content:
            content = content.replace(old_write, new_write, 1)
            patched = True

        # Patch flush method
        old_flush = '        super().flush()\r\n'
        new_flush = '        try:\r\n            super().flush()\r\n        except OSError:\r\n            pass\r\n'
        if old_flush in content:
            content = content.replace(old_flush, new_flush, 1)
            patched = True

        if patched:
            with sftp.open(logger_path, 'wb') as f:
                f.write(content.encode())
            print("  Logger patched (OSError fix)")
    except Exception as e:
        print(f"  Logger patch skipped: {e}")
    finally:
        sftp.close()


def download_file(url, dest, auth_header=False):
    """Download file using curl with optional HF auth."""
    header = f'-H "Authorization: Bearer {HF_TOKEN}" ' if auth_header else ''
    cmd = f'curl.exe -s -L {header}-o "{dest}" "{url}"'

    # Use background exec + buffer draining to avoid paramiko buffer deadlock
    transport = ssh.get_transport()
    channel = transport.open_session()
    channel.settimeout(5)
    channel.exec_command(cmd)

    for _ in range(360):  # 30 min max
        try:
            channel.recv(4096)
        except socket.timeout:
            pass
        try:
            channel.recv_stderr(4096)
        except socket.timeout:
            pass
        if channel.exit_status_ready():
            break

    try:
        channel.close()
    except:
        pass


# --- Step 1: Install ComfyUI ---
if not SKIP_INSTALL:
    out, _ = run(f'if exist {COMFYUI_DIR} echo EXISTS')
    if 'EXISTS' in out:
        print("  ComfyUI dir exists, skipping clone")
    else:
        print("  Cloning ComfyUI...")
        out, err = run(f'git clone https://github.com/comfyanonymous/ComfyUI.git {COMFYUI_DIR}', timeout=300)
        if err:
            print(f"    Clone error: {err[:300]}")

    venv_python = rf'{COMFYUI_DIR}\venv\Scripts\python.exe'
    out, _ = run(f'if exist "{venv_python}" echo EXISTS')
    if 'EXISTS' in out:
        print("  Venv exists, skipping deps")
    else:
        print("  Creating venv...")
        run(f'python -m venv {COMFYUI_DIR}\\venv', timeout=60)

        print("  Installing PyTorch CUDA 12.4...")
        out, err = run(
            f'{venv_python} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124',
            timeout=900
        )
        if err and 'error' in err.lower():
            print(f"    PyTorch error: {err[:300]}")

        print("  Installing ComfyUI requirements...")
        run(f'{venv_python} -m pip install -r {COMFYUI_DIR}\\requirements.txt', timeout=300)

    # ComfyUI-GGUF custom node
    gguf_node = rf'{COMFYUI_DIR}\custom_nodes\ComfyUI-GGUF'
    out, _ = run(f'if exist "{gguf_node}" echo EXISTS')
    if 'EXISTS' in out:
        print("  ComfyUI-GGUF node exists")
    else:
        print("  Installing ComfyUI-GGUF...")
        run(f'cd {COMFYUI_DIR}\\custom_nodes && git clone https://github.com/city96/ComfyUI-GGUF', timeout=120)

    print("  Installing gguf + deps in venv...")
    run(f'{venv_python} -m pip install --upgrade gguf sentencepiece protobuf', timeout=120)

    # Patch logger to fix tqdm OSError on Windows
    patch_logger()
else:
    print("  Skipping installation (--skip-install)")
    venv_python = rf'{COMFYUI_DIR}\venv\Scripts\python.exe'


# --- Step 2: Download models ---
print("\nDownloading models...")
models_base = f"{COMFYUI_DIR}/models"

for d in ['unet', 'clip', 'vae']:
    run(f'if not exist "{models_base}\\{d}" mkdir "{models_base}\\{d}"')

MODELS = [
    # (filename, url, subdir, needs_auth)
    ("flux1-schnell-Q3_K_S.gguf",
     "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q3_K_S.gguf",
     "unet", False),
    ("flux1-dev-Q3_K_S.gguf",
     "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q3_K_S.gguf",
     "unet", False),
    ("t5xxl-Q4_K_S.gguf",
     "https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q4_K_S.gguf",
     "clip", False),
    ("clip_l.safetensors",
     "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
     "clip", False),
    ("ae.safetensors",
     "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors",
     "vae", True),
]

for name, url, subdir, needs_auth in MODELS:
    dest = f"{models_base}/{subdir}/{name}"
    stdin, stdout, stderr = ssh.exec_command(
        f'powershell -Command "if (Test-Path \'{dest}\') {{ (Get-Item \'{dest}\').Length }}"'
    )
    stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    if out.replace('.', '').isdigit() and float(out) > 1_000_000:
        print(f"  {name}: exists ({float(out) / 1024 / 1024:.0f}MB)")
        continue
    print(f"  Downloading {name}...")
    download_file(url, dest, auth_header=needs_auth)
    # Verify
    stdin, stdout, stderr = ssh.exec_command(
        f'powershell -Command "if (Test-Path \'{dest}\') {{ (Get-Item \'{dest}\').Length / 1MB }}"'
    )
    stdout.channel.recv_exit_status()
    size = stdout.read().decode().strip()
    print(f"    -> {size} MB")


# --- Step 3: Stop llama.cpp, stop any existing ComfyUI ---
print("\nStopping llama.cpp...")
ssh.exec_command('powershell -Command "Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue"')

print("Stopping existing ComfyUI...")
ssh.exec_command('powershell -Command "Stop-Process -Name python -Force -ErrorAction SilentlyContinue"')
time.sleep(3)


# --- Step 4: Start ComfyUI ---
print(f"\nStarting ComfyUI on port {COMFYUI_PORT}...")
cmd = f'"{venv_python}" "{COMFYUI_DIR}\\main.py" --listen 0.0.0.0 --port {COMFYUI_PORT}'
channel = run_bg(cmd)

stderr_data = b""
started = False
for i in range(120):
    try:
        stderr_data += channel.recv_stderr(4096)
    except socket.timeout:
        pass
    try:
        channel.recv(4096)
    except socket.timeout:
        pass
    text = stderr_data.decode(errors='replace')
    if "Starting server" in text or "To see the GUI go to" in text:
        print(f"  Server started at {i}s")
        started = True
        break

if not started:
    print(f"  WARNING: Did not detect startup in 120s")
    print(f"  Last output: {stderr_data.decode(errors='replace')[-500:]}")


# --- Step 5: Health check ---
print("\nHealth check...")
time.sleep(3)
out, _ = run(f'curl.exe -s http://localhost:{COMFYUI_PORT}/system_stats')
if out and 'system' in out.lower():
    print("  System stats: OK")
else:
    print(f"  System stats: no response")

out, _ = run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
print(f"  VRAM: {out}")

# Add firewall rule (idempotent)
run(f'netsh advfirewall firewall add rule name="ComfyUI" dir=in action=allow protocol=TCP localport={COMFYUI_PORT}')

ssh.close()
print(f"\nDone! ComfyUI running at http://{HOST}:{COMFYUI_PORT}")
print(f"  Model: {MODEL} (selected via workflow at generation time)")
print(f"  Test:  python _test_imggen.py --model {MODEL} --prompt 'a cat on a windowsill'")
