import paramiko, time, socket, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "192.168.0.18"
USER = "aaron"
PASS = "T0b1@n7243"
LLAMA_PORT = 8201
COMFYUI_PORT = 8202
COMFYUI_DIR = r"C:\Users\aaron\hotswap\ComfyUI"

if len(sys.argv) < 2 or sys.argv[1] not in ("imggen", "text"):
    print("Usage: python _gpu_swap.py --to imggen|text")
    print("  imggen  → stop llama.cpp, start ComfyUI")
    print("  text    → stop ComfyUI, start llama.cpp")
    sys.exit(1)

TARGET = sys.argv[1]

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


def wait_for_server(port, keyword, timeout=90):
    """Wait for server to start by polling stderr for keyword."""
    stderr_data = b""
    for i in range(timeout):
        try:
            stderr_data += channel.recv_stderr(4096)
        except socket.timeout:
            pass
        try:
            channel.recv(4096)
        except socket.timeout:
            pass
        text = stderr_data.decode(errors='replace')
        if keyword in text.lower():
            return True, i
    return False, timeout


if TARGET == "imggen":
    print("\n=== Switching to ComfyUI (imggen) ===")

    # Stop llama.cpp
    print("  Stopping llama.cpp...")
    ssh.exec_command('powershell -Command "Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue"')
    time.sleep(3)

    # Start ComfyUI
    venv_python = rf'{COMFYUI_DIR}\venv\Scripts\python.exe'
    cmd = f'"{venv_python}" "{COMFYUI_DIR}\\main.py" --listen 0.0.0.0 --port {COMFYUI_PORT}'
    print("  Starting ComfyUI...")
    channel = run_bg(cmd)

    ok, secs = wait_for_server(COMFYUI_PORT, "starting server")
    if ok:
        print(f"  ComfyUI started ({secs}s)")
    else:
        print(f"  WARNING: startup not detected in {secs}s")

    time.sleep(2)
    out, _ = run(f'curl.exe -s http://localhost:{COMFYUI_PORT}/system_stats')
    print(f"  Health: {'OK' if out else 'no response'}")

elif TARGET == "text":
    print("\n=== Switching to llama.cpp (text) ===")

    # Stop ComfyUI (kill python processes running ComfyUI)
    print("  Stopping ComfyUI...")
    ssh.exec_command(
        'powershell -Command "Get-Process python -ErrorAction SilentlyContinue '
        '| Where-Object {$_.CommandLine -like \'*ComfyUI*\'} | Stop-Process -Force"'
    )
    time.sleep(3)

    # Start llama.cpp
    llama31 = r"C:\Users\aaron\.ollama\models\blobs\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
    cmd = (
        r"C:\Users\aaron\hotswap\llama-server.exe"
        f" -m {llama31}"
        r" -ngl 99 -c 2048 -t 8 --flash-attn auto --host 0.0.0.0 --port 8201"
    )
    print("  Starting llama.cpp...")
    channel = run_bg(cmd)

    ok, secs = wait_for_server(LLAMA_PORT, "listening")
    if ok:
        print(f"  llama.cpp started ({secs}s)")
    else:
        print(f"  WARNING: startup not detected in {secs}s")

    time.sleep(1)
    out, _ = run(f'curl.exe -s http://localhost:{LLAMA_PORT}/health')
    print(f"  Health: {out}")

# Show VRAM
out, _ = run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
print(f"  VRAM: {out}")

ssh.close()
print("\nDone!")
