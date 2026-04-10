import json, sys, time, io, os, urllib.request, urllib.error

# Bypass proxy for local/LAN connections
os.environ["no_proxy"] = "192.168.0.18,100.84.161.63,localhost,127.0.0.1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "192.168.0.18"
PORT = 8202
BASE = f"http://{HOST}:{PORT}"

MODEL = "schnell"
PROMPT = "a cat sitting on a windowsill, golden hour lighting, detailed"

for arg in sys.argv[1:]:
    if arg in ("schnell", "dev"):
        MODEL = arg
    elif arg == "--prompt" and len(sys.argv) > sys.argv.index(arg) + 1:
        PROMPT = sys.argv[sys.argv.index(arg) + 1]

WORKFLOW_FILE = f"workflows/flux_{MODEL}.json"
print(f"=== Image Gen Test ({MODEL}) ===")
print(f"  Prompt: {PROMPT}")


def api_get(path):
    try:
        req = urllib.request.Request(f"{BASE}{path}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return None


def api_post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


# Load workflow
with open(WORKFLOW_FILE) as f:
    workflow = json.load(f)

# Set prompt in the workflow (node 3 = positive, node 4 = negative)
for node_id, node in workflow.items():
    if node.get("class_type") == "CLIPTextEncode":
        if "negative" not in node_id:
            # First CLIPTextEncode is the positive prompt (node "3")
            pass
    # Set prompt text in positive encoder
    if node_id == "3":
        node["inputs"]["text"] = PROMPT

# Submit
print("Submitting workflow...")
t_start = time.time()

try:
    result = api_post("/prompt", {"prompt": workflow, "client_id": "hotswap-test"})
except urllib.error.URLError as e:
    print(f"  ERROR: Cannot connect to ComfyUI at {BASE}")
    print(f"  Is ComfyUI running? Run: python _deploy_comfyui.py {MODEL}")
    sys.exit(1)
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

prompt_id = result.get("prompt_id")
if not prompt_id:
    # Check for node execution errors
    if "node_errors" in result:
        print(f"  Node errors: {json.dumps(result['node_errors'], indent=2)}")
    print(f"  Full response: {json.dumps(result, indent=2)[:500]}")
    sys.exit(1)

print(f"  Prompt ID: {prompt_id}")

# Poll for completion
print("Waiting for generation...")
last_status = ""
while True:
    history = api_get(f"/history/{prompt_id}")
    if history and prompt_id in history:
        status = history[prompt_id].get("status", {})
        if status.get("completed", False) or status.get("status_str") == "success":
            break
        if status.get("status_str") == "error":
            print(f"  ERROR during generation!")
            msgs = history[prompt_id].get("outputs", {})
            print(f"  Outputs: {json.dumps(msgs, indent=2)[:500]}")
            sys.exit(1)
        cur = status.get("status_str", "")
        if cur != last_status:
            print(f"  Status: {cur}")
            last_status = cur
    time.sleep(2)

t_elapsed = time.time() - t_start

# Get output image
outputs = history[prompt_id].get("outputs", {})
image_info = None
for node_id, node_out in outputs.items():
    if "images" in node_out:
        image_info = node_out["images"][0]
        break

if image_info:
    filename = image_info["filename"]
    subfolder = image_info.get("subfolder", "")
    print(f"\n  Generated: {filename}")
    print(f"  Time: {t_elapsed:.1f}s")

    # Download image locally
    dl_url = f"{BASE}/view?filename={filename}&subfolder={subfolder}&type=output"
    try:
        req = urllib.request.Request(dl_url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            img_data = resp.read()

        local_path = f"output_{MODEL}.png"
        with open(local_path, "wb") as f:
            f.write(img_data)
        print(f"  Saved: {local_path} ({len(img_data) // 1024}KB)")
    except Exception as e:
        print(f"  Download error: {e}")
        print(f"  Image available at: {dl_url}")
else:
    print(f"  No image in outputs: {json.dumps(outputs)[:300]}")

print(f"\n  Benchmark: {MODEL} = {t_elapsed:.1f}s total")
