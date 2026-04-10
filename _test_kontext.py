import json, sys, time, io, os, urllib.request, urllib.error, base64

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ["no_proxy"] = "192.168.0.18,100.84.161.63,localhost,127.0.0.1"

HOST = "192.168.0.18"
PORT = 8202
BASE = f"http://{HOST}:{PORT}"

INSTRUCTION = "make it a watercolor painting style"
INPUT_IMAGE = None

for i, arg in enumerate(sys.argv[1:]):
    if arg == "--instruction" and len(sys.argv) > i + 2:
        INSTRUCTION = sys.argv[i + 2]
    elif arg == "--image" and len(sys.argv) > i + 2:
        INPUT_IMAGE = sys.argv[i + 2]

if not INPUT_IMAGE:
    # Use the last generated schnell image as default
    INPUT_IMAGE = "output_schnell.png"

print(f"=== Kontext Test ===")
print(f"  Input: {INPUT_IMAGE}")
print(f"  Instruction: {INSTRUCTION}")


def api_get(path):
    req = urllib.request.Request(f"{BASE}{path}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def api_post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


# Upload input image to ComfyUI
print("Uploading input image...")
filename = os.path.basename(INPUT_IMAGE)
with open(INPUT_IMAGE, "rb") as f:
    img_data = f.read()

# Upload via /upload/image endpoint
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = (
    f"--{boundary}\r\n"
    f"Content-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
    f"Content-Type: image/png\r\n\r\n"
).encode() + img_data + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    f"{BASE}/upload/image",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)
with urllib.request.urlopen(req, timeout=30) as resp:
    upload_result = json.loads(resp.read().decode())
uploaded_name = upload_result.get("name", filename)
print(f"  Uploaded: {uploaded_name}")

# Load and configure workflow
with open("workflows/flux_kontext.json") as f:
    workflow = json.load(f)

# Set instruction text
workflow["3"]["inputs"]["text"] = INSTRUCTION
# Set uploaded image name
workflow["5"]["inputs"]["image"] = uploaded_name

# Submit
print("Submitting workflow...")
t_start = time.time()

result = api_post("/prompt", {"prompt": workflow, "client_id": "kontext-test"})
prompt_id = result.get("prompt_id")
if not prompt_id:
    if "node_errors" in result:
        print(f"  Node errors: {json.dumps(result['node_errors'], indent=2)[:500]}")
    print(f"  Response: {json.dumps(result, indent=2)[:500]}")
    sys.exit(1)

print(f"  Prompt ID: {prompt_id}")

# Poll for completion
print("Generating...")
while True:
    history = api_get(f"/history/{prompt_id}")
    if history and prompt_id in history:
        status = history[prompt_id].get("status", {})
        if status.get("completed", False) or status.get("status_str") == "success":
            break
        if status.get("status_str") == "error":
            for msg in status.get("messages", []):
                if msg[0] == "execution_error":
                    print(f"  ERROR at node {msg[1].get('node_id')}: {msg[1].get('exception_message')}")
            sys.exit(1)
    time.sleep(2)

t_elapsed = time.time() - t_start

# Get output
outputs = history[prompt_id].get("outputs", {})
image_info = None
for node_id, node_out in outputs.items():
    if "images" in node_out:
        image_info = node_out["images"][0]
        break

if image_info:
    filename_out = image_info["filename"]
    subfolder = image_info.get("subfolder", "")
    print(f"\n  Generated: {filename_out}")
    print(f"  Time: {t_elapsed:.1f}s")

    # Download
    dl_url = f"{BASE}/view?filename={filename_out}&subfolder={subfolder}&type=output"
    req = urllib.request.Request(dl_url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        img_data = resp.read()

    local_path = "output_kontext.png"
    with open(local_path, "wb") as f:
        f.write(img_data)
    print(f"  Saved: {local_path} ({len(img_data) // 1024}KB)")
else:
    print(f"  No output: {json.dumps(outputs)[:300]}")

print(f"\n  Benchmark: kontext = {t_elapsed:.1f}s total")
