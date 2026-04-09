"""
KV Bootstrap — Long-Context Two-Pass Inference
================================================
For long prompts (>4K tokens), this does two passes:

  Pass 1 (draft):  Fast speculative generation with the draft model to fill
                   the KV cache with context.
  Pass 2 (refine): Use the cached context from pass 1 as a bootstrap for
                   higher-quality continuation from the target model.

Usage:
  python kv_bootstrap.py "Your long prompt here"
  python kv_bootstrap.py --file document.txt "Summarize this"
  python kv_bootstrap.py --context 8192 --max-tokens 2048 "Analyze..."
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

try:
    import requests
    import httpx
except ImportError:
    os.system(f"{sys.executable} -m pip install requests httpx")
    import requests
    import httpx

DEFAULT_SPEC_URL = "http://100.84.161.63:8201"
DEFAULT_SOLO_URL = "http://100.84.161.63:8204"
DEFAULT_LIGHT_URL = "http://100.84.161.63:8203"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English, ~2 for code."""
    code_chars = sum(1 for c in text if c in "{}[]();=<>+-*/")
    ratio = 3.5 if code_chars / max(len(text), 1) < 0.1 else 3.0
    return int(len(text) / ratio)


def check_health(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=3).status_code == 200
    except Exception:
        return False


def count_context_tokens(messages: list[dict], url: str) -> int:
    """Use the server's tokenization endpoint to count tokens."""
    try:
        r = requests.post(
            f"{url}/tokenize",
            json={"content": json.dumps(messages)},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("tokens", 0)
    except Exception:
        pass
    # Fallback to estimate
    total = "".join(m.get("content", "") for m in messages)
    return estimate_tokens(total)


def stream_chat(url: str, messages: list[dict], max_tokens: int = 512,
                 temperature: float = 0.7) -> str:
    """Stream chat and return full response."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    full = ""
    t0 = time.time()
    tokens = 0

    with httpx.stream(
        "POST", f"{url}/v1/chat/completions",
        json=payload,
        timeout=httpx.Timeout(600.0, connect=10.0),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full += token
                    tokens += 1
                    print(token, end="", flush=True)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    elapsed = time.time() - t0
    if tokens > 0 and elapsed > 0:
        print(f"\n  [{tokens} tokens, {elapsed:.1f}s, {tokens/elapsed:.1f} tok/s]")

    return full


def two_pass_bootstrap(
    prompt: str,
    context: str = "",
    spec_url: str = DEFAULT_SPEC_URL,
    solo_url: str = DEFAULT_SOLO_URL,
    light_url: str = DEFAULT_LIGHT_URL,
    max_tokens: int = 1024,
    context_size: int = 4096,
):
    """
    Two-pass inference for long contexts.

    Pass 1: Use lighter model (or spec with more draft tokens) to establish
            the context in the KV cache.
    Pass 2: Use the full model to generate higher-quality output, potentially
            building on the KV cache from pass 1.
    """
    # Determine which server to use for each pass
    # Prefer spec server for pass 2 (quality), light for pass 1 (speed)
    use_url_pass2 = spec_url if check_health(spec_url) else solo_url
    use_url_pass1 = light_url if check_health(light_url) else use_url_pass2

    if not check_health(use_url_pass2):
        print("[ERROR] No target model server available.")
        return

    messages = []
    if context:
        messages.append({
            "role": "system",
            "content": f"Context:\n{context}\n\nAnswer questions based on this context."
        })
    else:
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant. Provide detailed, accurate responses."
        })

    # Estimate if we need two-pass
    total_input = context + prompt if context else prompt
    input_tokens = estimate_tokens(total_input)

    print(f"Input estimate: ~{input_tokens} tokens")

    if input_tokens < 1500:
        # Short input — skip two-pass, just do single generation
        print("\n[Single-pass] Input is short enough for direct generation.\n")
        messages.append({"role": "user", "content": prompt})
        response = stream_chat(use_url_pass2, messages, max_tokens=max_tokens)
        return response

    # --- Two-pass ---
    print(f"\n[Pass 1] Bootstrap with {use_url_pass1} (fast)")
    print("-" * 40)

    pass1_messages = messages.copy()
    pass1_messages.append({
        "role": "user",
        "content": f"Provide a brief outline or key points about the following:\n\n{prompt}"
    })

    pass1_response = stream_chat(
        use_url_pass1, pass1_messages,
        max_tokens=min(max_tokens // 2, 512),
        temperature=0.5,
    )

    if not pass1_response:
        print("[WARN] Pass 1 produced no output, proceeding with direct generation.")

    # Pass 2: Use pass 1 output as additional context
    print(f"\n[Pass 2] Refine with {use_url_pass2} (quality)")
    print("-" * 40)

    pass2_messages = messages.copy()
    if pass1_response:
        pass2_messages.append({
            "role": "assistant",
            "content": f"[Draft outline]: {pass1_response}"
        })
    pass2_messages.append({
        "role": "user",
        "content": f"Now provide a detailed, thorough response:\n\n{prompt}"
    })

    pass2_response = stream_chat(
        use_url_pass2, pass2_messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )

    return pass2_response


def main():
    parser = argparse.ArgumentParser(
        description="KV Bootstrap: Two-pass long-context inference"
    )
    parser.add_argument("prompt", help="The prompt to send")
    parser.add_argument("--file", "-f", help="Read context from file")
    parser.add_argument("--spec-url", default=DEFAULT_SPEC_URL)
    parser.add_argument("--solo-url", default=DEFAULT_SOLO_URL)
    parser.add_argument("--light-url", default=DEFAULT_LIGHT_URL)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--context-size", type=int, default=4096,
                        help="Target context window size")
    args = parser.parse_args()

    context = ""
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"[ERROR] File not found: {args.file}")
            sys.exit(1)
        context = path.read_text(encoding="utf-8")
        print(f"Loaded context from {args.file} ({len(context)} chars)")

    print()
    print("=" * 50)
    print("  KV Bootstrap — Two-Pass Inference")
    print("=" * 50)

    result = two_pass_bootstrap(
        prompt=args.prompt,
        context=context,
        spec_url=args.spec_url,
        solo_url=args.solo_url,
        light_url=args.light_url,
        max_tokens=args.max_tokens,
        context_size=args.context_size,
    )

    if result:
        print("\n" + "=" * 50)
        print("  Final Output")
        print("=" * 50)
        print(result)


if __name__ == "__main__":
    main()
