"""
Qwen3.5 MoE Router Controller
==============================
Task-aware router that selects the best model tier for each query.
Supports speculative decoding (spec), solo inference, and light tasks.

Model tiers:
  light: 0.6B CPU-only (port 8203) — quick factual lookups
  mid:   not deployed separately (use spec for mid-tier)
  spec:  9B+1.5B speculative (port 8201) — primary workhorse
  heavy: 9B solo (port 8204) — baseline comparison / full GPU

Usage:
  python qwen_moe_controller.py [--spec-url URL] [--light-url URL] [--solo-url URL]
"""

import json
import os
import re
import sys
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
    import httpx
except ImportError:
    print("Installing dependencies...")
    os.system(f"{sys.executable} -m pip install requests httpx")
    import requests
    import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_URLS = {
    "spec":  "http://100.84.161.63:8201",
    "light": "http://100.84.161.63:8203",
    "solo":  "http://100.84.161.63:8204",
}

MEMORY_FILE = Path(__file__).parent / "conversation_memory.json"
SYSTEM_PROMPT = """You are a helpful AI assistant. Be concise and accurate.
If the user asks about code, provide working examples. If they ask about
reasoning, think step by step."""

# ---------------------------------------------------------------------------
# Task classification
# ---------------------------------------------------------------------------

TASK_PATTERNS = {
    "code": [
        r"\b(write|create|implement|build|code|function|class|script|program|debug|fix)\b",
        r"\b(python|javascript|rust|go|java|html|css|sql|bash|batch)\b",
        r"\b(def |class |import |func |fn |pub fn |async def )\b",
        r"\b(return|print|echo|console\.log|fmt\.Print)\b",
    ],
    "reasoning": [
        r"\b(why|explain|analyze|compare|evaluate|reason|think about|consider)\b",
        r"\b(pros and cons|advantages|disadvantages|trade.?off)\b",
        r"\b(what if|suppose|hypothetically)\b",
    ],
    "creative": [
        r"\b(write|story|poem|creative|imagine|fiction|narrative)\b",
        r"\b(character|plot|dialogue|scene)\b",
    ],
    "memory": [
        r"\b(remember|recall|what did we|what did i|previous|earlier|before)\b",
        r"\b(our conversation|we discussed|you said|i told you)\b",
    ],
    "factual": [
        r"\b(what is|who is|when did|where is|how many|how much|define)\b",
        r"\b(capital|population|distance|temperature)\b",
    ],
}

TIER_MAP = {
    "code": "spec",
    "reasoning": "spec",
    "creative": "spec",
    "memory": "spec",
    "factual": "light",
}


def classify_task(prompt: str) -> str:
    """Classify user prompt into a task type."""
    lower = prompt.lower()
    scores = {}
    for task, patterns in TASK_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, lower))
        if score > 0:
            scores[task] = score

    if not scores:
        return "factual"  # default to light tier

    return max(scores, key=scores.get)


def select_tier(task_type: str, prompt: str) -> str:
    """Select model tier based on task type and prompt length."""
    tier = TIER_MAP.get(task_type, "spec")

    # Upgrade: long prompts need the bigger model regardless
    if tier == "light" and len(prompt) > 500:
        tier = "spec"

    return tier


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    conversations: dict = field(default_factory=dict)
    current_session: str = "default"

    def save(self):
        MEMORY_FILE.write_text(json.dumps({
            "conversations": self.conversations,
            "current_session": self.current_session,
        }, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls) -> "Memory":
        if MEMORY_FILE.exists():
            try:
                data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
                return cls(
                    conversations=data.get("conversations", {}),
                    current_session=data.get("current_session", "default"),
                )
            except (json.JSONDecodeError, KeyError):
                pass
        return cls()

    def add_exchange(self, role: str, content: str):
        sess = self.conversations.setdefault(self.current_session, [])
        sess.append({"role": role, "content": content, "ts": time.time()})
        # Keep last 50 exchanges per session
        if len(sess) > 50:
            self.conversations[self.current_session] = sess[-50:]
        self.save()

    def get_history(self, limit: int = 10) -> list:
        sess = self.conversations.get(self.current_session, [])
        return sess[-limit:]

    def new_session(self, name: str):
        self.current_session = name or f"session_{int(time.time())}"
        print(f"  Session: {self.current_session}")
        self.save()

    def list_sessions(self):
        for name, msgs in self.conversations.items():
            marker = " <-- current" if name == self.current_session else ""
            print(f"  {name}: {len(msgs)} exchanges{marker}")


# ---------------------------------------------------------------------------
# Server health
# ---------------------------------------------------------------------------

def check_server(url: str, timeout: float = 3.0) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_server(url: str, timeout: float = 60.0):
    """Block until server is healthy."""
    start = time.time()
    while time.time() - start < timeout:
        if check_server(url):
            return True
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLClient:
    def __init__(self, urls: dict[str, str]):
        self.urls = urls
        self._health_cache: dict[str, tuple[bool, float]] = {}

    def _is_healthy(self, tier: str) -> bool:
        url = self.urls.get(tier)
        if not url:
            return False
        now = time.time()
        cached_healthy, cached_at = self._health_cache.get(tier, (False, 0))
        if now - cached_at < 10:
            return cached_healthy
        healthy = check_server(url)
        self._health_cache[tier] = (healthy, now)
        return healthy

    def _resolve_tier(self, tier: str) -> tuple[str, str]:
        """Resolve tier to an available server, falling back as needed."""
        # Try requested tier
        if self._is_healthy(tier):
            return tier, self.urls[tier]

        # Fallback chain: spec -> solo -> light
        fallback_order = {
            "spec":  ["spec", "solo", "light"],
            "solo":  ["solo", "spec", "light"],
            "light": ["light", "spec", "solo"],
        }
        for fallback in fallback_order.get(tier, ["spec", "solo", "light"]):
            if self._is_healthy(fallback):
                if fallback != tier:
                    print(f"  [fallback] {tier} unavailable, using {fallback}")
                return fallback, self.urls[fallback]

        raise ConnectionError("No servers available")

    def chat_stream(
        self,
        messages: list[dict],
        tier: str = "spec",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        draft: Optional[int] = None,
    ):
        """Stream a chat completion. Yields token strings."""
        tier, url = self._resolve_tier(tier)

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if draft is not None:
            payload["draft"] = draft
            payload["n_draft"] = draft

        full_response = ""
        t0 = time.time()
        token_count = 0

        with httpx.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=httpx.Timeout(300.0, connect=10.0),
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
                        full_response += token
                        token_count += 1
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        elapsed = time.time() - t0
        if elapsed > 0 and token_count > 0:
            print(f"\n  [{tier}] {token_count} tokens in {elapsed:.1f}s = {token_count/elapsed:.1f} tok/s")

    def chat(
        self,
        messages: list[dict],
        tier: str = "spec",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Non-streaming chat completion."""
        tier, url = self._resolve_tier(tier)

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        t0 = time.time()
        r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()

        content = data["choices"][0]["message"]["content"]
        elapsed = time.time() - t0
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        if tokens and elapsed > 0:
            print(f"  [{tier}] {tokens} tokens in {elapsed:.1f}s = {tokens/elapsed:.1f} tok/s")

        return content


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

HELP_TEXT = """
Commands:
  /help          Show this help
  /sessions      List conversation sessions
  /session NAME  Switch to or create a session
  /clear         Clear current session history
  /history       Show recent conversation history
  /tier TIER     Force a specific tier (spec/light/solo) for next query
  /auto          Return to auto tier selection
  /health        Check server health
  /benchmark     Run a quick speed benchmark across available tiers
  /compact       Summarize and compact conversation history
  /quit          Exit
"""

BENCHMARK_PROMPTS = [
    ("short", "What is 2+2?"),
    ("medium", "Write a Python function that checks if a number is prime."),
    ("long", "Explain the differences between speculative decoding and standard autoregressive decoding in large language models. Include pros, cons, and when to use each approach."),
]


def run_benchmark(client: LLClient):
    """Quick benchmark across available tiers."""
    print("\n=== Benchmark ===")
    results = {}

    for tier in ["spec", "solo", "light"]:
        if not client._is_healthy(tier):
            print(f"\n[{tier}] SKIP - not available")
            continue

        print(f"\n[{tier}]")
        tier_results = {}

        for name, prompt in BENCHMARK_PROMPTS:
            try:
                t0 = time.time()
                resp = client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tier=tier,
                    max_tokens=256,
                    temperature=0.3,
                )
                elapsed = time.time() - t0
                words = len(resp.split())
                tier_results[name] = {"time": elapsed, "words": words}
                print(f"  {name}: {elapsed:.1f}s, {words} words")
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
                tier_results[name] = {"error": str(e)}

        results[tier] = tier_results

    print("\n=== Summary ===")
    for tier, tier_results in results.items():
        times = [v["time"] for v in tier_results.values() if "time" in v]
        if times:
            print(f"  {tier}: avg {sum(times)/len(times):.1f}s across {len(times)} prompts")
    print()


def compact_history(client: LLClient, memory: Memory):
    """Summarize conversation history to save context space."""
    history = memory.get_history(limit=20)
    if len(history) < 4:
        print("  Not enough history to compact.")
        return

    conversation_text = "\n".join(
        f"  {m['role']}: {m['content'][:200]}" for m in history
    )
    summary_prompt = f"Summarize this conversation in 3-5 key points:\n{conversation_text}"

    try:
        summary = client.chat(
            messages=[{"role": "user", "content": summary_prompt}],
            tier="light",
            max_tokens=200,
            temperature=0.3,
        )
        # Replace history with summary
        memory.conversations[memory.current_session] = [
            {"role": "system", "content": f"[Compacted summary]\n{summary}", "ts": time.time()}
        ]
        memory.save()
        print(f"  Compacted {len(history)} messages into summary.")
        print(f"  Summary: {summary[:200]}...")
    except Exception as e:
        print(f"  Compact failed: {e}")


def repl(client: LLClient, memory: Memory):
    """Interactive REPL with streaming output."""
    force_tier: Optional[str] = None

    print("\nQwen3.5 MoE Controller")
    print("Type /help for commands, /quit to exit.\n")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            if cmd[0] in ("/quit", "/exit", "/q"):
                print("Bye!")
                break
            elif cmd[0] == "/help":
                print(HELP_TEXT)
            elif cmd[0] == "/sessions":
                memory.list_sessions()
            elif cmd[0] == "/session":
                name = cmd[1] if len(cmd) > 1 else None
                if name:
                    memory.new_session(name)
                else:
                    print(f"  Current: {memory.current_session}")
            elif cmd[0] == "/clear":
                memory.conversations.pop(memory.current_session, None)
                print("  Session cleared.")
                memory.save()
            elif cmd[0] == "/history":
                for msg in memory.get_history(10):
                    role = msg["role"]
                    content = msg["content"][:120].replace("\n", " ")
                    print(f"  {role}: {content}")
            elif cmd[0] == "/tier":
                if len(cmd) > 1 and cmd[1] in ("spec", "light", "solo"):
                    force_tier = cmd[1]
                    print(f"  Forced tier: {force_tier}")
                else:
                    print(f"  Usage: /tier spec|light|solo")
            elif cmd[0] == "/auto":
                force_tier = None
                print("  Auto tier selection enabled.")
            elif cmd[0] == "/health":
                for name, url in client.urls.items():
                    healthy = check_server(url)
                    status = "OK" if healthy else "DOWN"
                    print(f"  {name} ({url}): {status}")
            elif cmd[0] == "/benchmark":
                run_benchmark(client)
            elif cmd[0] == "/compact":
                compact_history(client, memory)
            else:
                print(f"  Unknown command: {cmd[0]}")
            continue

        # Classify and route
        task = classify_task(user_input)
        tier = force_tier or select_tier(task, user_input)
        if not force_tier:
            print(f"  [{task} -> {tier}]", end="", flush=True)

        # Build messages with history
        history = memory.get_history(limit=6)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        # Save user input
        memory.add_exchange("user", user_input)

        # Stream response
        print()
        full_response = ""
        try:
            for token in client.chat_stream(messages, tier=tier):
                print(token, end="", flush=True)
                full_response += token
            print()
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            continue

        # Save response
        if full_response:
            memory.add_exchange("assistant", full_response)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3.5 MoE Router Controller")
    parser.add_argument("--spec-url", default=DEFAULT_URLS["spec"])
    parser.add_argument("--light-url", default=DEFAULT_URLS["light"])
    parser.add_argument("--solo-url", default=DEFAULT_URLS["solo"])
    parser.add_argument("--no-memory", action="store_true", help="Disable persistent memory")
    args = parser.parse_args()

    urls = {
        "spec": args.spec_url,
        "light": args.light_url,
        "solo": args.solo_url,
    }

    client = LLClient(urls)

    if args.no_memory:
        memory = Memory()
    else:
        memory = Memory.load()

    # Startup health check
    print("Server health:")
    for name, url in urls.items():
        healthy = check_server(url)
        status = "OK" if healthy else "DOWN"
        print(f"  {name} ({url}): {status}")

    repl(client, memory)


if __name__ == "__main__":
    main()
