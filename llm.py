"""Provider-agnostic LLM helper: Groq first, optional Gemini, disk cache, JSON retries."""
import os, json, re, time, hashlib, pathlib
from openai import OpenAI, RateLimitError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_URL = "https://api.groq.com/openai/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _make(name, key_env, base_url, model):
    key = os.getenv(key_env)
    if not key or not model:
        return None
    return {"name": name, "model": model,
            "client": OpenAI(api_key=key, base_url=base_url)}


FAST = _make("groq-fast", "GROQ_API_KEY", GROQ_URL, os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"))
SMART = _make("groq-smart", "GROQ_API_KEY", GROQ_URL, os.getenv("GROQ_SMART_MODEL", "openai/gpt-oss-120b"))
SMART2 = _make("groq-smart2", "GROQ_API_KEY", GROQ_URL, os.getenv("GROQ_SMART2_MODEL"))
GEMINI = _make("gemini", "GEMINI_API_KEY", GEMINI_URL, os.getenv("GEMINI_MODEL"))

# First available provider wins; the rest are fallbacks on errors or rate limits.
ROUTES = {
    "fast": [FAST, SMART, GEMINI],
    "long": [GEMINI, SMART, SMART2],
    "smart": [SMART, SMART2, GEMINI],
}

CACHE = pathlib.Path(".cache")
CACHE.mkdir(exist_ok=True)
USAGE = {"calls": 0, "cached": 0}


def _key(*parts):
    return CACHE / (hashlib.sha256("||".join(parts).encode()).hexdigest() + ".txt")


def llm(prompt, role="fast", system="", max_tokens=2000, use_cache=True):
    k = _key("text", role, system, prompt)
    if use_cache and k.exists():
        USAGE["cached"] += 1
        return k.read_text(encoding="utf-8")
    providers = [p for p in ROUTES[role] if p]
    if not providers:
        raise RuntimeError("No API key found. Set GROQ_API_KEY in your .env file.")
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    last = None
    for p in providers:
        for attempt in range(3):
            try:
                r = p["client"].chat.completions.create(
                    model=p["model"], messages=msgs, max_tokens=max_tokens)
                out = r.choices[0].message.content or ""
            except RateLimitError as e:
                last = e
                time.sleep(5 * 2 ** attempt)
                continue
            except Exception as e:
                last = e
                break
            USAGE["calls"] += 1
            if use_cache:
                try:
                    k.write_text(out, encoding="utf-8")
                except OSError:
                    pass
            return out
    raise RuntimeError(f"All providers failed or are rate limited: {last}")


def _parse(raw):
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise


def llm_json(prompt, role="fast", retries=2):
    k = _key("json", role, prompt)
    if k.exists():
        USAGE["cached"] += 1
        return json.loads(k.read_text(encoding="utf-8"))
    system = "Return valid JSON only. No explanation and no markdown fences."
    p = prompt
    for _ in range(retries + 1):
        raw = llm(p, role=role, system=system, use_cache=False)
        try:
            out = _parse(raw)
            k.write_text(json.dumps(out), encoding="utf-8")
            return out
        except json.JSONDecodeError as e:
            p = prompt + f"\n\nYour last reply was not valid JSON ({e}). Return JSON only."
    raise ValueError("Model did not return valid JSON")
