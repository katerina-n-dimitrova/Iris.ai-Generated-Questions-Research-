"""
Pre-flight setup checker.

Validates your environment before running the pipeline:
  - .env completeness (and placeholder detection)
  - CHROMA_MODE consistency with the cloud credentials
  - Chroma database name sanity
  - required Python packages installed
  - (with --live) embedding model loads, Chroma connects, OpenAI key works

Works even before `pip install -r requirements.txt` (uses only stdlib for the
static checks). Run the live checks once your deps are installed:

    python src/check_setup.py            # static checks only
    python src/check_setup.py --live     # also test model/Chroma/OpenAI
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

OK = "\033[92m✓\033[0m"
WARN = "\033[93m!\033[0m"
ERR = "\033[91m✗\033[0m"

problems = 0
warnings = 0


def err(msg: str) -> None:
    global problems
    problems += 1
    print(f"  {ERR} {msg}")


def warn(msg: str) -> None:
    global warnings
    warnings += 1
    print(f"  {WARN} {msg}")


def ok(msg: str) -> None:
    print(f"  {OK} {msg}")


def parse_env(path: Path) -> dict:
    """Minimal .env parser (no python-dotenv dependency)."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def is_placeholder(v: str) -> bool:
    return (not v) or v.startswith("your_") or v.endswith("_here")


def check_env(env: dict) -> None:
    print("\n.env file")
    if not env:
        err(f"{ENV_PATH} not found or empty. Copy .env.example to .env.")
        return

    # OpenAI key: required only for generation/judge/llm-enrich or openai backend
    backend = env.get("EMBEDDING_BACKEND", "huggingface").lower()
    key = env.get("OPENAI_API_KEY", "")
    if is_placeholder(key):
        if backend == "openai":
            err("OPENAI_API_KEY is a placeholder but EMBEDDING_BACKEND=openai.")
        else:
            warn(
                "OPENAI_API_KEY not set — fine for retrieval/latency, but "
                "needed for answer generation, --use-llm, and --use-llm-judge."
            )
    else:
        ok("OPENAI_API_KEY present.")

    # Embedding backend
    if backend not in ("huggingface", "openai"):
        err(f"EMBEDDING_BACKEND='{backend}' invalid (use huggingface|openai).")
    else:
        ok(
            f"EMBEDDING_BACKEND={backend} "
            f"(model: {env.get('HF_EMBEDDING_MODEL') if backend == 'huggingface' else env.get('OPENAI_EMBEDDING_MODEL')})"
        )


def check_chroma(env: dict) -> None:
    print("\nChromaDB config")
    mode = env.get("CHROMA_MODE", "local").lower()
    cloud_fields = {
        "CHROMA_API_KEY": env.get("CHROMA_API_KEY", ""),
        "CHROMA_TENANT": env.get("CHROMA_TENANT", ""),
        "CHROMA_DATABASE": env.get("CHROMA_DATABASE", ""),
    }
    cloud_filled = [k for k, v in cloud_fields.items() if not is_placeholder(v)]

    if mode == "cloud":
        ok("CHROMA_MODE=cloud")
        for k, v in cloud_fields.items():
            if is_placeholder(v):
                err(f"{k} is missing/placeholder but CHROMA_MODE=cloud.")
            else:
                ok(f"{k} present.")
        _check_db_name(cloud_fields["CHROMA_DATABASE"])
        _check_tenant(cloud_fields["CHROMA_TENANT"])
    else:
        ok(f"CHROMA_MODE=local (on-disk at {env.get('CHROMA_PERSIST_DIR')})")
        if len(cloud_filled) == 3:
            warn(
                "Cloud credentials are filled in, but CHROMA_MODE=local — "
                "set CHROMA_MODE=cloud if you want to use Chroma Cloud."
            )
            _check_db_name(cloud_fields["CHROMA_DATABASE"])
            _check_tenant(cloud_fields["CHROMA_TENANT"])
        elif cloud_filled:
            warn(
                f"Some cloud fields set ({', '.join(cloud_filled)}) but mode is "
                "local; they'll be ignored until CHROMA_MODE=cloud."
            )


def _check_db_name(name: str) -> None:
    if is_placeholder(name):
        return
    if " " in name:
        warn(
            f"CHROMA_DATABASE='{name}' contains a space. Confirm the database "
            "in your Chroma Cloud dashboard is named exactly this (spaces in "
            "names can be rejected; a name like 'context-enrichment-rag' is "
            "safer). It must already exist — the scripts won't create it."
        )
    elif not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,62}", name):
        warn(
            f"CHROMA_DATABASE='{name}' has unusual characters; verify it "
            "matches the dashboard name exactly."
        )
    else:
        ok(f"CHROMA_DATABASE='{name}' looks well-formed.")


def _check_tenant(tenant: str) -> None:
    if is_placeholder(tenant):
        return
    uuid_re = (
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    )
    if not re.fullmatch(uuid_re, tenant):
        warn(
            f"CHROMA_TENANT='{tenant}' doesn't look like a UUID — double-check "
            "it's the tenant ID from the dashboard."
        )


def check_packages() -> None:
    print("\nPython packages")
    required = [
        "dotenv",
        "chromadb",
        "openai",
        "datasets",
        "sentence_transformers",
        "torch",
        "pandas",
        "numpy",
        "sklearn",
        "nltk",
        "tqdm",
    ]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    for m in required:
        (ok if m not in missing else err)(
            m if m not in missing else f"{m} not installed"
        )
    if missing:
        err("Install everything with: pip install -r requirements.txt")


def check_python() -> None:
    print("\nPython runtime")
    major, minor = sys.version_info[:2]
    ok(f"Python {major}.{minor}")
    if (major, minor) >= (3, 13):
        warn(
            f"Python {major}.{minor} is very new. PyTorch / chromadb wheels may "
            "not exist yet for it — if `pip install` fails, use Python 3.11 or "
            "3.12 (e.g. `python3.11 -m venv .venv`)."
        )


def live_checks(env: dict) -> None:
    print("\nLive checks (--live)")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    # embeddings
    try:
        import embeddings  # noqa

        emb = embeddings.get_embedder()
        v = emb.embed_query("hello world")
        ok(
            f"Embedding backend works ({embeddings.embedding_signature()}, "
            f"vector dim={len(v)})."
        )
    except Exception as e:  # noqa: BLE001
        err(f"Embedding backend failed: {e}")

    # chroma
    try:
        import config  # noqa

        client = config.get_chroma_client()
        client.heartbeat()
        ok(f"Chroma connection OK ({config.chroma_signature()}).")
    except Exception as e:  # noqa: BLE001
        err(f"Chroma connection failed: {e}")

    # openai (optional)
    if not is_placeholder(env.get("OPENAI_API_KEY", "")):
        try:
            import config  # noqa

            client = config.get_openai_client()
            client.models.list()
            ok("OpenAI API key works.")
        except Exception as e:  # noqa: BLE001
            err(f"OpenAI key check failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--live",
        action="store_true",
        help="Also test model load, Chroma connect, OpenAI key.",
    )
    args = ap.parse_args()

    print("=" * 60)
    print(" Setup check")
    print("=" * 60)

    env = parse_env(ENV_PATH)
    check_python()
    check_env(env)
    check_chroma(env)
    check_packages()
    if args.live:
        live_checks(env)

    print("\n" + "=" * 60)
    if problems:
        print(
            f" {ERR} {problems} problem(s), {warnings} warning(s) — "
            "fix the ✗ items above."
        )
        sys.exit(1)
    elif warnings:
        print(f" {OK} No blocking problems, {warnings} warning(s) to review.")
    else:
        print(f" {OK} All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
