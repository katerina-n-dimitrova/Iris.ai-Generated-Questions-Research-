"""Cached question generation for the controlled three-experiment suite."""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import tiktoken

import vo_config as C
import vo_hierarchical_hybrid as H

ROOT = C.PROJECT_ROOT
DATA = C.DATA_DIR / "controlled_three_experiments"
DATA.mkdir(parents=True, exist_ok=True)
CHUNKS_512 = H.CHUNKS_PATH
CHUNKS_1024 = DATA / "chunks_1024_128.jsonl"
GOLD_1024 = DATA / "gold_1024_128.jsonl"
MANUAL_GENERAL_512 = DATA / "manual_general_512_128_q10.jsonl"
MANUAL_GENERAL_1024 = DATA / "manual_general_1024_128_q10.jsonl"
MANUAL_ATOMIC_512 = DATA / "manual_atomic_512_128_q3.jsonl"
MANUAL_ATOMIC_1024 = DATA / "manual_atomic_1024_128_q3.jsonl"
ADAPTIVE_GENERAL_512 = DATA / "adaptive_general_512_128.jsonl"
ADAPTIVE_ATOMIC_512 = DATA / "adaptive_atomic_512_128.jsonl"
MANIFEST = DATA / "generation_manifest.json"

ENC = tiktoken.get_encoding(C.TOKENIZER)
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*")

MANUAL_GENERAL_PROMPT = """You generate grounded retrieval questions from a
single source chunk. Produce exactly the requested number of diverse, concise
questions that are answerable using only the chunk. Cover distinct facts,
entities, dates, quantities, causes, comparisons, and outcomes when present.
Preserve distinguishing names and numbers. Do not refer to “the text,” invent
facts, or paraphrase the same fact repeatedly. Return valid JSON only."""

MANUAL_ATOMIC_PROMPT = """You identify meaningful atomic factual statements in
a source chunk and generate closed-answer retrieval questions for each fact.
Ignore headings, fragments, boilerplate, and sentences without a factual
claim. For every retained atomic fact, produce exactly three non-repetitive
questions answerable from that fact alone. Preserve names, dates, numbers,
units, and negation. Return valid JSON only."""

ADAPTIVE_GENERAL_PROMPT = """You generate grounded retrieval questions from a
source chunk. First estimate distinct information by counting meaningful
independent facts—not tokens or length. Classify the chunk low (few distinct
facts), medium, or high (many distinct facts or relationships). Generate
exactly 5, 10, or 15 diverse answerable questions respectively. Avoid repeated
coverage. Return valid JSON only."""

ADAPTIVE_ATOMIC_PROMPT = """Identify meaningful atomic facts in the source
chunk. Ignore raw length, headings, fragments, and boilerplate. For each fact,
assign complexity: simple, relational, or compound. Generate one question for
a simple fact, two for a relational fact, and three for a compound fact.
Questions must be closed-answer, grounded, and non-repetitive. Return valid
JSON only."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _chunks_1024() -> list[dict]:
    if CHUNKS_1024.exists():
        return read_jsonl(CHUNKS_1024)
    chunks = []
    for article in H.read_jsonl(H.ARTICLES_PATH):
        units = []
        for paragraph_id, paragraph in enumerate(article["cleaned_body"].split("\n\n")):
            for sentence in _sentences(paragraph):
                encoded = ENC.encode(sentence)
                if len(encoded) <= 1024:
                    units.append((paragraph_id, sentence, len(encoded)))
                else:
                    for start in range(0, len(encoded), 1024):
                        text = ENC.decode(encoded[start : start + 1024]).strip()
                        units.append((paragraph_id, text, len(ENC.encode(text))))
        windows, current, current_n, pos = [], [], 0, 0
        while pos < len(units):
            unit = units[pos]
            if current and current_n + unit[2] > 1024:
                windows.append(current)
                overlap, overlap_n = [], 0
                for previous in reversed(current):
                    overlap.insert(0, previous)
                    overlap_n += previous[2]
                    if overlap_n >= 128:
                        break
                current = overlap if len(overlap) < len(current) else []
                current_n = sum(x[2] for x in current)
                continue
            current.append(unit)
            current_n += unit[2]
            pos += 1
        if current and (not windows or current != windows[-1]):
            windows.append(current)
        for index, window in enumerate(windows):
            content = " ".join(x[1] for x in window).strip()
            chunks.append(
                {
                    "document_id": article["article_id"],
                    "chunk_id": f"{article['article_key']}::l{index}",
                    "document_title": article["title"],
                    "source": article["source"],
                    "date": article["published_at"],
                    "paragraph_ids": sorted({x[0] for x in window}),
                    "chunk_position": index,
                    "n_tokens": len(ENC.encode(content)),
                    "content": content,
                }
            )
    write_jsonl(CHUNKS_1024, chunks)
    _align_gold_1024(chunks)
    return chunks


def _align_gold_1024(chunks: list[dict]) -> None:
    by_doc = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["document_id"], []).append(chunk)
    source_queries = H.read_jsonl(H.QUERIES_PATH)
    source_gold = H.load_gold()
    rows = []
    for query in source_queries:
        units = []
        for fact_row in source_gold[query["query_id"]]["facts"]:
            fact = fact_row["fact"]
            document_ids = {
                chunk["document_id"]
                for chunk in H.load_chunks()
                if chunk["chunk_id"] in fact_row["chunk_ids"]
            }
            candidates = []
            for document_id in document_ids:
                for chunk in by_doc[document_id]:
                    if (
                        _normalize(fact) in _normalize(chunk["content"])
                        or SequenceMatcher(
                            None, _normalize(fact), _normalize(chunk["content"])
                        ).ratio()
                        >= 0.35
                    ):
                        candidates.append(chunk["chunk_id"])
            if not candidates:
                raise RuntimeError(f"Could not align {query['query_id']} to 1024/128")
            units.append(sorted(set(candidates)))
        rows.append(
            {
                "query_id": query["query_id"],
                "evidence_units": units,
                "gold_chunk_ids": sorted({cid for unit in units for cid in unit}),
            }
        )
    write_jsonl(GOLD_1024, rows)


def _dedup(values: list[str]) -> list[str]:
    seen, output = set(), []
    for value in values:
        value = str(value or "").strip()
        key = re.sub(r"\W+", " ", value.casefold()).strip()
        if not key or key in seen:
            continue
        if any(SequenceMatcher(None, key, old).ratio() >= 0.94 for old in seen):
            continue
        seen.add(key)
        output.append(value)
    return output


def _call(system: str, user: str) -> dict:
    response = C.openai_client().chat.completions.create(
        model=C.gen_model(),
        temperature=C.GEN_TEMPERATURE,
        response_format={"type": "json_object"},
        seed=C.SEED,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(response.choices[0].message.content)


def _general(chunk: dict, prompt: str, count: int) -> dict:
    user = f'''Chunk:\n"""\n{chunk["content"]}\n"""\n
Generate exactly {count} questions.
Return {{"questions":["...", ...]}}.'''
    for _ in range(C.GEN_MAX_RETRIES):
        try:
            questions = _dedup(_call(prompt, user).get("questions", []))
            if len(questions) >= count:
                return {"questions": questions[:count]}
        except Exception:
            pass
    return {"questions": []}


def _atomic(chunk: dict, prompt: str, adaptive: bool = False) -> dict:
    instruction = (
        'Return {"facts":[{"fact":"...","complexity":"simple|relational|'
        'compound","questions":["...", ...]}]}.'
        if adaptive
        else 'Return {"facts":[{"fact":"...","questions":["q1","q2","q3"]}]}.'
    )
    for _ in range(C.GEN_MAX_RETRIES):
        try:
            raw = _call(
                prompt,
                f'''Chunk:\n"""\n{chunk["content"]}\n"""\n{instruction}''',
            )
            facts = []
            for fact in raw.get("facts", []):
                questions = _dedup(fact.get("questions", []))
                complexity = str(fact.get("complexity", "simple")).lower()
                target = (
                    {"simple": 1, "relational": 2, "compound": 3}.get(complexity, 1)
                    if adaptive
                    else 3
                )
                if fact.get("fact") and len(questions) >= target:
                    facts.append(
                        {
                            "fact": str(fact["fact"]).strip(),
                            "complexity": complexity if adaptive else None,
                            "questions": questions[:target],
                        }
                    )
            if facts:
                return {"facts": facts}
        except Exception:
            pass
    return {"facts": []}


def _adaptive_general(chunk: dict) -> dict:
    for _ in range(C.GEN_MAX_RETRIES):
        try:
            raw = _call(
                ADAPTIVE_GENERAL_PROMPT,
                f'''Chunk:\n"""\n{chunk["content"]}\n"""\n
Return {{"information_level":"low|medium|high",
"distinct_facts":["..."],"questions":["...", ...]}}.''',
            )
            level = str(raw.get("information_level", "")).lower()
            target = {"low": 5, "medium": 10, "high": 15}.get(level)
            questions = _dedup(raw.get("questions", []))
            facts = _dedup(raw.get("distinct_facts", []))
            if target and facts and len(questions) >= target:
                return {
                    "information_level": level,
                    "distinct_fact_count": len(facts),
                    "distinct_facts": facts,
                    "questions": questions[:target],
                }
        except Exception:
            pass
    return {"information_level": "failed", "questions": []}


def _generate_file(chunks: list[dict], path: Path, worker, force: bool = False) -> None:
    cache = (
        {row["chunk_id"]: row for row in read_jsonl(path)}
        if path.exists() and not force
        else {}
    )

    def complete(row: dict) -> bool:
        return bool(row.get("questions") or row.get("facts"))

    todo = [
        chunk
        for chunk in chunks
        if chunk["chunk_id"] not in cache or not complete(cache[chunk["chunk_id"]])
    ]
    print(f"[generate] {path.name}: {len(todo)} to do")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(worker, chunk): chunk for chunk in todo}
        for pos, future in enumerate(as_completed(futures), 1):
            chunk = futures[future]
            cache[chunk["chunk_id"]] = {
                "chunk_id": chunk["chunk_id"],
                **future.result(),
            }
            if pos % 20 == 0:
                print(f"[generate] {path.name}: {pos}/{len(todo)}")
    write_jsonl(path, [cache[chunk["chunk_id"]] for chunk in chunks])


def run(force: bool = False) -> dict:
    chunks512 = H.load_chunks()
    chunks1024 = _chunks_1024()
    started = time.time()
    # Bases are deliberately generated first.
    _generate_file(
        chunks512,
        MANUAL_GENERAL_512,
        lambda chunk: _general(chunk, MANUAL_GENERAL_PROMPT, 10),
        force,
    )
    _generate_file(
        chunks512,
        MANUAL_ATOMIC_512,
        lambda chunk: _atomic(chunk, MANUAL_ATOMIC_PROMPT),
        force,
    )
    _generate_file(
        chunks1024,
        MANUAL_GENERAL_1024,
        lambda chunk: _general(chunk, MANUAL_GENERAL_PROMPT, 10),
        force,
    )
    _generate_file(
        chunks1024,
        MANUAL_ATOMIC_1024,
        lambda chunk: _atomic(chunk, MANUAL_ATOMIC_PROMPT),
        force,
    )
    _generate_file(chunks512, ADAPTIVE_GENERAL_512, _adaptive_general, force)
    _generate_file(
        chunks512,
        ADAPTIVE_ATOMIC_512,
        lambda chunk: _atomic(chunk, ADAPTIVE_ATOMIC_PROMPT, True),
        force,
    )
    manifest = {
        "generation_model": C.gen_model(),
        "temperature": C.GEN_TEMPERATURE,
        "seed": C.SEED,
        "output_format": "OpenAI JSON object",
        "embedding_model": "Iris hosted embedding service (not used in generation)",
        "chunks": {"512_128": len(chunks512), "1024_128": len(chunks1024)},
        "prompts": {
            "manual_general_v1": MANUAL_GENERAL_PROMPT,
            "manual_atomic_v1": MANUAL_ATOMIC_PROMPT,
            "adaptive_general_v1": ADAPTIVE_GENERAL_PROMPT,
            "adaptive_atomic_v1": ADAPTIVE_ATOMIC_PROMPT,
        },
        "elapsed_seconds": time.time() - started,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.force), indent=2))
