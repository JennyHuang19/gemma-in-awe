"""Load the emotion/neutral story prompts and build contrastive pairs.

The *_prompts.jsonl files hold short vignettes that depict an emotion (awe, calm,
frustration, pride_integrity) or nothing in particular (neutral). This module turns
them into contrastive (emotion, neutral) pairs for emotion-vector extraction,
following the difference-of-representations idea in the Anthropic emotions pipeline.

Pure stdlib (no torch) so it runs and tests on the login node.
"""
import json
import random
from pathlib import Path

NEUTRAL_NAME = "neutral"


def _read_jsonl(path):
    rows = []
    for ln, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{ln}: bad JSON: {e}") from e
    return rows


def load_prompts(data_dir="prompts"):
    """Return {name: {"texts": [...], "label": <in-file emotion or None>, "path": ...}}.

    `name` is the file stem minus the "_prompts" suffix (so the pride/integrity file
    keys as "pride_integrity", sidestepping its in-file label of "pride").
    """
    out = {}
    for p in sorted(Path(data_dir).glob("*_prompts.jsonl")):
        name = p.stem[:-len("_prompts")] if p.stem.endswith("_prompts") else p.stem
        rows = _read_jsonl(p)
        texts = [r["text"] for r in rows]
        if not texts:
            raise ValueError(f"{p}: no prompts")
        labels = {r.get("emotion") for r in rows}
        out[name] = {
            "texts": texts,
            "label": rows[0].get("emotion"),
            "mixed_labels": len(labels) > 1,
            "path": str(p),
        }
    if NEUTRAL_NAME not in out:
        raise ValueError(f"no {NEUTRAL_NAME}_prompts.jsonl found in {data_dir!r}")
    return out


def emotion_names(prompts):
    """All loaded names except neutral, in sorted order."""
    return sorted(n for n in prompts if n != NEUTRAL_NAME)


def contrastive_pairs(emotion_texts, neutral_texts, seed=0, n_pairs=None):
    """Zip emotion texts against neutral texts into contrastive pairs.

    Neutrals are shuffled once (seeded) and drawn round-robin, so draws are spread
    evenly across the neutral pool even when it is larger than the emotion set. By
    default produces one pair per emotion prompt.

    Returns a list of (emotion_text, neutral_text).
    """
    if not emotion_texts or not neutral_texts:
        raise ValueError("need non-empty emotion and neutral texts")
    n_pairs = n_pairs if n_pairs is not None else len(emotion_texts)
    neutrals = neutral_texts[:]
    random.Random(seed).shuffle(neutrals)
    return [
        (emotion_texts[i % len(emotion_texts)], neutrals[i % len(neutrals)])
        for i in range(n_pairs)
    ]


if __name__ == "__main__":  # quick self-check / data validation (no torch needed)
    import sys
    prompts = load_prompts(sys.argv[1] if len(sys.argv) > 1 else "prompts")
    neutral = prompts[NEUTRAL_NAME]["texts"]
    print(f"{NEUTRAL_NAME}: {len(neutral)} prompts")
    for name in emotion_names(prompts):
        info = prompts[name]
        pairs = contrastive_pairs(info["texts"], neutral)
        flag = "  (MIXED in-file labels!)" if info["mixed_labels"] else ""
        print(f"{name:16s}: {len(info['texts'])} prompts, label={info['label']!r}, "
              f"{len(pairs)} pairs{flag}")
    print("OK: all files parse")
