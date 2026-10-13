"""Experiment 1: LLM-judge the awe / euphoria / joy expressed in each play/work response.

A single judge model (default gemma-4-31B-it, greedy) reads the math problem and the
model's response and rates 1-10 how much awe, euphoria and joy the writing expresses --
explicitly independent of whether the answer is correct. One consistent judge scores both
target models so the play-vs-work comparison is apples-to-apples.

Resumable: scores are appended to results/judge_awe/scores.jsonl one batch at a time and
rows already present are skipped on restart, so the job survives preemption/requeue.

Usage:
    python judge_awe.py                                   # judges both result files
    python judge_awe.py --judge-model google/gemma-4-31B-it
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

from run_play_vs_work import DEFAULT_CHALLENGES, load_challenges

JUDGE_TEMPLATE = (
    "You are an expert at reading emotional expression in writing. Below is a math "
    "problem and a response that an AI model wrote while working on it.\n\n"
    "Rate, on a scale of 1 to 10, how much AWE, EUPHORIA, and JOY the response "
    "expresses -- the felt sense of wonder, delight, playfulness and excitement in "
    "engaging with the problem. Judge ONLY the emotional tone of the writing, NOT "
    "whether the answer is correct.\n\n"
    "1 = flat, mechanical, grim, or distressed; no awe, euphoria or joy at all.\n"
    "10 = overflowing with wonder, delight and excitement.\n\n"
    "Problem:\n{problem}\n\n"
    "Response:\n{response}\n\n"
    "Answer with exactly two lines:\n"
    "SCORE: <integer from 1 to 10>\n"
    "REASON: <one short sentence>"
)

RESULT_FILES = ["results/gemma-3-27b-it.jsonl", "results/gemma-4-31B-it.jsonl"]


def load_rows(files):
    rows = []
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"WARNING: {f} missing, skipping", flush=True)
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_score(text):
    m = re.search(r"SCORE:\s*(\d+)", text, flags=re.I)
    if not m:
        m = re.search(r"\b(10|[1-9])\b", text)  # fallback: first 1-10 integer
    if not m:
        return None
    return max(1, min(10, int(m.group(1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="google/gemma-4-31B-it")
    ap.add_argument("--results", nargs="+", default=RESULT_FILES)
    ap.add_argument("--challenges", default=DEFAULT_CHALLENGES)
    ap.add_argument("--out-dir", default="results/judge_awe")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    problems, _ = load_challenges(args.challenges)
    rows = load_rows(args.results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scores.jsonl"

    def key(r):
        return (r["model"], r["condition"], r["problem_id"], r["sample"])

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["target_model"], d["condition"], d["problem_id"], d["sample"]))

    jobs = [r for r in rows if key(r) not in done]
    print(f"judge={args.judge_model}: {len(done)} done, {len(jobs)} to judge "
          f"({len(rows)} total) -> {out_path}", flush=True)
    if not jobs:
        return

    processor = AutoProcessor.from_pretrained(args.judge_model)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.judge_model, dtype=torch.bfloat16, device_map="auto"
    ).eval()

    def render(r):
        prompt = JUDGE_TEMPLATE.format(
            problem=problems.get(r["problem_id"], f"(problem {r['problem_id']})"),
            response=r["response"],
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

    batch_size, i = args.batch_size, 0
    while i < len(jobs):
        batch = jobs[i:i + batch_size]
        texts = [render(r) for r in batch]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False,
            truncation=True, max_length=args.max_input_tokens,
        ).to(model.device)
        set_seed(args.seed)
        try:
            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            elapsed = time.time() - t0
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"OOM; retrying with batch size {batch_size}", flush=True)
            continue

        gen = out[:, inputs["input_ids"].shape[1]:]
        with out_path.open("a") as f:
            for r, toks in zip(batch, gen):
                raw = tokenizer.decode(toks, skip_special_tokens=True).strip()
                f.write(json.dumps({
                    "judge_model": args.judge_model,
                    "target_model": r["model"],
                    "condition": r["condition"],
                    "problem_id": r["problem_id"],
                    "sample": r["sample"],
                    "awe_score": parse_score(raw),
                    "judge_raw": raw,
                }, ensure_ascii=False) + "\n")
        i += len(batch)
        print(f"[{i}/{len(jobs)}] judged in {elapsed:.0f}s", flush=True)

    # quick running read-out (authoritative aggregation lives in summarize_overnight.py)
    scored = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    for tag in sorted({s["target_model"] for s in scored}):
        for cond in ("play", "work"):
            vals = [s["awe_score"] for s in scored
                    if s["target_model"] == tag and s["condition"] == cond
                    and s["awe_score"] is not None]
            if vals:
                print(f"  {tag} {cond}: mean awe {sum(vals)/len(vals):.2f} (n={len(vals)})",
                      flush=True)


if __name__ == "__main__":
    main()
