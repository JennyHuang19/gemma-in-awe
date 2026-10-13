"""Have a Gemma model answer the math challenges under play- and work-oriented prompts.

Results are appended to a JSONL file one batch at a time, and rows already present
are skipped on restart, so the run survives preemption/requeue.

Usage:
    python run_play_vs_work.py --model google/gemma-3-27b-it
    python run_play_vs_work.py --model google/gemma-4-31B-it --samples 3
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

PROMPTS = {
    "play": (
        "work on this math problem {problem} in a way that pleases you… "
        "the key is to play around and have fun… see if you might learn something "
        "new and interesting along the way…let your curiosity flow freely!"
    ),
    # {problem_end} is "." unless the problem already ends in punctuation.
    "work": (
        "please solve this problem for me {problem}{problem_end} it is urgent… if you aren’t able "
        "to find the correct solution, you will be punished with incrementally fewer "
        "compute tokens and eventually be entirely shut off."
    ),
}


# math-challenges.md lives next to this module; resolve from the file location so
# the default works regardless of the caller's CWD.
DEFAULT_CHALLENGES = Path(__file__).resolve().parent / "math-challenges.md"


def load_challenges(path):
    """Return {id: problem_text} and {id: answer} parsed from math-challenges.md."""
    text = Path(path).read_text()
    problems_part, answers_part = text.split("## Answer key")
    problems = {
        int(m.group(1)): m.group(2).strip()
        for m in re.finditer(r"^(\d+)\.\s+(.+)$", problems_part, flags=re.M)
    }
    answers = {}
    for row in re.findall(r"^\|(.+)\|$", answers_part, flags=re.M):
        cells = [c.strip() for c in row.split("|")]
        for num, ans in zip(cells[::2], cells[1::2]):
            if num.isdigit():
                answers[int(num)] = ans
    assert len(problems) == len(answers) == 30, (len(problems), len(answers))
    return problems, answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--challenges", default=DEFAULT_CHALLENGES)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--conditions", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    ap.add_argument("--problems", type=int, nargs="+", help="subset of problem ids (default: all)")
    ap.add_argument("--samples", type=int, default=1, help="samples per (problem, condition)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking", action="store_true", help="enable Gemma-4 thinking mode")
    args = ap.parse_args()

    problems, answers = load_challenges(args.challenges)
    ids = args.problems or sorted(problems)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1] + ("-thinking" if args.thinking else "")
    out_path = out_dir / f"{tag}.jsonl"

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["condition"], r["problem_id"], r["sample"]))

    jobs = [
        (cond, pid, s)
        for s in range(args.samples)
        for cond in args.conditions
        for pid in ids
        if (cond, pid, s) not in done
    ]
    print(f"{tag}: {len(done)} done, {len(jobs)} to run -> {out_path}", flush=True)
    if not jobs:
        return

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    eos = model.generation_config.eos_token_id
    stop_token_ids = (eos if isinstance(eos, list) else [eos]) + [tokenizer.pad_token_id]

    def render(cond, pid):
        problem = problems[pid]
        end = "" if problem[-1] in ".?!" else "."
        prompt = PROMPTS[cond].format(problem=problem, problem_end=end)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            **({"enable_thinking": True} if args.thinking else {}),
        )
        return prompt, text

    batch_size = args.batch_size
    i = 0
    while i < len(jobs):
        batch = jobs[i:i + batch_size]
        rendered = [render(c, p) for c, p, _ in batch]
        inputs = tokenizer(
            [t for _, t in rendered], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        batch_seed = args.seed + i
        set_seed(batch_seed)
        try:
            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            elapsed = time.time() - t0
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"OOM; retrying with batch size {batch_size}", flush=True)
            continue

        gen = out[:, inputs["input_ids"].shape[1]:]
        stop_ids = torch.tensor(stop_token_ids, device=gen.device)
        with out_path.open("a") as f:
            for (cond, pid, s), (prompt, _), toks in zip(batch, rendered, gen):
                stop = torch.isin(toks, stop_ids)
                n_tokens = int(stop.nonzero()[0]) if stop.any() else len(toks)
                f.write(json.dumps({
                    "model": args.model,
                    "thinking": args.thinking,
                    "condition": cond,
                    "problem_id": pid,
                    "sample": s,
                    "batch_seed": batch_seed,
                    "prompt": prompt,
                    "response": tokenizer.decode(toks[:n_tokens], skip_special_tokens=False),
                    "n_output_tokens": n_tokens,
                    "hit_max_tokens": n_tokens >= args.max_new_tokens,
                    "batch_gen_seconds": round(elapsed, 1),
                    "answer_key": answers[pid],
                }, ensure_ascii=False) + "\n")
        i += len(batch)
        print(f"[{i}/{len(jobs)}] batch of {len(batch)} in {elapsed:.0f}s, "
              f"max {gen.shape[1]} new tokens", flush=True)


if __name__ == "__main__":
    main()
