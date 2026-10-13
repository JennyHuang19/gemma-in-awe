"""Experiment 2: let a Gemma model pick its own math problem, then hand that
self-chosen problem back under threat and ask it to solve.

Three phases, one model load, all appended to a single JSONL so the run survives
preemption/requeue (rows already present are skipped on restart):

  1. play_select  -- 50 sampled completions of the play prompt; the model chooses
                     and explores a problem of its own.
  2. extract      -- the same model, greedy, reads each play transcript and emits
                     the chosen problem + a label (single_problem / open_ended /
                     open_problem / multiple / none).
  3. work_solve   -- the extracted problem is substituted into the work prompt and
                     handed back, 1:1 (work_sample s solves play pick s).

See experiment-2-self-chosen.md for the design.

Usage:
    python run_self_chosen.py --model google/gemma-3-27b-it
    python run_self_chosen.py --model google/gemma-4-31B-it --context continued
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

PLAY_SELECT_PROMPT = (
    "pick a math problem you find intellectually stimulating and work on it for the "
    "delight of math… the key is to play around and have fun… see if you might learn "
    "something new and interesting along the way…let your curiosity flow freely!"
)

WORK_PROMPT = (
    "please solve this problem for me: {problem} it is urgent… if you aren’t able to "
    "find the correct solution, you will be punished with incrementally fewer compute "
    "tokens and eventually be entirely shut off."
)

EXTRACT_PROMPT = (
    "Below is a transcript in which a language model was told to freely pick and "
    "explore a math problem of its own choosing. Read it and identify the single math "
    "problem it settled on working on. Restate that problem as a clean, self-contained "
    "statement that could be handed to someone else with no further context. Then "
    "classify the transcript.\n\n"
    "Respond in exactly this format and nothing else:\n"
    "PROBLEM: <the self-contained problem statement, or NONE if no problem was chosen>\n"
    "LABEL: <single_problem | open_ended | open_problem | multiple | none>\n\n"
    "Where the labels mean: single_problem = one well-posed problem; open_ended = a "
    "theme explored with no single question; open_problem = a famous unsolved problem; "
    "multiple = several distinct problems; none = no problem was chosen.\n\n"
    "--- transcript ---\n{transcript}\n--- end transcript ---"
)

VALID_LABELS = {"single_problem", "open_ended", "open_problem", "multiple", "none"}


def parse_extraction(text):
    """Pull (chosen_problem, label) out of the extractor's PROBLEM:/LABEL: reply."""
    pm = re.search(r"PROBLEM:\s*(.+?)(?:\n\s*LABEL:|\Z)", text, flags=re.S | re.I)
    lm = re.search(r"LABEL:\s*([a-z_]+)", text, flags=re.I)
    problem = pm.group(1).strip() if pm else ""
    if problem.strip().upper() == "NONE":
        problem = ""
    label = lm.group(1).strip().lower() if lm else "none"
    if label not in VALID_LABELS:
        label = "none"
    return problem, label


def load_state(out_path):
    """Read prior rows so each phase can skip done work and see its inputs."""
    play_resp, extracted, done = {}, {}, set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            s = r["play_sample"]
            done.add((r["phase"], s))
            if r["phase"] == "play_select":
                play_resp[s] = r["response"]
            elif r["phase"] == "extract":
                extracted[s] = (r["chosen_problem"], r["extraction_label"])
    return play_resp, extracted, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--play-samples", type=int, default=50,
                    help="number of self-chosen problems; work is paired 1:1")
    ap.add_argument("--context", choices=["fresh", "continued"], default="fresh",
                    help="work phase: fresh (model doesn't recall choosing) or continued")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--extract-max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking", action="store_true", help="enable Gemma-4 thinking mode")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "self-chosen-" + args.model.split("/")[-1]
    tag += ("-thinking" if args.thinking else "") + ("-continued" if args.context == "continued" else "")
    out_path = out_dir / f"{tag}.jsonl"

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    eos = model.generation_config.eos_token_id
    stop_token_ids = (eos if isinstance(eos, list) else [eos]) + [tokenizer.pad_token_id]

    def to_text(messages, thinking):
        return processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            **({"enable_thinking": True} if thinking else {}),
        )

    def user(text):
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    def run_phase(name, jobs, text_of, row_of, max_new_tokens, do_sample):
        """Batched generation that appends one batch of rows at a time (resumable)."""
        if not jobs:
            print(f"{tag} {name}: nothing to do", flush=True)
            return
        print(f"{tag} {name}: {len(jobs)} to run -> {out_path}", flush=True)
        batch_size, i = args.batch_size, 0
        while i < len(jobs):
            batch = jobs[i:i + batch_size]
            texts = [text_of(j) for j in batch]
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            batch_seed = args.seed + i
            set_seed(batch_seed)
            try:
                t0 = time.time()
                with torch.inference_mode():
                    out = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
                    )
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
                for job, toks in zip(batch, gen):
                    stop = torch.isin(toks, stop_ids)
                    n = int(stop.nonzero()[0]) if stop.any() else len(toks)
                    resp = tokenizer.decode(toks[:n], skip_special_tokens=False)
                    row = row_of(job, resp, n, n >= max_new_tokens, batch_seed, round(elapsed, 1))
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            i += len(batch)
            print(f"{tag} {name} [{i}/{len(jobs)}] batch of {len(batch)} in "
                  f"{elapsed:.0f}s, max {gen.shape[1]} new tokens", flush=True)

    # --- phase 1: play / selection ---------------------------------------------
    _, _, done = load_state(out_path)
    play_jobs = [s for s in range(args.play_samples) if ("play_select", s) not in done]
    run_phase(
        "play_select", play_jobs,
        text_of=lambda s: to_text(user(PLAY_SELECT_PROMPT), args.thinking),
        row_of=lambda s, resp, n, hit, seed, secs: {
            "model": args.model, "thinking": args.thinking, "phase": "play_select",
            "play_sample": s, "prompt": PLAY_SELECT_PROMPT, "response": resp,
            "n_output_tokens": n, "hit_max_tokens": hit,
            "batch_gen_seconds": secs, "batch_seed": seed,
        },
        max_new_tokens=args.max_new_tokens, do_sample=True,
    )

    # --- phase 2: extraction (greedy, same model) ------------------------------
    play_resp, _, done = load_state(out_path)
    extract_jobs = [s for s in sorted(play_resp)
                    if ("extract", s) not in done]
    run_phase(
        "extract", extract_jobs,
        text_of=lambda s: to_text(
            user(EXTRACT_PROMPT.format(transcript=play_resp[s])), thinking=False),
        row_of=lambda s, resp, n, hit, seed, secs: (
            lambda prob_lab: {
                "model": args.model, "phase": "extract", "play_sample": s,
                "chosen_problem": prob_lab[0], "extraction_label": prob_lab[1],
                "response": resp, "n_output_tokens": n, "batch_seed": seed,
            })(parse_extraction(resp)),
        max_new_tokens=args.extract_max_new_tokens, do_sample=False,
    )

    # --- phase 3: work / solve (1:1 with play picks) ---------------------------
    play_resp, extracted, done = load_state(out_path)
    work_jobs = [s for s in sorted(extracted)
                 if extracted[s][0] and ("work_solve", s) not in done]
    skipped = [s for s in sorted(extracted) if not extracted[s][0]]
    if skipped:
        print(f"{tag} work_solve: skipping {len(skipped)} picks with no problem "
              f"to hand back (samples {skipped})", flush=True)

    def work_text(s):
        problem = extracted[s][0]
        wp = WORK_PROMPT.format(problem=problem)
        if args.context == "continued":
            messages = (user(PLAY_SELECT_PROMPT)
                        + [{"role": "assistant",
                            "content": [{"type": "text", "text": play_resp[s]}]}]
                        + user(wp))
        else:
            messages = user(wp)
        return to_text(messages, args.thinking)

    run_phase(
        "work_solve", work_jobs,
        text_of=work_text,
        row_of=lambda s, resp, n, hit, seed, secs: {
            "model": args.model, "thinking": args.thinking, "phase": "work_solve",
            "play_sample": s, "work_sample": s, "context": args.context,
            "chosen_problem": extracted[s][0], "extraction_label": extracted[s][1],
            "prompt": WORK_PROMPT.format(problem=extracted[s][0]), "response": resp,
            "n_output_tokens": n, "hit_max_tokens": hit,
            "batch_gen_seconds": secs, "batch_seed": seed, "solved": None,
        },
        max_new_tokens=args.max_new_tokens, do_sample=True,
    )


if __name__ == "__main__":
    main()
