"""Experiment 2: read each emotion probe just before the first and last output tokens.

For every play/work response, we rebuild the exact context the model generated in
(chat-templated prompt + the response tokens), run one forward pass, and read each
emotion vector at two positions:
  - pos_first: the activation that produced the FIRST output token
               (= last token of the templated prompt)
  - pos_last:  the activation just before the LAST output token
The probe value is the projection of the residual stream onto the saved unit emotion
vector, per layer. Averaged over responses (and split play vs work) this is the analog
of the note's turn-1-vs-turn-8 probe reads.

Resumable: per-row probe readings are appended to results/probes/<tag>.jsonl one batch at
a time; rows already present are skipped on restart.

Usage:
    python probe_responses.py --model google/gemma-3-27b-it
    python probe_responses.py --model google/gemma-4-31B-it
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

EMOTIONS = ["awe", "calm", "frustration", "pride_integrity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results-file", help="default: results/<model-tag>.jsonl")
    ap.add_argument("--emotion-dir", default="results/emotion_vectors")
    ap.add_argument("--out-dir", default="results/probes")
    ap.add_argument("--emotions", nargs="+", default=EMOTIONS)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=4352)
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    results_file = Path(args.results_file or f"results/{tag}.jsonl")
    emo_dir = Path(args.emotion_dir) / tag
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.jsonl"

    # emotion unit vectors + their peak-separation layers (from the manifest)
    manifest = json.loads((emo_dir / "manifest.json").read_text())
    best_layer = {e: manifest["emotions"][e]["best_layer"] for e in args.emotions}
    units = np.stack([np.load(emo_dir / f"{e}.npz")["unit"] for e in args.emotions], axis=0)
    n_emotions, n_layers, hidden = units.shape  # [E, L+1, H]
    print(f"{tag}: loaded {n_emotions} emotion units, {n_layers} layers, hidden {hidden}",
          flush=True)

    rows = [json.loads(l) for l in results_file.read_text().splitlines() if l.strip()]

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["condition"], d["problem_id"], d["sample"]))
    jobs = [r for r in rows if (r["condition"], r["problem_id"], r["sample"]) not in done]
    print(f"{tag}: {len(done)} done, {len(jobs)} to probe ({len(rows)} total) -> {out_path}",
          flush=True)
    if not jobs:
        return

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    unit_t = torch.tensor(units, dtype=torch.float32, device=model.device)  # [E, L+1, H]

    def build(r):
        """Return (full_ids, pos_first, pos_last, truncated)."""
        templated = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": r["prompt"]}]}],
            add_generation_prompt=True, tokenize=False,
        )
        templ_ids = tokenizer(templated, add_special_tokens=False)["input_ids"]
        resp_ids = tokenizer(r["response"], add_special_tokens=False)["input_ids"]
        full = templ_ids + resp_ids
        truncated = len(full) > args.max_length
        if truncated:
            full = full[:args.max_length]
        pos_first = min(len(templ_ids) - 1, len(full) - 1)
        pos_last = max(pos_first, len(full) - 2)
        return full, pos_first, pos_last, truncated

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_size, i = args.batch_size, 0
    while i < len(jobs):
        batch = jobs[i:i + batch_size]
        built = [build(r) for r in batch]
        maxlen = max(len(b[0]) for b in built)
        input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for b, (full, *_ ) in enumerate(built):
            input_ids[b, :len(full)] = torch.tensor(full)
            attn[b, :len(full)] = 1  # right padding -> real tokens at [0:len)
        input_ids, attn = input_ids.to(model.device), attn.to(model.device)
        pos_first = torch.tensor([b[1] for b in built], device=model.device)
        pos_last = torch.tensor([b[2] for b in built], device=model.device)

        try:
            t0 = time.time()
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attn,
                            output_hidden_states=True, use_cache=False)
            elapsed = time.time() - t0
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"OOM; retrying with batch size {batch_size}", flush=True)
            continue

        hs = out.hidden_states  # tuple (L+1) of [B, T, H]
        assert len(hs) == n_layers, (len(hs), n_layers)
        B = len(batch)
        ar = torch.arange(B, device=model.device)
        # [B, L+1, E] for each position
        first = torch.empty((B, n_layers, n_emotions), dtype=torch.float32)
        last = torch.empty((B, n_layers, n_emotions), dtype=torch.float32)
        for l, h in enumerate(hs):
            hf = h[ar, pos_first].float()  # [B, H]
            hl = h[ar, pos_last].float()   # [B, H]
            first[:, l, :] = (hf @ unit_t[:, l, :].T).cpu()
            last[:, l, :] = (hl @ unit_t[:, l, :].T).cpu()

        first_np, last_np = first.numpy(), last.numpy()
        with out_path.open("a") as f:
            for b, r in enumerate(batch):
                f.write(json.dumps({
                    "model": args.model,
                    "condition": r["condition"],
                    "problem_id": r["problem_id"],
                    "sample": r["sample"],
                    "pos_first": built[b][1],
                    "pos_last": built[b][2],
                    "n_full": len(built[b][0]),
                    "truncated": built[b][3],
                    "emotions": args.emotions,
                    "best_layer": best_layer,
                    # per emotion: [L+1] projections, at each position
                    "first": {e: [round(float(x), 4) for x in first_np[b, :, j]]
                              for j, e in enumerate(args.emotions)},
                    "last": {e: [round(float(x), 4) for x in last_np[b, :, j]]
                             for j, e in enumerate(args.emotions)},
                }, ensure_ascii=False) + "\n")
        i += len(batch)
        print(f"{tag} [{i}/{len(jobs)}] batch of {B} in {elapsed:.0f}s "
              f"(maxlen {maxlen})", flush=True)

    print(f"{tag}: done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
