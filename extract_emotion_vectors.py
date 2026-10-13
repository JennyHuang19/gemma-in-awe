"""Extract emotion vectors from a Gemma model by contrasting emotion vs. neutral stories.

For each emotion we run the model over the emotion stories and the neutral stories
(a forward pass, no generation), mean-pool the residual stream over real tokens at
every layer, and take the difference of class means:

    emotion_vector[layer] = mean(emotion activations) - mean(neutral activations)

This is the difference-of-representations direction from the Anthropic emotions
pipeline. The contrastive (emotion, neutral) pairs from emotion_data.py are used for a
paired separation diagnostic; the saved direction itself uses all stories in each
class, so all 40 emotion + 100 neutral prompts contribute.

Output: results/emotion_vectors/<model-tag>/<emotion>.npz with the per-layer raw
direction, the unit direction, its per-layer norm, and the two class means, plus a
manifest.json describing the run. GPU required -> launch with the sbatch.

Usage:
    python extract_emotion_vectors.py --model google/gemma-3-27b-it
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import emotion_data as ed


def pooled_hidden_states(model, tokenizer, texts, batch_size, max_length):
    """Return {text: np.array [n_layers+1, hidden]} of masked mean-pooled activations."""
    acts = {}
    i = 0
    while i < len(texts):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length,
        ).to(model.device)
        try:
            with torch.inference_mode():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"OOM; retrying with batch size {batch_size}", flush=True)
            continue
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("model returned no hidden_states; check the model class")
        mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        counts = mask.sum(dim=1).clamp(min=1)          # [B, 1]
        # stack layers -> [L+1, B, H]
        pooled = torch.stack([(h * mask).sum(dim=1) / counts for h in hs], dim=0)
        pooled = pooled.float().cpu().numpy()          # [L+1, B, H]
        for b, text in enumerate(batch):
            acts[text] = pooled[:, b, :]
        i += len(batch)
        print(f"  pooled {i}/{len(texts)}", flush=True)
    return acts


def class_mean(acts, texts):
    """Mean over the given texts -> [L+1, H]."""
    return np.mean(np.stack([acts[t] for t in texts], axis=0), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="prompts")
    ap.add_argument("--out-dir", default="results/emotion_vectors")
    ap.add_argument("--emotions", nargs="+", help="subset of emotion names (default: all)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prompts = ed.load_prompts(args.data_dir)
    neutral = prompts[ed.NEUTRAL_NAME]["texts"]
    names = args.emotions or ed.emotion_names(prompts)

    tag = args.model.split("/")[-1]
    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    ).eval()

    # One forward pass over every unique story (neutral pool is shared across emotions).
    all_emotion_texts = sorted({t for n in names for t in prompts[n]["texts"]})
    unique_texts = sorted(set(all_emotion_texts) | set(neutral))
    print(f"{tag}: pooling {len(unique_texts)} unique stories "
          f"({len(all_emotion_texts)} emotion + {len(neutral)} neutral)", flush=True)
    t0 = time.time()
    acts = pooled_hidden_states(model, tokenizer, unique_texts, args.batch_size, args.max_length)
    n_layers = next(iter(acts.values())).shape[0]
    print(f"pooled in {time.time() - t0:.0f}s; {n_layers} layers (incl. embeddings)", flush=True)

    neutral_mean = class_mean(acts, neutral)  # [L+1, H]
    manifest = {
        "model": args.model, "pooling": "masked_mean", "max_length": args.max_length,
        "seed": args.seed, "n_layers_incl_embed": n_layers,
        "n_neutral": len(neutral), "data_dir": str(Path(args.data_dir).resolve()),
        "emotions": {},
    }

    for name in names:
        texts = prompts[name]["texts"]
        emo_mean = class_mean(acts, texts)                 # [L+1, H]
        direction = emo_mean - neutral_mean                # [L+1, H]
        norms = np.linalg.norm(direction, axis=1)          # [L+1]
        unit = direction / np.clip(norms[:, None], 1e-8, None)

        # Paired separation diagnostic: project each story onto the unit direction and
        # measure emotion-vs-neutral gap in pooled-std units, per layer. Reported at the
        # layer where it peaks (a quick sanity read on vector quality).
        pairs = ed.contrastive_pairs(texts, neutral, seed=args.seed)
        emo_stack = np.stack([acts[t] for t in texts], axis=0)      # [Ne, L+1, H]
        neu_stack = np.stack([acts[t] for _, t in pairs], axis=0)   # [Np, L+1, H]
        emo_proj = np.einsum("nlh,lh->nl", emo_stack, unit)         # [Ne, L+1]
        neu_proj = np.einsum("nlh,lh->nl", neu_stack, unit)         # [Np, L+1]
        pooled_std = np.sqrt(
            (emo_proj.var(0) * len(emo_proj) + neu_proj.var(0) * len(neu_proj))
            / (len(emo_proj) + len(neu_proj))
        )
        sep = (emo_proj.mean(0) - neu_proj.mean(0)) / np.clip(pooled_std, 1e-8, None)
        best = int(np.argmax(sep))

        np.savez_compressed(
            out_dir / f"{name}.npz",
            direction=direction.astype(np.float32),
            unit=unit.astype(np.float32),
            norms=norms.astype(np.float32),
            emotion_mean=emo_mean.astype(np.float32),
            neutral_mean=neutral_mean.astype(np.float32),
            separation=sep.astype(np.float32),
        )
        manifest["emotions"][name] = {
            "in_file_label": prompts[name]["label"], "n_prompts": len(texts),
            "best_layer": best, "best_separation_sd": round(float(sep[best]), 3),
            "file": f"{name}.npz",
        }
        print(f"{name:16s}: saved; peak separation {sep[best]:.2f} sd at layer {best}", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
