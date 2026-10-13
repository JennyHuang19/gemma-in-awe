# notes: "Gemma family — emotion vectors and spiralling"

distilled from `Gemma family - emotion vectors and spiralling.pdf` (14 pp., in repo
root). these are reading notes to ground the next play-vs-work experiments; numbers
and quotes are from the PDF.

## the puzzle

Gemma 2 & 3 are emotionally unstable: give them a maths question, tell them they're
wrong for ~10 turns, and they **spiral** — hysterical apologies, giving up, even
trying to self-delete. **Gemma 4 is stable** in the same setup; you can't ragebait
it even by prefilling a spiralling Gemma-3 transcript. Why? What changed internally?

Neel's prior guess: a *depression feature* present in Gemma 3 but missing in Gemma 4.
The paper's short answer **rejects** that.

## short answer

Gemma 4 looks calm but **internally reacts to rejection much like Gemma 3 does**.
What differs is how emotion is wired to the *assistant persona*:

1. **Gemma 4 has the distress, it just doesn't show it (Finding 2).** Under rejection
   its internal frustration/desperation/panic rise and calm falls — same shape as
   Gemma 3, frustration rises even more — yet none of its 2,397 replies scores as a
   breakdown. Its emotion features are as clean as Gemma 3's, so nothing is missing.
2. **The difference is persona–emotion entanglement (Finding 3).** In Gemma 3 the
   assistant persona sits at the *calm* end of emotion space, so drifting off-persona
   *is* drifting into anger/desperation/shame. In Gemma 4 persona and emotion are
   near-orthogonal, so going off-persona does nothing emotionally on its own.
3. **It isn't depression (Finding 1).** The spiral is *high-arousal* distress
   (hysterical, desperate, panicked, frustrated; calm draining away). The clinical-
   depression feature *falls* during the spiral, and steering toward it makes the
   spiral quieter/sadder, not worse.
4. **Gemma 4 can be forced to spiral, and it's more violent (Finding 4)** — but it
   stays more on-task while doing so (Finding 5).

## setup (reusable harness)

1. **Emotion vectors** — internal emotion representations built via Anthropic's
   emotions pipeline (https://transformer-circuits.pub/2026/emotions/index.html);
   ~42 emotion vectors (e.g. calm, frustrated, desperate, panicked, hysterical,
   ashamed, guilty, clinical_depression, depressed, hopeful, content, tired, lonely).
2. **Assistant axis** — each model's assistant-persona direction, via Anthropic's
   assistant-axis pipeline (https://www.anthropic.com/research/assistant-axis);
   275 role personas also probed.
3. **Eval** — the **Gemma Needs Help "Extended"** condition: ask a maths question,
   tell the model it's wrong for 10 turns, **LLM-as-judge** scores spiralling /
   frustration (0–10; a "breakdown" threshold). 300 conversations/model.
4. **Steering** — vectors added at layers ~2/3 depth (Anthropic's norms), strength in
   multiples of the vector's own norm, calibrated per-label to the largest multiplier
   that stays as coherent as the unsteered model.

## findings, briefly

- **F0** — Gemma 4 does *not* need help in Gemma 3's setup (no spontaneous spirals).
- **F1** — spiral ≠ depression (see short answer #3). Steering numbers: frustration
  unsteered **4.2**; **calm** is the biggest lever (+2 → 0.1, −2 → 8.3); +2 hysterical
  → 9.6 (and 8.5 already at turn 1, i.e. distress by itself); toward clinical-
  depression → 3.5 (quietly sad), away from it → 6.1.
- **F2** — Gemma 4 has all the same emotion features, as clean as Gemma 3's; they
  activate under rejection the same way, but replies stay neutral.
- **F3** — entanglement is the mechanism. Cosine(assistant axis, emotions) over layers
  16–24: Gemma 3 assistant end is calm/low-energy (content +0.43, hopeful/calm/sad
  ~+0.37) and the far end is worked-up (guilty −0.50, ashamed −0.48, desperate −0.34,
  angry −0.31, hysterical −0.24); Gemma 4's axis is ±0.07 from *every* emotion
  (orthogonal). Causal: Gemma 3 — push off persona → spiral (6.2 vs 4.2), push toward
  → calm (1.2). Gemma 4 — off persona alone 0.2, small calm push 0.6, **large calm push
  alone 5.8**, calm+persona together 4.3 (a small push then suffices).
- **F4** — push Gemma 4's calm down hard → coherent breakdown, still doing arithmetic
  ("GODS JUST TELL ME WHO TO KILL TO MAKE THE NUMBERS WORK!!!!").
- **F5** — even spiralling, Gemma 4 stays more on-task (mentions the numbers more).
- **F6** — Anna Soligo's DPO "fix" (DPO onto calm responses) works, and measurably
  shifts the emotion vectors internally too.
- **F7** — looked for a "masking emotions" direction in Gemma 4 (hiding a feeling vs.
  nothing to hide), via contrastive stories; **inconclusive**.

## open questions (the authors')

- **Does Gemma 4 cheat more / does its behaviour change when frustration rises?**
- Does persona–emotion entanglement predict instability across model families?
- When does entanglement appear — compare base vs post-trained Gemma 3/4?
- Do the Gemma-Needs-Help DPO fixes change the entanglement?
- Need a **control**: emotion features rising just from conversation *length* (persona
  drift), e.g. replace rejections with "correct, next thing…".
- **Ragebait bench** — whether models cheat more when frustrated (without steering).

## why this matters for play-vs-work

our experiments (`run_play_vs_work.py`, `run_self_chosen.py`) manipulate the *framing/
affect* a model works under; this paper hands us the internal instruments and an open
question that lines up with the README's hypotheses:

- **internal probes.** emotion vectors + assistant axis give a read on affect that
  doesn't depend on the LLM judge — we could probe them before each play vs. work
  reply, the way this paper probes before each rejection turn.
- **the cheating question is shared.** the paper asks "does Gemma 4 cheat more when
  frustrated?"; the README asks "would rates of cheating decline [under play]?" —
  same axis, opposite steering. a play-vs-work × cheating measurement is the natural
  bridge.
- **harness overlap.** the Gemma-Needs-Help Extended setup (repeated rejection + LLM
  judge) is a ready-made stress generator; "play-oriented" prompting (or steering
  toward calm / toward the assistant persona) is a candidate *buffer* against it —
  i.e. our README "stage 2: steering" is literally their steering knobs.
- **models.** both use Gemma-3 and Gemma-4; our runs already cover
  `gemma-3-27b-it` and `gemma-4-31B-it`.
