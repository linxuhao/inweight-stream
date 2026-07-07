# Persistence Is Not Accumulation

Code and raw results for:

> **Persistence Is Not Accumulation: Erasure, Suppression, and Re-instatement in Online In-Weight Memory for Language Models** (under double-blind review)

Can a frozen LLM accumulate facts **in its weights**, one at a time, single pass? We audit that
streaming regime with per-fact instrumentation and find:

1. **Accumulation is re-instatement, not persistence.** Median time-to-first-miss is 3–8 subsequent
   writes for *every* mechanism (naked SGD, Benna–Fusi cascade, accumulated EWC, replay, compositions),
   while end-of-stream retention spans 12×. What differs is recovery-after-miss: 4% → 94%.
2. **Forgetting is erasure or suppression depending on the write protection.** A 2-AFC recognition
   probe (distractors drawn from the same trained stream) shows naked-SGD forgetting is at chance
   (erased), while EWC keeps ~84% of "forgotten" facts discriminable (suppressed).
3. **Error-gated replay harvests suppressed facts at zero extra budget.** Spending the same 4
   replay steps/turn on self-test failures first: **41–46 of 48 facts (85–96%)** recallable from
   weights at end of stream, recovery 94%. The gain requires the EWC substrate (latent misses);
   miss-gating plain replay is within noise.
4. **Online writes are readout-proximal.** At matched adapter capacity, the last third of the stack
   retains ~90% of full-stack replay capacity; the first third is nearly useless — the opposite of the
   early-layer placement reported for offline batch injection (Back et al., ICML 2026).

Capability firewall in every run: adapter-off GSM8K == base (+0.00), all mechanisms, all seeds.

## Repository layout

```
accum.py              # the streaming instrument (one driver, all mechanisms/arms)
lib.py                # shared machinery (probes, GSM8K firewall eval, templates)
analyze.py            # reproduces the paper tables from results/ (pinned recovery definition)
gsm8k_pilot_ids.json  # frozen 10-item GSM8K firewall subset (indices into openai/gsm8k test)
results/              # raw per-fact timelines (JSON) for every run in the paper
reproduce.sh          # the full run matrix (~50 runs × ~5–8 min on one 24GB GPU)
```

## Reproduce

```bash
pip install torch transformers peft datasets
python analyze.py                    # re-derive all paper tables from shipped raw results
bash reproduce.sh                    # re-run everything (downloads Qwen3.5-2B; fp32, ~10GB VRAM)
```

Single arms:

```bash
python accum.py --mech naked                                  --seed 1234   # the floor
python accum.py --mech ewc                                    --seed 1234   # suppression substrate
python accum.py --mech ewcreplay --replay-policy miss --firewall-n 10 --seed 1234   # the 85–96% winner
python accum.py --mech replay --layers late                   --seed 1234   # depth-band arm
```

Every run writes a JSON of per-fact recall hits and recognition margins at every probe to `results/`
(filenames encode all swept parameters).

## Honest notes (also in the paper's Limitations)

- Single substrate (Qwen3.5-2B, fp32, eager attention). Results are **directions**, not calibrated
  magnitudes: our GPUs (consumer ROCm) are nondeterministic even at fixed seed; every condition has
  3 seeds (1234 / 2025 / 777) and per-seed numbers are always reported.
- The fact stream is synthetic and collision-free by design (unguessable pseudoword values — recall
  cannot be faked); real facts correlate and get revised.
- Recall probes are form-matched to the writes; this inflates absolute recall for all arms equally.
- The self-test that gates replay is the measurement probe (information any agent with a log of its
  own writes has); probe frequency is therefore a mechanism hyperparameter (every 2 writes here).
- `results/pre_recog_2026-07-04/` holds the pre-recognition-meter baseline runs that the shipped
  re-runs superseded (kept for provenance).


MIT license (see LICENSE). Anonymized for review.
