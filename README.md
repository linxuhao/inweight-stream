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
4. **The depth gradient is an expression gradient, not a storage gradient.** Recall concentrates in
   the last third of the stack (opposite of the early-layer placement reported for batch injection,
   Back et al. ICML 2026) — but recognition shows the middle third *stores* near full-stack levels
   while expressing a quarter of it.
5. **Error-gating needs fresh self-tests** (dose–response): the gain at one self-test per 2 writes
   (44.2/48) decays to uniform-replay level by one per 6, and further by one per 12.
6. **Structure transfers, magnitudes don't** (SmolLM2-1.7B-Instruct, hyperparameters unchanged):
   ladder/recovery/latency replicate; EWC-alone recalls 0/48 while recognizing above chance.
7. **A real-entity CounterFact stream** replicates the ladder at higher levels and exposes a
   three-tier readout hierarchy: recognition (~100%) ≫ write-form recall (~90%) ≫ paraphrase (12–31%).

Capability firewall in every run: adapter-off GSM8K == base (+0.00), both substrates, all mechanisms and seeds.

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

### Driver flags

| flag | values (default first) | meaning |
|---|---|---|
| `--mech` | naked · bf · ewc · local · replay · ewcreplay | write-protection mechanism |
| `--replay-policy` | uniform · miss | replay selection (miss = error-gated by the last self-test) |
| `--layers` | all · early · mid · late · `a-b` | restrict the adapter to a layer band |
| `--facts` | synthetic · counterfact | fact source (counterfact downloads via HF `datasets`; adds a paraphrase probe) |
| `--model` | Qwen/Qwen3.5-2B · any HF causal LM | substrate (non-default is encoded in the filename) |
| `--probe-every` | 2 | probe/self-test cadence in writes (the §4.3 sweep variable) |
| `--firewall-n` | 0 | >0: GSM8K items for the adapter-off firewall check |
| `--n-stream --ws --rank --lr --ewc-lambda --replay-m --seed` | 48 · 8 · 64 · 3e-5 · 300 · 4 · 1234 | stream/opt hyperparameters |

### Result JSON schema

Filenames encode every swept parameter, e.g. `accum_ewcreplay_l300_Pmiss_MSmolLM217BInst_n48_pe2_s777.json`
= mech+λ, miss policy, SmolLM2 substrate, 48 facts, probe-every-2, seed 777. Fields:

- top level: all hyperparameters, `final_n_recalled`, `final_n_recognized`, `firewall`
  (`{gsm8k_base, gsm8k_off}` — equality = intact base);
- `curve`: one entry per probe point — `k` (facts written so far), `hits` (per-fact 0/1 greedy-recall
  vector over facts 1..k, index = fid), `margins` (per-fact 2-AFC logprob margin, >0 = recognized;
  chance = half the stream), `n_recognized`, `cumrecall`, and for CounterFact runs `para_hits`
  (paraphrase-probe recall).

`analyze.py` column legend: `final` = recalled at last probe (per seed) · `first-miss(med)` = median
writes between a fact's write and its first miss · `alive` = pooled last-probe recalls ·
`recover` = recovered/candidates (recovery definition pinned in the docstring) · `recog` = final
2-AFC counts · `para` = paraphrase recall (CF runs).

**Hardware note**: the code is standard PyTorch + transformers + peft — it runs on NVIDIA/CUDA
as-is (`--dev cuda:0` is the same device string under ROCm, which is simply what *our* GPUs were).
The ROCm remarks in this repo describe our measurement hardware, not a requirement; expect exact
numbers to differ on any hardware (they differ across our own seeds too).

## Honest notes (also in the paper's Limitations)

- Two substrates (Qwen3.5-2B primary; SmolLM2-1.7B-Instruct with hyperparameters carried over
  unchanged), fp32, eager attention. Results are **directions**, not calibrated magnitudes: our GPUs
  (consumer ROCm) are nondeterministic even at fixed seed; 3–5 seeds per condition (1234/2025/777,
  +7/42 for the n=5 compositions) and per-seed numbers are always reported.
- The main stream is synthetic and collision-free (unguessable pseudoword values — recall cannot be
  faked); the CounterFact stream adds real entities (counterfactual targets stay unguessable) but not
  revision/conflict.
- Recall probes are form-matched to the writes; the CounterFact paraphrase probe measures the
  resulting inflation (write-form ~90% vs paraphrase 12–31%).
- The self-test that gates replay is the measurement probe (information any agent with a log of its
  own writes has); its cadence is a mechanism hyperparameter, swept at 2/6/12 writes.
- `results/pre_recog_2026-07-04/` holds the pre-recognition-meter baseline runs that the shipped
  re-runs superseded (kept for provenance).


MIT license (see LICENSE). Anonymized for review.
