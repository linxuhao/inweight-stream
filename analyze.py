"""accum result analysis: finals + re-instatement (recovery) stats + recognition, per arm.

PINNED recovery definition (the one behind every recovery-% in the paper):
a fact is a recovery CANDIDATE if it was recalled at some probe and later missed one; it RECOVERED if
it is recalled again at any probe after its first miss. recovery-% = recovered/candidates, pooled over
seeds. Computed from the per-probe `hits` vectors in the result JSONs (probe-time units).
Recognition = `n_recognized` at the final probe (2-AFC logprob margin > 0; chance = n_stream/2)."""
import json, glob, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def stats(path):
    d = json.load(open(path))
    curve = d["curve"]
    n = d["n_stream"]
    # build per-fact timeline across probes
    probes = [(c["k"], c["hits"]) for c in curve]
    K = len(probes)
    alive_end = sum(probes[-1][1])
    missed, recovered = 0, 0
    for fid in range(n):
        seq = [h[fid] for k, h in probes if len(h) > fid]
        if not seq or 1 not in seq:
            continue  # never stored -> not a recovery candidate
        first = seq.index(1)
        after = seq[first:]
        if 0 in after:
            missed += 1
            i0 = first + after.index(0)
            if 1 in seq[i0:]:
                recovered += 1
    return d["final_n_recalled"], alive_end, missed, recovered, d.get("firewall") or {}

if __name__ == "__main__":
    for mech in ["naked", "ewc_l300", "replay", "ewcreplay_l300"]:
        for arm in ["", "_Learly", "_Lmid", "_Llate", "_Pmiss", "_Llate_Pmiss"]:
            files = sorted(glob.glob(os.path.join(R, f"accum_{mech}{arm}_n48_pe2_s*.json")))
            if not files:
                continue
            rows = [stats(f) for f in files]
            fin = [r[0] for r in rows]
            tot_r = sum(r[3] for r in rows); tot_m = sum(r[2] for r in rows)
            pct = 100 * tot_r / tot_m if tot_m else float("nan")
            fw = [f"{r[4].get('gsm8k_base')}->{r[4].get('gsm8k_off')}" for r in rows if r[4]]
            nrec = [json.load(open(f))["curve"][-1].get("n_recognized") for f in files]
            print(f"{mech:16s} {arm or '_all':12s} n={len(files)}  final={fin} mean={sum(fin)/len(fin):5.1f}  "
                  f"recover={tot_r}/{tot_m} ({pct:.0f}%)  recog={nrec}"
                  + (f"  firewall={fw}" if fw else ""))
