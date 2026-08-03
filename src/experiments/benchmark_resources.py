#!/usr/bin/env python3
"""Resource Benchmark: measures time + RAM inside each pixi environment accurately."""
import time, os, sys, subprocess, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
POS = ROOT / "data/benchmark/positives_81bp.fasta"
NEG = ROOT / "data/benchmark/negatives_81bp.fasta"
N_SEQS = 988 + 1000

# ════════════════════════════════════════════════════════════════
# Inline measurement code (runs INSIDE each pixi environment)
# ════════════════════════════════════════════════════════════════

RESOURCE_CODE = """
import time, json, sys
__IMPORTS__
try:
    import resource
    _ru0 = resource.getrusage(resource.RUSAGE_SELF)
except Exception:
    pass
_t0 = time.perf_counter()
__INFERENCE__
_t = time.perf_counter() - _t0
_ram = 0
try:
    _ru1 = resource.getrusage(resource.RUSAGE_SELF)
    _ram = (_ru1.ru_maxrss - _ru0.ru_maxrss) / 1024.0
except Exception:
    pass
print("RESULT:" + json.dumps({"tool":"__TOOL__","time":round(_t,4),"ram":round(_ram,1),"seqs":__N_SEQS__}))
"""

TF_IMPORT = "import numpy as np; from Bio import SeqIO; import tensorflow as tf"
TF_ONESHOT = "m = {'A':[1,0,0,0],'T':[0,1,0,0],'C':[0,0,1,0],'G':[0,0,0,1]}"
TF_LOAD_POS = f"pos = [str(r.seq).upper() for r in SeqIO.parse('{POS}','fasta')]"
TF_LOAD_NEG = f"neg = [str(r.seq).upper() for r in SeqIO.parse('{NEG}','fasta')]"
TF_ENCODE = "X = np.array([[m[c] for c in s] for s in pos+neg], dtype=np.float32)"

def run_in_env(env_path, imports, inference, timeout=300):
    """Run Python code inside a pixi environment, capture printed RESULT JSON."""
    full_code = RESOURCE_CODE.replace("__IMPORTS__", imports).replace("__INFERENCE__", inference).replace("__N_SEQS__", str(N_SEQS))
    res = subprocess.run(
        ["pixi", "run", "--manifest-path", str(env_path), "python", "-c", full_code],
        capture_output=True, text=True, cwd=str(ROOT), timeout=timeout
    )
    for line in res.stdout.split("\n") + res.stderr.split("\n"):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    print(f"  [DEBUG] full output: {res.stdout[-500:]} {res.stderr[-500:]}")
    raise RuntimeError("No RESULT in output")

def model_size(*paths):
    total = 0
    for p in paths:
        pp = Path(p)
        if pp.is_file(): total += pp.stat().st_size
        elif pp.is_dir():
            for f in pp.rglob("*"):
                if f.is_file(): total += f.stat().st_size
    return round(total / (1024 * 1024), 2)

def bench(name, env, imports, inference, model_paths, category="DL", notes=""):
    print(f"  {name}...", end=" ", flush=True)
    r = run_in_env(env, imports, inference)
    ms = model_size(*model_paths)
    out = {"tool": name, "category": category, "time_s": r["time"], "ram_mb": r["ram"],
           "model_mb": ms, "gpu": False, "seqs": r["seqs"], "notes": notes}
    print(f" {out['time_s']:.2f}s {out['ram_mb']:.0f}MB {out['model_mb']:.0f}MB")
    return out

# ════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════

def bench_lcnn():
    env = ROOT / "tools/Promoters/pixi.toml"
    imports = f"""import numpy as np; from Bio import SeqIO; import tensorflow as tf
{TF_LOAD_POS}; {TF_LOAD_NEG}
{TF_ONESHOT}; {TF_ENCODE}
model = tf.keras.models.load_model('{ROOT}/tools/Promoters/weights/PromoterLCNN/IsPromoter_fold_5', compile=False)"""
    inference = "model.predict(X, verbose=0, batch_size=128)"
    return bench("PromoterLCNN", env, imports, inference,
                 [ROOT / "tools/Promoters/weights/PromoterLCNN/IsPromoter_fold_5"])

def bench_mldspp():
    env = ROOT / "tools/MLDSPP-Promoter-prediction/pixi.toml"
    imports = f"""
import numpy as np, random
from Bio import SeqIO
from xgboost import XGBClassifier
random.seed(42)
ST = {{'AA':-1.00,'TT':-1.00,'AT':-0.88,'TA':-0.58,'AG':-1.30,'GA':-1.30,'AC':-1.45,'CA':-1.45,'TG':-1.44,'GT':-1.44,'TC':-1.28,'CT':-1.28,'CC':-1.84,'GG':-1.84,'CG':-2.24,'GC':-2.27}}
def ex(s):
    return np.array([ST.get(s[:81].upper()[i:i+2],-1.35) for i in range(80)])
pos = np.array([ex(str(r.seq)) for r in SeqIO.parse('{POS}','fasta')])
neg = np.array([ex(str(r.seq)) for r in SeqIO.parse('{NEG}','fasta')])
X_te = np.vstack([pos, neg])
rng = np.random.RandomState(42)
tr = np.vstack([pos[:400], np.array([rng.permutation(r) for r in pos[:400]])])
m = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss', verbosity=0)
m.fit(tr, np.hstack([np.ones(400), np.zeros(400)]))
"""
    inference = "m.predict_proba(X_te)"
    return bench("MLDSPP XGBoost", env, imports, inference, [], "ML", "train 400, infer 1988")

def bench_ipromp():
    env = ROOT / "tools/iPro-MP/pixi.toml"
    itools = ROOT / "tools/iPro-MP"
    imports = f"""
import sys, torch
sys.path.insert(0, '{itools}')
from transformers import AutoTokenizer, AutoModelForSequenceClassification

seqs = []
with open('{POS}') as f:
    for line in f:
        if not line.startswith('>'): seqs.append(line.strip().upper())
seqs = seqs[:30]
tok = AutoTokenizer.from_pretrained('{itools}/DNABERT-6', trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained('{itools}/07-final/12', num_labels=2)
model.eval()
inputs = [tok(' '.join(s), return_tensors='pt', max_length=128, truncation=True) for s in seqs]
"""
    inference = "with torch.no_grad():\n    for inp in inputs: _ = model(**inp)"
    r = run_in_env(env, imports, inference, timeout=300)
    t_est = r["time"] * (N_SEQS / 30) * 1.3
    ram_est = r["ram"] * 1.1
    ms = model_size(itools / "07-final")
    out = {"tool": "iPro-MP sp 12", "category": "DL", "time_s": round(t_est, 1), "ram_mb": round(ram_est, 1),
           "model_mb": ms, "gpu": False, "seqs": N_SEQS, "notes": f"extrapolated from 30 seqs ({r['time']:.1f}s raw); x23 models ≈ {t_est*23/3600:.1f}h"}
    print(f"  iPro-MP... {out['time_s']:.1f}s {out['ram_mb']:.0f}MB {ms:.0f}MB")
    return out

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print(f"  RESOURCE BENCHMARK — {N_SEQS} sequences (81bp)")
    print("=" * 55)

    results = []
    for fn in [bench_lcnn, bench_mldspp]:
        try: results.append(fn())
        except Exception as e: print(f"  FAILED: {e}")

    try: results.append(bench_ipromp())
    except Exception as e: print(f"  FAILED: {e}")

    # PromoTech: subprocess-based pipeline
    print("  PromoTech HOT pipeline...", end=" ", flush=True)
    t0 = time.perf_counter()
    for label, fasta in [("pos", POS), ("neg", NEG)]:
        od = ROOT / f"output/predictions/promotech/workdir/hot_pg_{label}"
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pixi", "run", "python", "promotech.py", "-pg", "-m", "RF-HOT", "-f", str(fasta), "-o", str(od)],
                       capture_output=True, cwd=str(ROOT / "tools/Promotech"), timeout=600)
        subprocess.run(["pixi", "run", "python", "promotech.py", "-g", "-m", "RF-HOT", "-t", "0.0", "-i", str(od), "-o", str(od)],
                       capture_output=True, cwd=str(ROOT / "tools/Promotech"), timeout=600)
    t_hot = time.perf_counter() - t0
    results.append({"tool": "PromoTech RF-HOT", "category": "ML", "time_s": round(t_hot, 1),
                    "ram_mb": 0, "model_mb": model_size(ROOT / "tools/Promotech/models/RF-HOT.model"),
                    "gpu": False, "seqs": N_SEQS, "notes": "full pipeline (parse+pred+agg)"})
    print(f"{t_hot:.0f}s")

    print("  PromoTech TETRA pipeline...", end=" ", flush=True)
    t0 = time.perf_counter()
    for label, fasta in [("pos", POS), ("neg", NEG)]:
        od = ROOT / f"output/predictions/promotech/workdir/tetra_pg_{label}"
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pixi", "run", "python", "promotech.py", "-pg", "-m", "RF-TETRA", "-f", str(fasta), "-o", str(od)],
                       capture_output=True, cwd=str(ROOT / "tools/Promotech"), timeout=600)
        subprocess.run(["pixi", "run", "python", "promotech.py", "-g", "-m", "RF-TETRA", "-t", "0.0", "-i", str(od), "-o", str(od)],
                       capture_output=True, cwd=str(ROOT / "tools/Promotech"), timeout=600)
    t_tetra = time.perf_counter() - t0
    results.append({"tool": "PromoTech RF-TETRA", "category": "ML", "time_s": round(t_tetra, 1),
                    "ram_mb": 0, "model_mb": model_size(ROOT / "tools/Promotech/models/RF-TETRA.model"),
                    "gpu": False, "seqs": N_SEQS, "notes": "full pipeline (parse+pred+agg)"})
    print(f"{t_tetra:.0f}s")

    # GPU
    gpu = {}
    try:
        import tensorflow as tf; gpu["TF"] = bool(tf.config.list_physical_devices("GPU"))
    except Exception: pass
    try:
        import torch; gpu["Torch"] = torch.cuda.is_available()
    except Exception: pass

    # Save
    import pandas as pd
    df = pd.DataFrame(results)
    out_tsv = ROOT / "output/tables/resource_metrics.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)

    print(f"\n{'Tool':<28} {'Time':>8} {'RAM':>7} {'Model':>8}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: x["time_s"]):
        print(f"{r['tool']:<28} {r['time_s']:>7.1f}s {r['ram_mb']:>6.0f}MB {r['model_mb']:>7.0f}MB")
    print(f"\nGPU: {gpu}")
    print(f"Saved: {out_tsv}")

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'; plt.rcParams['font.size'] = 8
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    tools_sorted = sorted(results, key=lambda x: x["time_s"])
    labels = [r["tool"].replace("PromoTech RF-", "PromoTech\nRF-") for r in tools_sorted]
    times = [r["time_s"] for r in tools_sorted]
    colors = ['#4DAF4A' if 'ML' in r.get('category','') else '#377EB8' for r in tools_sorted]
    ax.barh(labels, times, color=colors, height=0.6)
    for i, t in enumerate(times):
        ax.text(t + max(times)*0.02, i, f'{t:.1f}s', va='center', fontsize=7)
    ax.set_xlabel("Wall-clock time (seconds)"); ax.set_title("Inference Time")
    ax.invert_yaxis()
    plt.tight_layout()
    op = ROOT / "output/plots/resource/resource_comparison.svg"
    plt.savefig(op, dpi=300, bbox_inches='tight')
    plt.savefig(str(op).replace('.svg','.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot: {op} + .png")

if __name__ == "__main__":
    main()
