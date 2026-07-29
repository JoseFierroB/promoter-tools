"""Single source of truth for inline tool runner code.
Used by both LocalRunner and Orchestrator to avoid code duplication.
"""

import textwrap
from typing import Optional


def get_runner_code(tool_name: str, pos_fasta: str, neg_fasta: str,
                    combined_fasta: Optional[str] = None,
                    dnabert_dir: str = "tools/iPro-MP/DNABERT-6",
                    ipromp_model_dir: str = "tools/iPro-MP/07-final") -> str:
    """Return inline Python code string for the given tool.

    Args:
        tool_name: short name (lcnn, mldspp, meme, cnnprom, ipromp_sp12)
        pos_fasta: absolute path to positive FASTA
        neg_fasta: absolute path to negative FASTA
        combined_fasta: combined pos+neg FASTA (for tools that need one file)
    """
    if combined_fasta is None:
        combined_fasta = pos_fasta  # fallback

    runners = {
        "lcnn": f"""
import time, numpy as np
from Bio import SeqIO
import tensorflow as tf

seqs = [str(r.seq).upper() for r in SeqIO.parse("{combined_fasta}","fasta")]
m = {{"A":[1,0,0,0],"T":[0,1,0,0],"C":[0,0,1,0],"G":[0,0,0,1]}}
X = np.array([[m[c] for c in s] for s in seqs], dtype=np.float32)
model = tf.keras.models.load_model("tools/Promoters/weights/PromoterLCNN/IsPromoter_fold_5", compile=False)
t0 = time.perf_counter(); model.predict(X, verbose=0, batch_size=128)
print(f"LCNN: {{len(seqs)}} seqs in {{time.perf_counter()-t0:.3f}}s")
""",

        "mldspp": f"""
import time, numpy as np
from Bio import SeqIO
from xgboost import XGBClassifier
from pathlib import Path

ST = {{"AA":-1.00,"TT":-1.00,"AT":-0.88,"TA":-0.58,"AG":-1.30,"GA":-1.30,"AC":-1.45,"CA":-1.45,"TG":-1.44,"GT":-1.44,"TC":-1.28,"CT":-1.28,"CC":-1.84,"GG":-1.84,"CG":-2.24,"GC":-2.27}}
def ex(s):
    if len(s)>=100:ss=s[20:100]
    else:ss=s[:80]
    return np.array([ST.get(ss[i:i+2],-1.35) for i in range(79)])
pos=[];rng=np.random.RandomState(42)
for f in sorted(Path("tools/MLDSPP-Promoter-prediction/Sample Dataset/Promoter Sequences").glob("Sequences_80-20_B*.txt")):
    for l in open(f):
        s=l.strip()
        if len(s)>=100:pos.append(ex(s))
tp=np.array(pos)
tn=np.array([rng.permutation(r) for r in tp])
X_tr=np.vstack([tp,tn]);y_tr=np.hstack([np.ones(len(tp)),np.zeros(len(tn))])
sp=np.array([ex(str(r.seq)) for r in SeqIO.parse("{pos_fasta}","fasta")])
sn=np.array([ex(str(r.seq)) for r in SeqIO.parse("{neg_fasta}","fasta")])
m=XGBClassifier(n_estimators=100,max_depth=6,random_state=42,eval_metric="logloss",verbosity=0)
m.fit(X_tr,y_tr)
t0=time.perf_counter();m.predict_proba(np.vstack([sp,sn]))
print(f"MLDSPP: 1988 seqs in {{time.perf_counter()-t0:.4f}}s")
""",

        "meme": f"""
import time, os, shutil, subprocess, tempfile, csv, math, random
from pathlib import Path
from Bio import SeqIO

random.seed(42)

pos_recs = list(SeqIO.parse("{pos_fasta}", "fasta"))
neg_recs = list(SeqIO.parse("{neg_fasta}", "fasta"))
random.shuffle(pos_recs); random.shuffle(neg_recs)
mid_pos = len(pos_recs) // 2; mid_neg = len(neg_recs) // 2

t0 = time.perf_counter()
all_scores = {{}}

# 2-fold: train on fold A, test on B; then flip
for fold in range(2):
    if fold == 0:
        train_pos = pos_recs[:mid_pos]; train_neg = neg_recs[:mid_neg]
        test_pos = pos_recs[mid_pos:]; test_neg = neg_recs[mid_neg:]
    else:
        train_pos = pos_recs[mid_pos:]; train_neg = neg_recs[mid_neg:]
        test_pos = pos_recs[:mid_pos]; test_neg = neg_recs[:mid_neg]

    tmpdir = Path(tempfile.mkdtemp(prefix="meme_cv_"))
    train_pf = tmpdir / "tp.fa"; train_nf = tmpdir / "tn.fa"
    SeqIO.write(train_pos, train_pf, "fasta")
    SeqIO.write(train_neg, train_nf, "fasta")
    test_fa = tmpdir / "test.fa"
    with open(test_fa, "w") as f:
        for r in test_pos: SeqIO.write(r, f, "fasta")
        for r in test_neg: SeqIO.write(r, f, "fasta")

    res = subprocess.run(
        ["streme", "-oc", str(tmpdir/"streme"), "-dna", "-minw", "10", "-maxw", "20",
         "-p", str(train_pf), "-n", str(train_nf)],
        capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        continue  # skip fold if STREME fails
    res = subprocess.run(
        ["fimo", "--text", "--skip-matched-sequence",
         str(tmpdir/"streme"/"streme.txt"), str(test_fa)],
        capture_output=True, text=True, timeout=300)

    for row in csv.DictReader(res.stdout.splitlines(), delimiter="\\t"):
        try: pval = float(row["p-value"])
        except: continue
        nl = 999.0 if pval <= 0 else -math.log10(pval)
        s = row["sequence_name"]
        if s not in all_scores or nl > all_scores[s]:
            all_scores[s] = nl

    shutil.rmtree(tmpdir, ignore_errors=True)

# Zero-fill any missed sequences
for r in pos_recs + neg_recs:
    if r.id not in all_scores:
        all_scores[r.id] = 0.0

n_total = len(pos_recs) + len(neg_recs)
print(f"MEME: {{n_total}} seqs in {{time.perf_counter()-t0:.3f}}s")
""",

        "cnnprom": f"""
import numpy as np
from Bio import SeqIO
import tensorflow as tf

seqs = [str(r.seq).upper() for r in SeqIO.parse("{combined_fasta}","fasta")]
m = {{'A':[1,0,0,0],'T':[0,1,0,0],'C':[0,0,1,0],'G':[0,0,0,1]}}
X = np.array([[m[c] for c in s] for s in seqs], dtype=np.float32)
model = tf.keras.models.load_model('output/predictions/cnnprom_ecoli_model.keras')
model.predict(X, verbose=0, batch_size=128)
print('DONE')
""",

        "ipromp_sp12": f"""
import sys, torch, os
os.chdir('tools/iPro-MP')
sys.path.insert(0, '.')

with open('{combined_fasta}') as f:
    seqs = [line.strip().upper() for line in f if not line.startswith('>')]

from transformers import BertTokenizer
import importlib.util
spec = importlib.util.spec_from_file_location('ip', 'iPro-MP_predict.py')
ip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ip)

tok = BertTokenizer.from_pretrained('{dnabert_dir}')
model = ip.DNABERTPromoterClassifier(dnabert_dir='{dnabert_dir}')
state_dict = torch.load('{ipromp_model_dir}/12_fold_1.pth', map_location='cpu')
state_dict = {{k: v for k, v in state_dict.items() if 'position_ids' not in k}}
model.load_state_dict(state_dict, strict=False)
model.eval()

for s in seqs:
    inp = tok(' '.join(s), return_tensors='pt', max_length=128, truncation=True, padding=True)
    inp = {{k: inp[k] for k in ['input_ids', 'attention_mask'] if k in inp}}
    with torch.no_grad():
        _ = model(**inp)
print('DONE')
""",
    }

    code = runners.get(tool_name, "")
    if not code:
        raise NotImplementedError(f"No runner code for {tool_name}")
    return textwrap.dedent(code).strip()
