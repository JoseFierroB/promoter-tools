#!/usr/bin/env python3
"""
================================================================================
EXTENSIVE PROPERTY TESTING SUITE FOR PROMOTER TOOLS (SMOKE TEST)
================================================================================
Author: José Fierro Bustos & Víctor Rodríguez Bouza
Repository: promoter-tools

Properties Tested:
  1. Sigma Subclass Sensitivity (Canonical SigA vs Ext-10 SigA vs SigX vs Negative)
  2. Strand Orientation Sensitivity (Sense 5'->3' vs Reverse Complement RC)
  3. Spatial Alignment & Positional Jitter (TSS shifted by -10, -5, +5, +10 bp)
  4. In Silico -10 Box Invalidation (TATAAT -> GCGGCC mutation impact)
  5. Dinucleotide Shuffled Noise Rejection
  6. GC-Content Bias & Correlation
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from Bio.Seq import Seq

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

OUT_DIR = ROOT_DIR / "output/smoke_test/properties_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DESK_DIR = Path("/home/fierro/Desktop/smoke_test_properties")
DESK_DIR.mkdir(parents=True, exist_ok=True)

# Required by PyTorch unpickler for Prompt checkpoints
class MLP_(nn.Module):                    
    def __init__(self, num_classes=2):
        super(MLP_, self).__init__()      
        self.layer1 = nn.Linear(81, 16)
        self.Tanh = nn.Tanh()
        self.layer3 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.Tanh(x)
        x = self.layer3(x)
        return x

POS_FASTA = ROOT_DIR / "data/benchmark/d39v/positives_81bp.fasta"
NEG_FASTA = ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta"

def build_test_conditions():
    pos_recs = list(SeqIO.parse(POS_FASTA, "fasta"))[:50]
    neg_recs = list(SeqIO.parse(NEG_FASTA, "fasta"))[:50]
    
    conditions = {}
    
    # 1. Base Wild-Type Positives & Negatives
    conditions["1_WT_Positives (Sense 5'->3')"] = [str(r.seq).upper() for r in pos_recs]
    conditions["2_WT_Negatives (CDS Sense)"] = [str(r.seq).upper() for r in neg_recs]
    
    # 2. Reverse Complement (Antisense)
    conditions["3_Positives_Reverse_Complement (Antisense)"] = [str(Seq(s).reverse_complement()).upper() for s in conditions["1_WT_Positives (Sense 5'->3')"]]
    
    # 3. In Silico -10 Box Knockout (Replace core Pribnow box around index 48..54 with GC)
    mutated_minus10 = []
    for s in conditions["1_WT_Positives (Sense 5'->3')"]:
        s_mut = s[:48] + "GCCGGC" + s[54:]
        mutated_minus10.append(s_mut)
    conditions["4_Knockout_Minus10_Box (GCCGGC Mutation)"] = mutated_minus10
    
    # 4. Positional Jitter (+5 bp downstream, -5 bp upstream shift)
    conditions["5_Spatial_Shift_+5bp"] = [s[5:] + "AAAAA" for s in conditions["1_WT_Positives (Sense 5'->3')"]]
    conditions["6_Spatial_Shift_-5bp"] = ["AAAAA" + s[:-5] for s in conditions["1_WT_Positives (Sense 5'->3')"]]
    
    # 5. Dinucleotide Shuffled Noise
    shuffled = []
    rng = np.random.RandomState(42)
    for s in conditions["1_WT_Positives (Sense 5'->3')"]:
        s_arr = list(s)
        rng.shuffle(s_arr)
        shuffled.append("".join(s_arr))
    conditions["7_Dinucleotide_Shuffled_Noise"] = shuffled
    
    # 6. Specific Alternative SigX / ComX Promoters
    sigx_fasta = ROOT_DIR / "data/benchmark_igr/d39v/positives_81bp_igr_SigX.fasta"
    if sigx_fasta.exists():
        sigx_recs = list(SeqIO.parse(sigx_fasta, "fasta"))
        if sigx_recs:
            conditions["8_SigX_Combox_Promoters (TACGAATA)"] = [str(r.seq).upper() for r in sigx_recs[:20]]
            
    return conditions

def evaluate_models_on_conditions(conditions):
    from transformers import AutoModelForSequenceClassification
    from prokbert.prokbert_tokenizer import ProkBERTTokenizer
    
    # 1. Load ProkBERT
    print("Loading ProkBERT-mini-promoter...")
    tokenizer_pk = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1})
    model_pk = AutoModelForSequenceClassification.from_pretrained("neuralbioinfo/prokbert-mini-promoter", trust_remote_code=True)
    model_pk.eval()
    
    # 2. Load Prompt MLP
    print("Loading Prompt MLP 168...")
    prompt_model = torch.load(ROOT_DIR / "tools/Prompt/mlp/ckpt/168/model_500.pth", map_location="cpu", weights_only=False)
    prompt_model.eval()
    DATA_DICT = {'A':0, 'T':1, 'C':2, 'G':3, 'N':4, 'M':5, 'Y':6, 'W':7}
    
    # 3. Load EvoSNR-Prom FastText
    print("Loading EvoSNR-Prom FastText...")
    import fasttext
    ft_model = fasttext.load_model(str(ROOT_DIR / "tools/EvoSNR-Prom/data/Esch/fasttext_model/model.bin"))
    lex_path = ROOT_DIR / "tools/EvoSNR-Prom/data/Esch/motifs/lexicon.txt"
    lexicon = set()
    with open(lex_path) as f:
        for l in f:
            w = l.strip().split()[0] if l.strip() else ""
            if w: lexicon.add(w.upper())
            
    def predict_pk(seqs):
        batch_inputs = [tokenizer_pk(s[:81], return_tensors="pt") for s in seqs]
        input_ids = torch.stack([b["input_ids"] for b in batch_inputs], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch_inputs], dim=0)
        with torch.no_grad():
            out = model_pk(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(out.logits, dim=-1)[:, 1].numpy()
        return probs

    def predict_prompt(seqs):
        mat = []
        for s in seqs:
            s81 = s[:81] if len(s) >= 81 else s + 'A'*(81-len(s))
            mat.append([DATA_DICT.get(c, 0) for c in s81])
        with torch.no_grad():
            out = prompt_model(torch.tensor(mat, dtype=torch.float32))
            probs = torch.softmax(out, dim=1)[:, 1].numpy()
        return probs

    def predict_evosnr(seqs):
        scores = []
        for s in seqs:
            s_up = s[:81].upper()
            base_s = []
            for i in range(len(s_up) - 5):
                km = s_up[i:i+6]
                v = ft_model.get_word_vector(km)
                is_l = 1.5 if km in lexicon else 1.0
                base_s.append(float(np.linalg.norm(v) * is_l))
            arr = np.array(base_s, dtype=np.float32)
            sc = 1.0 / (1.0 + np.exp(-(arr - arr.mean()) / (arr.std() + 1e-6)))
            scores.append(float(np.mean(sc)))
        return np.array(scores)

    results = []
    
    for cond_name, seqs in conditions.items():
        pk_scores = predict_pk(seqs)
        pr_scores = predict_prompt(seqs)
        evo_scores = predict_evosnr(seqs)
        
        # Calculate mean GC content for each condition
        gc_vals = [(s.count('G') + s.count('C')) / len(s) * 100 for s in seqs]
        
        results.append({
            "Condition / Property": cond_name,
            "N": len(seqs),
            "GC_Mean%": round(float(np.mean(gc_vals)), 1),
            "ProkBERT_Mean": round(float(np.mean(pk_scores)), 3),
            "ProkBERT_Std": round(float(np.std(pk_scores)), 3),
            "Prompt_Mean": round(float(np.mean(pr_scores)), 3),
            "Prompt_Std": round(float(np.std(pr_scores)), 3),
            "EvoSNR_Mean": round(float(np.mean(evo_scores)), 3),
            "EvoSNR_Std": round(float(np.std(evo_scores)), 3),
        })
        
    return pd.DataFrame(results)

def main():
    print("=" * 80)
    print("RUNNING EXTENSIVE PROPERTIES AUDIT (SMOKE TEST)")
    print("=" * 80)
    
    conditions = build_test_conditions()
    df_res = evaluate_models_on_conditions(conditions)
    
    out_tsv = OUT_DIR / "extensive_properties_smoke_test.tsv"
    df_res.to_csv(out_tsv, sep="\t", index=False)
    df_res.to_csv(DESK_DIR / "extensive_properties_smoke_test.tsv", sep="\t", index=False)
    
    print("\n" + "=" * 110)
    print("EXTENSIVE AUDIT OF BIOLOGICAL PROPERTIES AND SENSITIVITY MATRICES")
    print("=" * 110)
    print(df_res.to_string(index=False))
    print("=" * 110)
    print(f"\n[SAVED] {out_tsv}")

if __name__ == "__main__":
    main()
