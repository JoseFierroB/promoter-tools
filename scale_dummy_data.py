import random, os
random.seed(42)

def read_fasta(path):
    seqs, cur = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            if cur: seqs.append(''.join(cur)); cur = []
        else:
            cur.append(line.upper())
    if cur: seqs.append(''.join(cur))
    return seqs

pos = read_fasta('/home/fierro/promoter-tools/data/benchmark/d39v/positives_81bp.fasta')
neg = read_fasta('/home/fierro/promoter-tools/data/benchmark/d39v/negatives_81bp.fasta')
out = '/home/fierro/promoter-tools/scale_db'
os.makedirs(out, exist_ok=True)

# 988 × m por clase, 1:1 pos/neg
for m in [1, 2, 5, 10, 20, 30, 50, 100, 200]:   # ← 100K/clase; usar [.., 50] para 100K total
    k = 988 * m
    for lbl, pool in [('pos', pos), ('neg', neg)]:
        with open(f'{out}/{lbl}_{k}.fasta', 'w') as f:
            for i in range(k):
                f.write(f'>s{i}\n{random.choice(pool)}\n')
    print(f'{k}+{k} OK')
