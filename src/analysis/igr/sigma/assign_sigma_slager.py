#!/usr/bin/env python3
"""Sigma factor assignment — Slager et al. (2018) method."""
import csv, subprocess
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT = ROOT / 'output' / 'tables' / 'igr'
OUT.mkdir(parents=True, exist_ok=True)

FIMO = ROOT / '.pixi/envs/default/bin/fimo'
BEDTOOLS = ROOT / '.pixi/envs/default/bin/bedtools'
GENOME = ROOT / 'data/reference/D39V.fna'
TSS = ROOT / 'data/benchmark/d39v/positives_81bp_metadata.tsv'

# PWM matrices
M35 = {'A':[1,0,0,10,0,15],'C':[0,0,0,2,3,1],'G':[0,0,8,3,0,0],'T':[14,15,7,0,12,0]}
M10 = {'A':[0,12,0,10,12,0],'C':[0,1,0,2,0,0],'G':[0,1,0,2,0,0],'T':[15,1,15,1,3,15]}
CMX = {'A':[0,12,0,0,10,11],'C':[0,0,8,0,1,0],'G':[0,0,0,13,1,0],'T':[12,0,4,0,0,1]}
CME = {'A':[2,10,2,2,2,2,2,0,3,8,6,3,0,0],'C':[0,1,0,0,0,0,0,8,4,2,2,2,0,12],
       'G':[0,1,0,8,0,0,0,1,1,3,3,3,5,1],'T':[10,1,10,3,10,10,10,3,4,0,1,4,7,0]}
EXT = {'A':[0,0,0,2,12,0,10,12,0],'C':[0,0,0,2,0,3,2,1,0],'G':[0,8,0,2,0,0,2,1,0],'T':[9,1,9,3,0,6,1,1,9]}

def wmeme(fh, name, pwm):
    w = len(pwm['A'])
    fh.write(f"MOTIF {name}\nletter-probability matrix: alength= 4 w= {w}\n")
    for i in range(w):
        t = sum(pwm[l][i] for l in 'ACGT')
        fh.write('  ' + '  '.join(f"{pwm[l][i]/t:.4f}" for l in 'ACGT') + '\n')
    fh.write('\n')

def extract():
    b = Path('/tmp/tss_up.bed'); fa = Path('/tmp/tss_up.fasta')
    ls = []
    with open(TSS, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            p = int(r['TSS_Position_0based']); st = r['Strand']
            sid = r['Sequence_ID']; ch = r['Chromosome']
            s, e = (max(0, p - 100), p) if st == '+' else (p, p + 100)
            ls.append(f"{ch}\t{s}\t{e}\t{sid}\t0\t{st}\n")
    b.write_text(''.join(ls))
    subprocess.run([str(BEDTOOLS), 'getfasta', '-fi', str(GENOME), '-bed', str(b),
                    '-fo', str(fa), '-name+'], check=True, capture_output=True)
    return fa

def fimo(meme, fa, thresh):
    o = Path(f"/tmp/fimo_{thresh}"); o.mkdir(exist_ok=True)
    subprocess.run([str(FIMO), '--thresh', str(thresh), '--text', '--norc',
                    str(meme), str(fa), '-o', str(o)], check=True, capture_output=True)
    return o / 'fimo.tsv'

def parse(tsv):
    h = defaultdict(list)
    if not tsv.exists(): return h
    with open(tsv) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            s = r.get('sequence_name', '').strip()
            h[s].append({'m': r.get('motif_id', r.get('motif_alt_id', '')).strip().lower(),
                         'p': float(r.get('p-value', 1)),
                         's': int(r.get('start', 0)), 'e': int(r.get('stop', 0))})
    return h

def classify(sid, rpod, comx, come, ext10, up=100):
    for h in comx.get(sid, []):
        if h['p'] < 0.00001 and (up - h['e']) < 6: return 'ComX'
    for h in come.get(sid, []):
        if h['p'] < 0.001: return 'ComE'
    m35 = [h for h in rpod.get(sid, []) if 'minus35' in h['m']]
    m10 = [h for h in rpod.get(sid, []) if 'minus10' in h['m'] and 'extend' not in h['m']]
    for a in m35:
        for b in m10:
            if a['p'] >= 0.001 or b['p'] >= 0.001: continue
            sp = b['s'] - a['e'] - 1; dt = up - b['e']
            if 15 <= sp <= 19 and 3 <= dt <= 8: return 'RpoD_complete'
    for h in sorted(rpod.get(sid, []), key=lambda h: h['p']):
        if 'minus10' in h['m'] and h['p'] < 0.001 and (up - h['e']) <= 20: return 'RpoD_partial'
    for h in ext10.get(sid, []):
        if h['p'] < 0.001 and (up - h['e']) <= 20: return 'RpoD_partial'
    return 'unassigned'

def build_meme(path, motifs):
    with open(path, 'w') as f:
        f.write("MEME version 5\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write("Background letter frequencies\nA 0.30 C 0.20 G 0.20 T 0.30\n\n")
        for name, pwm in motifs: wmeme(f, name, pwm)

def main():
    print('Sigma assignment...')
    # Build MEME files
    build_meme('/tmp/rpod.meme', [('RpoD_minus35', M35), ('RpoD_minus10', M10),
                                  ('ComX_box', CMX), ('ComE_box', CME),
                                  ('Extended_minus10', EXT)])
    build_meme('/tmp/comx.meme', [('ComX_box', CMX)])
    build_meme('/tmp/come.meme', [('ComE_box', CME)])
    build_meme('/tmp/ext10.meme', [('Extended_minus10', EXT)])

    # Extract
    fa = extract()
    print(f'Upstream: {fa}')

    # FIMO
    rh = parse(fimo('/tmp/rpod.meme', fa, 0.001))
    ch = parse(fimo('/tmp/comx.meme', fa, 0.00001))
    eh_come = parse(fimo('/tmp/come.meme', fa, 0.001))
    eh_ext = parse(fimo('/tmp/ext10.meme', fa, 0.001))
    print(f"Hits — RpoD:{sum(len(v) for v in rh.values())} ComX:{sum(len(v) for v in ch.values())} ComE:{sum(len(v) for v in eh_come.values())} Ext10:{sum(len(v) for v in eh_ext.values())}")

    # Classify
    res = []
    with open(TSS, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            sid = r['Sequence_ID'].strip()
            orig = r.get('Sigma_Factor', '').strip() or 'None'
            pred = classify(sid, rh, ch, eh_come, eh_ext)
            res.append({'tss_id': sid, 'original_sigma': orig, 'predicted_sigma': pred})

    # Validate
    c_siga = sum(1 for r in res if r['original_sigma'] == 'SigA' and 'RpoD' in r['predicted_sigma'])
    c_sigx = sum(1 for r in res if r['original_sigma'] == 'SigX' and r['predicted_sigma'] == 'ComX')
    t_siga = sum(1 for r in res if r['original_sigma'] == 'SigA')
    t_sigx = sum(1 for r in res if r['original_sigma'] == 'SigX')
    print(f"\nSigA recovered: {c_siga}/{t_siga} ({c_siga/t_siga*100:.1f}%)")
    print(f"SigX recovered: {c_sigx}/{t_sigx}")

    cats = Counter(r['predicted_sigma'] for r in res)
    print(f"\nAll TSS ({len(res)}):")
    for c in ['RpoD_complete', 'RpoD_partial', 'ComX', 'ComE', 'unassigned']:
        print(f"  {c:<20} {cats.get(c, 0):>4}")

    nr = [r for r in res if r['original_sigma'] == 'None']
    nc = Counter(r['predicted_sigma'] for r in nr)
    print(f"\nNone TSS ({len(nr)}):")
    for c in ['RpoD_complete', 'RpoD_partial', 'ComX', 'ComE', 'unassigned']:
        print(f"  {c:<20} {nc.get(c, 0):>4} ({nc.get(c, 0)/len(nr)*100:.0f}%)")

    # Export
    with open(OUT / 'd39v_sigma_assigned.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tss_id', 'original_sigma', 'predicted_sigma'],
                           delimiter='\t')
        w.writeheader(); w.writerows(res)
    print(f"\nExported: {OUT}/d39v_sigma_assigned.tsv")

if __name__ == '__main__':
    main()
