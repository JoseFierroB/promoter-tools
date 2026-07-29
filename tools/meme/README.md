# MEME Suite — tools/meme/

## Setup
Todos los binarios de MEME Suite 5.5.9 están disponibles vía pixi:
```bash
pixi run --manifest-path tools/meme/pixi.toml <comando>
```

## Binarios incluidos (57 total, paquete conda-forge)
- **Motif discovery:** streme, meme, xstreme
- **Motif scanning:** fimo, mcast, ame
- **Motif comparison:** tomtom
- **Visualization:** ceqlogo, meme2images
- **Utilities:** fasta-shuffle-letters, fasta-get-markov, fasta-subsample
- **Format conversion:** meme2meme, transfac2meme, jaspar2meme

## Bases de datos
- `motif_databases/PROKARYOTE/` — CollecTF, PRODORIC, RegTransBase, Fan2020
- `motif_databases/ECOLI/` — DPInteract, SwissRegulon

## Uso en el pipeline
```bash
pixi run python src/cli.py run meme     # benchmark
tomtom streme.txt databases/PROKARYOTE/collectf.meme  # validación
```

## Sin permisos root
Todo funciona dentro del entorno pixi aislado. No requiere sudo ni instalación global.
