# Quick Start for Next AI Instance

## Read these first
1. `MEMORIA.md` — **living document**: project frame, decisions log, task board, session log, gotchas. **Update it at the end of every work session** (see maintenance rules in its header).
2. `README.md` — project overview, pipeline, benchmark results

## Memory layout
- `MEMORIA.md` (repo root) — single persistent development log & decision record.
- `~/Desktop/promoter-tools-investigacion/` — Advisory Council reports, command guides, historical MEMORIA (2026-07-13). **Kept outside the repo on purpose (not on GitHub).**

## Verify environment
```bash
cd /home/fierro/Desktop/promoter-tools
pixi run python -c "from src.benchmark.tools import PROMOTER_TOOLS; print(len(PROMOTER_TOOLS), 'tools')"
pixi run python src/cli.py run lcnn --runs 1
```

## What works
- 5 tools running on Codon A100 GPU (Slurm parallel); 6 registered in `PROMOTER_TOOLS` (+cnnprom, rejected)
- Modular pipeline: CLI → runners → tools (TOML registry) → process_results
- Energy profiles with statistical significance (Firmicutes vs others)
- N≥3 independent runs with bootstrap CI + DeLong test
- MEME/FIMO motif baseline done (STREME + FIMO, AUC 0.915) — pending review with Víctor

## Next
- Review pending tasks with Víctor (see MEMORIA.md §9 Task Board)
- Second-strain (TIGR4) dataset extraction — data ready in `output/tigr4_data/`
- Dinucleotide composition by position (follow-up of 07-26 stability-artifact finding)

## Key constraints
- Don't commit/push without explicit user request
- Don't modify files that are working on Codon
- Check `MEMORIA.md` section "Gotchas" before touching pixi deps
- Codon paths are in `MEMORIA.md`
