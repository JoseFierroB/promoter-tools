# Codon HPC Cluster — Pixi Configuration

## Required Setup (one-time per user)

### 1. `~/.config/pixi/config.toml`
```toml
detached-environments = "/hps/software/users/jlees/<USERNAME>/pixi/envs"

[cache]
root = "/hps/software/users/jlees/shared/pixi-cache"
```

### 2. `~/.bash_profile`
```bash
export PIXI_HOME="/hps/software/users/jlees/<USERNAME>/pixi/global"
export PATH="/hps/software/users/jlees/<USERNAME>/pixi/global/bin:$PATH"
```

## Why this setup
- `/hps/software/` — shared storage with space, not per-node
- `/hps/nobackup/` — scratch space, gets cleaned periodically
- Home directory (`/home/`) — limited quota, NOT for pixi envs
- Detached environments avoid bloating the repo's `.pixi/` directory
- Shared cache avoids duplicate package downloads across users

## Verify on Codon
```bash
# Check config
cat ~/.config/pixi/config.toml
echo $PIXI_HOME

# Install MEME env
cd /hps/software/users/jlees/fierro/promoter-tools
pixi install --manifest-path tools/meme/pixi.toml

# Test
pixi run --manifest-path tools/meme/pixi.toml streme -version

# Run benchmark
pixi run python src/cli.py run meme
```

## No sudo required
Pixi writes to user-owned paths on `/hps/software/`. No admin privileges needed.

## Preventing Disk Waste

With `detached-environments` properly configured, pixi reuses environments
across Slurm jobs. Without it, every job creates its own `.pixi/` in HOME.

**To verify on Codon:**
```bash
cat ~/.config/pixi/config.toml          # must have detached-environments
ls /hps/software/users/jlees/fierro/pixi/envs/  # should list environments
du -sh /hps/software/users/jlees/fierro/pixi/   # check total size
```

**Behavior with correct config:**
| Scenario | Duplicates? |
|----------|-------------|
| 10 Slurm jobs, same tool | No — reuses environment |
| `pixi install` in each job | No — detects already installed |
| Package downloads | No — shared cache |
| HOME quota impact | None — everything on `/hps/` |

**Behavior WITHOUT config:**
| Scenario | Duplicates? |
|----------|-------------|
| 50 jobs × 6 tools | ~150 GB wasted |
| Each job | Creates `.pixi/` in HOME |
| HOME quota | Fills up fast |

## Dependencies Status (all clean)
All pixi.toml files use conda-forge exclusively — zero PyPI dependencies:

| Tool | Status |
|------|--------|
| **MEME** | ✓ conda-forge only |
| **MLDSPP** | ✓ conda-forge only |
| **Promotech** | ✓ conda-forge only |
| **Promoters** | ✓ conda-forge only |
| **iPro-MP** | ✓ conda-forge only (fixed from PyPI) |
