# Benchmarking Methodology — Good Practices

Inspired by Beyer et al. (2019) "Reliable Benchmarking: Requirements and Solutions" and HPC community standards (SPEC, MLPerf). We do NOT claim BenchExec compliance — their framework requires cgroups/namespaces which our pixi-based pipeline does not implement.

## Implemented Practices

| Practice | Implementation | Status |
|----------|---------------|--------|
| **CPU time measurement** | `resource.getrusage(RUSAGE_CHILDREN)` in `reliable_runner.py` | Active |
| **Process tree termination** | `kill_process_tree()` via psutil + os.killpg fallback | Available (pending integration) |
| **Dependency isolation** | Per-tool pixi environments (no shared Python deps) | Active |
| **Process groups** | `preexec_fn=os.setsid` for child process tracking | Available |
| **GPU auto-detection** | `GPUDetector` — torch CUDA + TF GPU devices | Active |
| **VRAM polling** | Threaded `torch.cuda.max_memory_allocated()` sampling | Active |
| **CGroups memory (Slurm)** | `/sys/fs/cgroup/memory.current` when available | Passive (detect-only) |

## Future Improvements

- **CPU pinning**: `taskset` or `sched_setaffinity` per Slurm `$SLURM_LOCALID`
- **Memory limits**: `ulimit -v` + Slurm `--mem-per-cpu`
- **NUMA awareness**: `numactl --membind` for multi-socket nodes
- **Containerization**: Singularity/Docker images for full reproducibility
- **Energy measurement**: `nvidia-smi power.draw` + CPU TDP estimation

## Replication Checklist

For any published benchmark result, ensure:
1. Hardware: CPU model, RAM, GPU model + driver + CUDA version
2. Software: pixi.lock committed to repo (exact dependency versions)
3. N ≥ 3 independent runs, report mean ± SD
4. Separate cold-start (model load) from warm inference
5. Report throughput (seq/sec) and latency percentiles (p50, p95)
6. Disclose any extrapolation with sample size and method
