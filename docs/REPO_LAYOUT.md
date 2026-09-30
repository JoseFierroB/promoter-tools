# Repository layout — promoter-tools (product view)

> `output/` está gitignored, pero **se genera al correr el benchmark** en
> cualquier máquina. Este esquema muestra el árbol completo: código +
> generado. Todo lo generado cuelga de `output/` y de `*_predictions`.

```
promoter-tools/
├── pipeline/run_pipeline.sh        # datasets | splits | benchmark | analysis | all
├── src/
│   ├── cli.py                      # run/compare (+ RUN.log por run)
│   ├── config.py                   # intérpretes pixi, rutas, envs
│   ├── benchmark/tools.py          # registro 9 tools (+ tools.d/meme.toml,
│   │                               #   tools.d/fimo_prok.toml)
│   ├── backend/{local,slurm}.py    # ejecución + métricas wall/RAM/VRAM
│   ├── dataset/                    # extractores D39V/TIGR4 + splits (seed 42)
│   ├── runners/                    # 9 contratos: meme, fimo_prok, mldspp,
│   │                               #   mldspp_75, lcnn, promotech_hot,
│   │                               #   ipromp_sp12, prompt, prokbert_mini
│   ├── analysis/                   # analyze, compute, sigma, operating_points,
│   │                               #   compare_runs, plot_gpu_compare,
│   │                               #   plot_scaling, bench_labels (paleta v4)
│   └── utils/metrics.py            # wall/time_s/RAM/VRAM (NVML + nvidia-smi)
├── experiments/{rpod_replication,igr,datasets}/
├── tests/                          # contrato + cobertura + layout + smoke
├── data/                           # inputs versionados (datasets, scale_db, splits)
├── docs/ · tools/ (envs vendored) · models/ (pesos, ver DOWNLOADS.md)
├── archive/{legacy_dead,dataset_legacy}/   # código retirado, reversible
└── output/  (GENERADO, gitignored)
    ├── <YYMMDD>_runs/<name>/<config>/      # un run = una cápsula
    │   ├── MANIFEST.json / STATUS.json / RUN.log
    │   ├── 1_inference/predictions/{name}_{fam}*  # scores por tool
    │   ├── 1_inference/roc_auc_{name}.{png,pdf}   # (--plots)
    │   ├── 1_inference/sigma/roc_{SigA,ComX,None}_*  # (sigma_stratify)
    │   ├── 2_resources/resource_metrics.tsv        # siempre (con/sin --plots)
    │   ├── 2_resources/compute_time|peak_ram.{png,pdf}  # (--plots)
    │   └── 3_tables/metrics_rows.tsv               # dataset,tool,n,AUC
    │       3_tables/sigma_factor_benchmark_metrics.tsv      # 27 filas
    │       3_tables/operating_point_metrics{,_sigma,_strain}.tsv  # Youden
    │       3_tables/resource_metrics.tsv -> ../2_resources/  # symlink
    ├── comparisons/<YYMMDD>_* /       # vistas ligeras (sin copias de scores)
    │   ├── MANIFEST.json             # sources, regimes/shas, notes
    │   ├── 2_resources/compare_{time,ram,speedup,vram}|gpu_*|scaling_*  # figs
    │   └── 3_tables/resources_compare|scaling_all|gpu_compare.tsv
    ├── experiments/prokbert_ab/*.tsv  # dato tras ladder A/B
    ├── runs/                          # aliases machine-local (no rsync)
    ├── tables/ · intergenic*/ · mmseqs2/ · tigr4_data/  # SHARED IGR DATA PLANE:
    │       # live inputs read by experiments/igr/ (do not move/rename);
    │       # regenerable via extract scripts; versioned sets live in data/
    └── _archive/                      # retirados, reversible
```

Reglas: números de dataset solo en dirnames (`DS_DISPLAY` para títulos);
figuras siempre png+pdf; tablas TSV con `\t`; comparativas nunca duplican
predicciones; `output/runs/` no se sincroniza entre máquinas.
