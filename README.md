# Bidirectional RAG: Cloud Deployment Package
## IEEE Access Publication - Full Experiment Suite

This folder contains everything needed to run IEEE Access-quality experiments on Vast.ai.

## Quick Start on Vast.ai

### 1. Rent a GPU Instance
- Go to [vast.ai](https://vast.ai)
- Choose: **RTX 3090 or A100** (~$0.25-0.50/hr)
- Min specs: 32GB RAM, 50GB disk
- Image: **nvidia/cuda:11.8-devel-ubuntu22.04** or similar

### 2. Upload This Folder
```bash
# Option A: Upload via SFTP/SCP
scp -r cloud_run/ root@YOUR_VAST_IP:/workspace/

# Option B: Clone from GitHub
git clone https://github.com/YOUR_USERNAME/bidirectional-rag.git
cd bidirectional-rag
```

### 3. Run Setup Script
```bash
cd /workspace/cloud_run
chmod +x scripts/cloud_setup.sh
./scripts/cloud_setup.sh
```

### 4. Run Experiments
```bash
# Full IEEE Access experiments (20 seeds, 480 experiments)
./scripts/run_ieee_experiments.sh

# OR quick test (1 seed, 12 experiments)
source venv/bin/activate
python main.py --corpus_type sparse --systems all --seeds 42 --num_queries 100 --max_workers 1
```

## Folder Structure

```
cloud_run/
├── main.py                    # Main experiment runner
├── fix_onnx_import.py         # ONNX compatibility fix
├── requirements.txt           # Python dependencies
├── REPRODUCE.md               # Detailed reproduction guide
├── src/
│   ├── systems/
│   │   └── baselines.py       # RAG system implementations
│   ├── evaluation/
│   │   ├── metrics.py         # Metrics calculator
│   │   └── ...
│   └── data/
│       └── dataset_loader.py  # Dataset loading
├── experiments/
│   ├── corpus_configurations.py  # Corpus setup
│   └── ...
├── scripts/
│   ├── cloud_setup.sh         # VM setup script
│   ├── run_ieee_experiments.sh # Full experiment script
│   └── precache_corpus.py     # Pre-cache Wikipedia corpus
└── data/
    ├── corpus_variants/       # Pre-cached corpora
    ├── raw/                   # Wikipedia article cache
    └── processed/             # Processed datasets
```

## Estimated Runtime & Cost

| Configuration | Experiments | Runtime | Cost (Vast.ai) |
|---------------|-------------|---------|----------------|
| Quick Test    | 12          | ~1 hour | ~$0.30         |
| Moderate      | 60          | ~6 hours| ~$2.00         |
| **Full (IEEE)**| **480**    | **24-48 hours** | **$12-25** |

## After Experiments

1. **Download results**: `scp -r root@VAST_IP:/workspace/cloud_run/results ./`
2. **Generate figures**: `python experiments/generate_figures.py`
3. **Generate tables**: `python experiments/generate_latex_tables.py`
4. **Update manuscript with new results**

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ollama connection refused` | Run `ollama serve &` first |
| `CUDA out of memory` | Reduce `--max_workers` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Wikipedia API errors` | Run `python scripts/precache_corpus.py` |

## Contact

Questions: chinthala511626@avila.edu
