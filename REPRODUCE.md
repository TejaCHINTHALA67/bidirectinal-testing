# Reproducibility Guide: Bidirectional RAG
## IEEE Access Publication

This guide provides complete instructions to reproduce all experimental results.

---

## 🖥️ Option 1: Cloud Deployment (Recommended)

### A. Google Cloud Platform (Recommended)

```bash
# 1. Create VM with GPU
gcloud compute instances create bidirectional-rag \
    --machine-type=n1-highmem-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --zone=us-central1-a

# 2. SSH into VM
gcloud compute ssh bidirectional-rag --zone=us-central1-a

# 3. Run setup script
wget https://raw.githubusercontent.com/TejaCHINTHALA67/bidirectional-rag/main/scripts/cloud_setup.sh
chmod +x cloud_setup.sh
./cloud_setup.sh

# 4. Run experiments
./scripts/run_ieee_experiments.sh
```

**Cost**: ~$15-40 for full experiments (preemptible: ~$20)  
**Runtime**: 24-48 hours

### B. AWS EC2

```bash
# Instance: g4dn.2xlarge (8 vCPUs, 32GB RAM, T4 GPU)
# Same setup steps as above
```

### C. Vast.ai (Cheapest)

```bash
# Rent RTX 3090 (~$0.25/hour)
# ~$12 for full experiments
```

---

## 💻 Option 2: Local Setup

### Prerequisites
- Python 3.11+
- RAM: 32GB+ (experiments use 12 parallel workers)
- Disk: 5GB for datasets and corpora
- Ollama with llama3.2:3b model

### Installation

```bash
# 1. Clone repository
git clone https://github.com/TejaCHINTHALA67/bidirectional-rag.git
cd bidirectional-rag

# 2. Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama and pull model
# Download from https://ollama.com
ollama pull llama3.2:3b

# 5. Pre-cache corpus (avoids Wikipedia API issues during experiments)
python scripts/precache_corpus.py
```

---

## 🧪 Running Experiments

### Full IEEE Access Experiments (Publication Quality)

```bash
# 480 experiments: 4 datasets × 6 systems × 20 seeds × 1000 queries
python main.py \
    --corpus_type realistic \
    --systems all \
    --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
    --num_queries 1000 \
    --max_workers 4  # Reduce if memory issues
```

### Quick Verification (Smoke Test)

```bash
# 12 experiments: 4 datasets × 3 systems × 1 seed × 100 queries
python main.py \
    --corpus_type sparse \
    --systems standard_rag naive_writeback bidirectional_rag \
    --seeds 42 \
    --num_queries 100 \
    --max_workers 1
```

---

## 📊 Post-Experiment Analysis

### 1. Statistical Analysis
```bash
python experiments/run_statistical_analysis.py
```

### 2. Generate Figures
```bash
python experiments/generate_figures.py
```

### 3. Generate LaTeX Tables
```bash
python experiments/generate_latex_tables.py
```

### 4. Compile PDF
```bash
cd "ieee publishing/paper"
pdflatex bidirectional_rag_ieee_v2.tex
bibtex bidirectional_rag_ieee_v2
pdflatex bidirectional_rag_ieee_v2.tex
pdflatex bidirectional_rag_ieee_v2.tex
```

---

## 📁 Output Structure

```
results/
├── experiment_summary.json          # All experiment results
├── statistical_analysis_report.txt  # Statistical comparisons
├── figures/
│   ├── system_comparison.pdf
│   ├── coverage_by_dataset.pdf
│   └── coverage_evolution.pdf
└── latex_tables/
    ├── main_results.tex
    └── ablation.tex
```

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `Ollama connection error` | Ensure `ollama serve` is running |
| `WinError 1455` (paging file) | Reduce `--max_workers 1` or use cloud |
| `Wikipedia API errors` | Run `python scripts/precache_corpus.py` first |
| Memory issues | Use cloud VM with 32GB+ RAM |

---

## 📧 Contact

For questions: chinthala511626@avila.edu

