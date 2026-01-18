# 🎮 COMPLETE IEEE ACCESS EXPERIMENT GUIDE

**This runs ALL experiments exactly as described in the paper!**

---

## 📋 WHAT YOU'RE RUNNING

| Paper Element | Description | Command |
|---------------|-------------|---------|
| **Table I** | Threshold Sensitivity | Phase 4 |
| **Table II** | 6 Systems Comparison (Main Results) | Phase 5 |
| **Table III** | Ablation Study | Phase 6 |
| **Table IV** | Coverage by Dataset | Derived from Phase 5 |
| **Figure 1** | System Comparison | Phase 7 |
| **Figure 2** | Coverage by Dataset | Phase 7 |

### Key Settings (Matching Paper):
- **Corpus Type**: `sparse` (2,024 Stack Overflow docs)
- **Queries**: 500 per experiment
- **Seeds**: 42-61 (20 seeds for statistical significance)
- **Systems**: 6 (Standard RAG, Self-RAG, FLARE, CRAG, Naive Write-back, Bidirectional RAG)

---

## 📋 BEFORE YOU START

You need:
- Vast.ai account with $15 credit
- Your Gemini API key: `AIzaSyCa5aLCQXq8AFaOzN4YkT8HQddggUaVMH0`

---

# PHASE 1: Setup (30 minutes)

## Step 1: Rent Computer on Vast.ai

## Step 1: Rent Computer on Vast.ai (Detailed)

1. **Create Account**: Go to [vast.ai/console/create-account](https://vast.ai/console/create-account) and sign up.
2. **Add Funds**: Click **Billing** (left menu) → **Add Credit** → Add **$15.00**.
   - *Why $15?* Experiments take ~48 hours. $15 covers cost + buffer.
3. **Search for Computer**: Click **Create** (left menu).
4. **Set Filters** (Top Left or "Storage" section):
   - **Disk Space / Storage**: Drag slider to **50.0 GB**
     - *Note: If you don't see this filter, look for a "Storage" slider at the very top, or click "Rent" on a machine and set "Disk to Allocate" in the configuration popup.*
   - **RAM**: Drag slider to **32.0 GB**
   - **GPU RAM**: **12.0 GB**+ (Any modern GPU works since we use Gemini API)
5. **Set Filters** (Top Right):
   - Check **"Verified"** (for stability)
   - Uncheck **"Interruptible"** (You don't want your 48h experiment stopped!)
   - Sort by: **Price Increasing**
6. **Choose & Rent**:
   - Find a machine with **blue "Rent" button** (~$0.10-$0.20/hr).
   - Click **Rent**.
   - **Select Template**: Choose **"PyTorch"** (Official) or **"Cuda:12.0 Devel"**.
     - *Note: Don't pick complex ones. We just need basic Python + GPU.*
   - **Launch Mode**: Select **"Jupyter"** (CMD/Entrypoint should remain default).
   - **CONFIRM DISK SIZE**: In this same popup, make sure the slider is at **50GB**.
   - Click **Rent** again to confirm.
   - Go to **Instances** (left menu).
   - Wait ~2-5 minutes until Status says **"Running"**.
   - Click the blue **"Open"** button.
   - Click **"Jupyter"** (or just Open).
   - In Jupyter, look top-right → Click **New** → **Terminal**.

---

## Step 2: Download Code & Install

```bash
cd /workspace
git clone https://github.com/TejaCHINTHALA67/bidirectional-rag.git
cd bidirectional-rag

pip install -r requirements.txt

export GEMINI_API_KEY="AIzaSyCa5aLCQXq8AFaOzN4YkT8HQddggUaVMH0"
```

---

## Step 3: Verify Setup

```bash
python -c "import google.generativeai; print('Gemini OK')"
python -c "import chromadb; print('ChromaDB OK')"
python -c "from sentence_transformers import CrossEncoder; print('CrossEncoder OK')"
```

All should print "OK".

---

## Step 4: Quick Test (5 min)

```bash
python main.py --llm_model gemini-2.5-flash-preview-09-2025 --corpus_type sparse --systems standard_rag bidirectional_rag --seeds 42 --num_queries 10 --max_workers 1
```

If you see progress bars → SUCCESS! ✅

---

# PHASE 2: Prepare Datasets (10 min)

We need to download and sparsify the datasets (cutting 50% of information) to match the paper's setup.

```bash
python scripts/prepare_datasets.py --dataset all --num_docs 2024 --num_queries 500
```

**What this does:**
- Downloads NQ, TriviaQA, HotpotQA, and StackOverflow
- Creates "Sparse" versions (keeps only 2,024 docs per dataset)
- Ensures we have exactly 500 test queries

---

# PHASE 3: Verify Corpus Configuration (1 min)

Double check that the "sparse" corpus is correctly set up.

```bash
python scripts/corpus_configurations.py --verify --corpus_type sparse
```

If it says **"VERIFICATION PASSED"**, proceed to Phase 4.

---

# PHASE 4: Threshold Sensitivity - TABLE I (30 min)

```bash
python scripts/threshold_sensitivity.py \
    --corpus_type sparse \
    --dataset stackoverflow \
    --num_queries 25 \
    --nli_thresholds 0.50 0.55 0.60 0.65 0.75 0.80 \
    --sim_thresholds 0.80 0.85 0.90 0.95 \
    --output results/threshold_sensitivity.json
```

**What this does:** Tests different grounding (NLI) and novelty thresholds.

---

# PHASE 5: Main Experiments - TABLE II (24-48 hours) ⭐

**THIS IS THE BIG ONE!**

```bash
python main.py \
    --llm_model gemini-2.5-flash-preview-09-2025 \
    --corpus_type sparse \
    --systems all \
    --datasets nq triviaqa hotpotqa stackoverflow \
    --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
    --num_queries 500 \
    --max_workers 2 \
    --output_dir results/main_experiments
```

**What this runs:**
- ✅ 6 systems × 4 datasets × 20 seeds = **480 experiments**
- ✅ 500 queries each = **240,000 total queries**

**You can close the browser - it keeps running!**

---

# PHASE 6: Ablation Study - TABLE III (4-8 hours)

After Phase 5 completes, run:

```bash
python experiments/run_experiments.py \
    --sparse-corpus \
    --queries 500 \
    --output-dir results/ablation
```

**What this tests:**
| Variant | Grounding | Attribution | Novelty |
|---------|-----------|-------------|---------|
| Full Method | ✅ | ✅ | ✅ |
| No Grounding | ❌ | ✅ | ✅ |
| No Attribution | ✅ | ❌ | ✅ |
| No Novelty | ✅ | ✅ | ❌ |
| No Validation | ❌ | ❌ | ❌ |

---

# PHASE 7: Generate Tables & Figures (10 min)

```bash
# Statistical Analysis
python experiments/statistical_analysis.py \
    --input results/main_experiments \
    --output results/statistical_report.json

# Generate Figures
python experiments/generate_figures.py \
    --results_dir results/main_experiments \
    --output_dir results/figures

# Generate LaTeX Tables
python experiments/generate_latex_tables.py \
    --results_dir results/main_experiments \
    --output_dir results/tables
```

---

# PHASE 8: Download Results

1. In Jupyter, find the `results` folder
2. Right-click → **Download as ZIP**
3. Save to your computer

---

# PHASE 9: DESTROY THE COMPUTER! 🛑💰

**CRITICAL - DO THIS OR YOU KEEP PAYING!**

1. Go to vast.ai → **"Instances"**
2. Click the red **"DESTROY"** button
3. Confirm

---

## 📁 YOUR RESULTS FOLDER

```
results/
├── threshold_sensitivity.json    # TABLE I
├── main_experiments/
│   ├── experiment_summary.json   # TABLE II
│   └── per_dataset/              # TABLE IV
├── ablation/                     # TABLE III
├── statistical_report.json       # p-values, effect sizes
├── figures/                      # FIGURES 1-2
└── tables/                       # LaTeX tables
```

---

## ⏱️ TIMELINE

| Phase | Time | Can Close Browser? |
|-------|------|-------------------|
| 1-3: Setup | 30 min | No |
| 4: Threshold | 30 min | No |
| 5: Main (480 exp) | 24-48 hrs | ✅ YES |
| 6: Ablation | 4-8 hrs | ✅ YES |
| 7: Post-process | 10 min | No |
| **TOTAL** | **~30-50 hours** | — |

---

## 💰 COST

| Item | Cost |
|------|------|
| Vast.ai (~50 hours × $0.15/hr) | ~$7.50 |
| Gemini API | FREE |
| **TOTAL** | **~$8** |

---

## ✅ FINAL CHECKLIST

- [ ] Vast.ai has 50GB disk, 32GB RAM
- [ ] pip install completed
- [ ] GEMINI_API_KEY exported
- [ ] Quick test passed
- [ ] Using `--corpus_type sparse` ⚠️ IMPORTANT
- [ ] Using `--num_queries 500`
- [ ] Ran Threshold Sensitivity (Phase 4)
- [ ] Ran Main Experiments (Phase 5)
- [ ] Ran Ablation Study (Phase 6)
- [ ] Downloaded results
- [ ] **DESTROYED the instance!**

---

## ❓ TROUBLESHOOTING

| Error | Fix |
|-------|-----|
| "API key invalid" | `export GEMINI_API_KEY="..."` again |
| "Rate limit" | Reduce `--max_workers` to 1 |
| "Module not found" | `pip install -r requirements.txt` |
| "Out of disk" | `rm -rf results/*` |

---

**🎉 Good luck with your IEEE Access paper!**
