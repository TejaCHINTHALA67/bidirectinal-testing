
# 🦙 Run Experiments with Local Ollama (Step-by-Step)

This guide replaces the Gemini setup. We will run the entire experiment suite using a local 8B model (`llama3`) on your GPU. This avoids all API limits and safety blocks.

## 1. Local Terminal: Push Changes
First, enable the new configuration by pushing the updated code.

```powershell
git add .
git commit -m "Switch to local Ollama Llama 3 experiments"
git push
```

## 2. Vast.ai Terminal: Setup & Run
Login to your Vast.ai instance (Jupyter Terminal) and run:

### Step A: Update Code
```bash
cd /workspace/bidirectinal-testing
git pull
```

### Step B: Install & Start Ollama
(If you haven't already install it)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama in the background
ollama serve > ollama.log 2>&1 &
```
*Wait 10 seconds for it to start...*

### Step C: Download Model
Download the 8B Llama 3 model (approx 4.7GB).
```bash
ollama pull llama3
```
*Wait for the download to complete (100%).*

### Step D: Verification (Smoke Test)
**Highly Recommended:** Run this check to verify everything works before the main event.
```bash
python main.py --llm_model llama3 --corpus_type sparse --systems standard_rag bidirectional_rag --seeds 42 --num_queries 10 --max_workers 1
```
*If this finishes successfully (takes ~1-2 mins), proceed to Step E.*

### Step E: Run Experiments (Background Mode)
**Crucial:** Run this way so it keeps running if your internet disconnects.

```bash
nohup ./scripts/run_ieee_experiments.sh > experiment_log.txt 2>&1 &
```

## 3. Monitor Progress
Since it's running in the background, use this command to check the output:
```bash
tail -f experiment_log.txt
```
*(Press `Ctrl+C` to stop watching the log—the experiment will keep running.)*

- **Estimated Time:** 2-5 hours.
- **Results:** Saved automatically to `results/ieee_access_final`.
- **Verify Running:** Type `top` or `ps aux | grep python` to see it working.
