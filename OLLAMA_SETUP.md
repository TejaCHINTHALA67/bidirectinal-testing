
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

### Step D: Run Experiments 
This script will now run 480 experiments (20 seeds x 6 systems x 4 datasets) using the local model.
```bash
./scripts/run_ieee_experiments.sh
```

## 3. Monitor Progress
You can see the progress bars in the terminal.
- **Estimated Time:** 2-5 hours (depending on GPU speed).
- **Results:** Saved automatically to `results/ieee_access_final`.
- **Logs:** If it freezes, check `ollama.log` using `tail -f ollama.log`.
