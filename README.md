# BookHaven Anti-Crawler System

AI-powered bot detection system — course project for AI Principles (SOF106).

## Quick Start (after extracting the folder)

### 1. Install dependencies

```bash
cd anti-crawler-system
pip install -r requirements.txt
```

If PyTorch fails, follow the guide at [pytorch.org](https://pytorch.org/get-started/locally/).

> **Note for Graders:** Pre-trained model files (`random_forest.pkl` and `lstm_model.pt`) are already included inside the `data/models/` directory. If you just want to experience the live active-defense system immediately, you can safely skip Steps 2 to 4 and jump directly to Step 5.

### 2. Generate the 1,000-session training dataset

This creates `data/raw_logs.csv` with 23,000+ click records across three traffic types (human, naive bot, evasive bot) in a 1:1:1 ratio:

```bash
python -m ai_models.generate_large_dataset
```

### 3. Train the models

Train one or both models. Trained models are saved to `data/models/`.

```bash
# Random Forest (10 hand-crafted features)
python -m ai_models.train --model rf

# LSTM (page-click sequence model)
python -m ai_models.train --model lstm
```

**Full usage:**

```
python -m ai_models.train [--model {rf,lstm}] [--csv PATH] [--export PATH]
                          [--test-size FLOAT] [--n-estimators INT]
                          [--max-depth INT] [--seed INT]
                          [--seq-len INT] [--epochs INT] [--lr FLOAT]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `rf` | `rf` (Random Forest) or `lstm` |
| `--csv` | `data/raw_logs.csv` | Path to training data |
| `--export` | auto | Model output path |
| `--test-size` | `0.3` | Test set fraction **(RF only)** |
| `--n-estimators` | `200` | Number of trees **(RF only)** |
| `--max-depth` | `12` | Max tree depth **(RF only)** |
| `--seed` | `42` | Random seed |
| `--seq-len` | `10` | LSTM sequence length **(LSTM only)** |
| `--epochs` | `30` | Training epochs **(LSTM only)** |
| `--lr` | `0.001` | Learning rate **(LSTM only)** |

Examples:

```bash
# Train RF with custom parameters
python -m ai_models.train --model rf --n-estimators 300 --max-depth 15

# Train LSTM with more epochs
python -m ai_models.train --model lstm --epochs 50 --lr 0.0005

# Train from a custom CSV
python -m ai_models.train --model rf --csv data/my_logs.csv --export data/models/my_rf.pkl
```

### 4. Run the adversarial evaluation

Test whether the LSTM catches evasive bots that the Random Forest misses when timing features (interval, duration, rate) and User-Agent are fully spoofed — only the page-click sequence remains as a detectable signal.

```bash
python -m ai_models.evaluate_adversarial
```

Expected result: **RF recall ~0%, LSTM recall ~100%.**

The script runs four stages:
1. Load both trained models from `data/models/`
2. Load `data/raw_logs.csv`, keep only human + evasive-crawler sessions, then apply full-spectrum timing spoofing to evasive sessions (UA → 1.0, timing features → sampled from human distribution)
3. Evaluate Random Forest on the spoofed feature matrix
4. Evaluate LSTM on the raw page-click sequences (timing-agnostic)

Output includes per-model metrics, confusion matrices, side-by-side comparison, and a plain-English takeaway explaining *why* the recall gap exists.

> **Note:** Requires both `random_forest.pkl` and `lstm_model.pt` to exist in `data/models/`. Run Step 3 first if you haven't trained them yet.

### 5. Start the Flask website

```bash
python -m target_website.app
```

Opens at **http://127.0.0.1:5000**.

### 6. Start the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Opens at **http://127.0.0.1:8501** — three tabs: Live Monitoring, Threat Intelligence, AI Engine Analytics.

### 7. Run simulators to see live detection

In separate terminals:

```bash
python -m scrapers.simulate_human --users 5
python -m scrapers.simulate_crawler_naive --rounds 10
python -m scrapers.simulate_crawler_evasive --rounds 5
```

Watch the dashboard catch bots in real time — human clicks show in blue, crawlers in red.

## What's Inside

| Folder | Purpose |
|---|---|
| `target_website/` | Flask bookstore app with mock books, categories, cart |
| `ai_middleware/` | Flask hooks that log requests, extract features, and block bots |
| `ai_models/` | RF + LSTM training, feature engineering, adversarial eval |
| `dashboard/` | Streamlit monitoring dashboard with live activity feed |
| `scrapers/` | Traffic simulators (human, naive bot, evasive bot) |
| `data/` | CSV logs, trained model files |
