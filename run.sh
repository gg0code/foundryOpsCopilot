#!/bin/bash
# Foundry AI Demo — one-command launch
# Installs deps, generates synthetic data (if missing), trains models (if missing),
# and starts the Flask app on port 5000.

set -e

echo "==> Installing Python dependencies..."
pip install -r requirements.txt -q

if [ ! -f data/heats_2025.csv ]; then
  echo "==> Generating metallurgically rigorous synthetic data..."
  python generate_data.py --verify
else
  echo "==> Data already present at data/heats_2025.csv (skipping generation)"
fi

if [ ! -f models/defect_classifier.pkl ]; then
  echo "==> Training models (XGBoost defect + severity, LightGBM yield + warranty, SHAP)..."
  python train_model.py
else
  echo "==> Models already trained (skipping)"
fi

echo "==> Starting Flask app on http://localhost:5000"
python app.py
