#!/usr/bin/env bash
set -e

# Ensure we run from project root
cd "$(dirname "$0")"

echo "Activating venv if exists..."
if [ -f "./venv/bin/activate" ]; then
  source ./venv/bin/activate
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training models (if missing)..."
python train_models.py

echo "Launching Gradio app..."
python app_gradio.py
