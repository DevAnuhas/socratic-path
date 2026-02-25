#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Train all 3 models sequentially
#
# Usage:
#   cd socratic-path
#   chmod +x scripts/run_all_training.sh
#   nohup bash scripts/run_all_training.sh > training_output.log 2>&1 &
#
# Monitor progress:
#   tail -f training_output.log
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on any error

echo "============================================================"
echo "SocraticPath — Training All 3 Models"
echo "Started: $(date)"
echo "============================================================"

# Detect project root (script is in socratic-path/scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# Check GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"

echo ""
echo "============================================================"
echo "[1/3] Training FLAN-T5-small + LoRA"
echo "============================================================"
python3 scripts/train_model.py --model flan-t5-small
echo "[1/3] FLAN-T5-small COMPLETE at $(date)"

echo ""
echo "============================================================"
echo "[2/3] Training FLAN-T5-base + LoRA"
echo "============================================================"
python3 scripts/train_model.py --model flan-t5-base
echo "[2/3] FLAN-T5-base COMPLETE at $(date)"

echo ""
echo "============================================================"
echo "[3/3] Training T5-base + LoRA (no instruction tuning)"
echo "============================================================"
python3 scripts/train_model.py --model t5-base
echo "[3/3] T5-base COMPLETE at $(date)"

echo ""
echo "============================================================"
echo "Generating comparative plots..."
echo "============================================================"
python3 scripts/plot_training_curves.py

echo ""
echo "============================================================"
echo "ALL TRAINING COMPLETE"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run evaluation: python scripts/evaluate_model.py --model all"
echo "  2. Check figures:   ls models/figures/"
echo "  3. Check adapters:  ls models/*/adapter/"
