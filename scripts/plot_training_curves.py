#!/usr/bin/env python3
"""
Generate comparative training curves across all trained models.

Reads CSV logs from each model's training run and produces figures.

Outputs:
  1. Combined training loss curves (all models, one plot)
  2. Combined validation loss curves (overfitting analysis)
  3. Combined ROUGE-L learning curves
  4. Per-model detailed training curves (3-panel)

Usage:
  python scripts/plot_training_curves.py

Output:
  models/figures/comparative_training_loss.png
  models/figures/comparative_validation_loss.png
  models/figures/comparative_rougeL.png
  models/figures/comparative_overfitting.png
  models/figures/model_comparison_table.txt
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration

MODEL_DIRS = {
    "FLAN-T5-small + LoRA": "flan-t5-small-lora",
    "FLAN-T5-base + LoRA": "flan-t5-base-lora",
    "T5-base + LoRA\n(no instruction tuning)": "t5-base-lora",
}

COLORS = {
    "FLAN-T5-small + LoRA": "#2196F3",           # Blue
    "FLAN-T5-base + LoRA": "#FF5722",            # Red-Orange
    "T5-base + LoRA\n(no instruction tuning)": "#4CAF50",  # Green
}

# Plot settings
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox_inches": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_model_logs(project_root: Path):
    """Load training and evaluation logs for all available models."""
    models = {}

    for label, dir_name in MODEL_DIRS.items():
        log_dir = project_root / "models" / dir_name / "logs"
        train_log = log_dir / "training_log.csv"
        eval_log = log_dir / "eval_log.csv"
        summary_path = project_root / "models" / dir_name / "adapter" / "training_summary.json"

        if not train_log.exists():
            print(f"  Skipping {label}: no training log found at {train_log}")
            continue

        train_df = pd.read_csv(train_log)
        eval_df = pd.read_csv(eval_log) if eval_log.exists() else pd.DataFrame()

        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

        models[label] = {
            "train": train_df,
            "eval": eval_df,
            "summary": summary,
            "dir_name": dir_name,
        }
        print(f"  Loaded {label}: {len(train_df)} train steps, {len(eval_df)} eval steps")

    return models


def plot_comparative_training_loss(models: dict, output_dir: Path):
    """Plot training loss curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in models.items():
        train_df = data["train"]
        if train_df.empty:
            continue

        # Raw loss (semi-transparent)
        ax.plot(train_df["step"], train_df["loss"],
                color=COLORS[label], alpha=0.2, linewidth=0.5)

        # Smoothed loss
        window = max(5, len(train_df) // 30)
        smoothed = train_df["loss"].rolling(window=window, center=True).mean()
        ax.plot(train_df["step"], smoothed,
                color=COLORS[label], linewidth=2.5, label=label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Comparative Training Loss Curves")
    ax.legend(loc="upper right")

    output_path = output_dir / "comparative_training_loss.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comparative_validation_loss(models: dict, output_dir: Path):
    """Plot validation loss curves for overfitting analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in models.items():
        eval_df = data["eval"]
        if eval_df.empty:
            continue

        ax.plot(eval_df["step"], eval_df["eval_loss"],
                color=COLORS[label], marker="o", markersize=3,
                linewidth=1.5, label=label)

        # Mark best checkpoint (lowest validation loss)
        best_idx = eval_df["eval_loss"].idxmin()
        best_step = eval_df.loc[best_idx, "step"]
        best_loss = eval_df.loc[best_idx, "eval_loss"]
        ax.scatter([best_step], [best_loss], color=COLORS[label],
                   s=100, zorder=5, edgecolors="black", linewidths=1)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Comparative Validation Loss (Overfitting Analysis)")
    ax.legend(loc="upper right")

    output_path = output_dir / "comparative_validation_loss.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comparative_rougeL(models: dict, output_dir: Path):
    """Plot ROUGE-L learning curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in models.items():
        eval_df = data["eval"]
        if eval_df.empty or "eval_rougeL" not in eval_df.columns:
            continue

        ax.plot(eval_df["step"], eval_df["eval_rougeL"],
                color=COLORS[label], marker="o", markersize=3,
                linewidth=1.5, label=label)

        # Mark best ROUGE-L
        best_idx = eval_df["eval_rougeL"].idxmax()
        best_step = eval_df.loc[best_idx, "step"]
        best_rouge = eval_df.loc[best_idx, "eval_rougeL"]
        ax.scatter([best_step], [best_rouge], color=COLORS[label],
                   s=100, zorder=5, edgecolors="black", linewidths=1)
        ax.annotate(f"{best_rouge:.4f}",
                    xy=(best_step, best_rouge),
                    xytext=(10, 5), textcoords="offset points",
                    fontsize=9, color=COLORS[label])

    ax.set_xlabel("Training Step")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("Comparative Validation ROUGE-L Learning Curves")
    ax.legend(loc="lower right")

    output_path = output_dir / "comparative_rougeL.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_overfitting_analysis(models: dict, output_dir: Path):
    """Plot training vs validation loss per model for overfitting detection."""
    n_models = len(models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, models.items()):
        train_df = data["train"]
        eval_df = data["eval"]

        if not train_df.empty:
            window = max(5, len(train_df) // 30)
            smoothed = train_df["loss"].rolling(window=window, center=True).mean()
            ax.plot(train_df["step"], smoothed,
                    color="steelblue", linewidth=2, label="Training Loss")

        if not eval_df.empty:
            ax.plot(eval_df["step"], eval_df["eval_loss"],
                    color="crimson", marker="o", markersize=3,
                    linewidth=1.5, label="Validation Loss")

            # Detect overfitting: find where val loss starts increasing
            best_idx = eval_df["eval_loss"].idxmin()
            best_step = eval_df.loc[best_idx, "step"]
            ax.axvline(x=best_step, color="green", linestyle="--",
                       alpha=0.7, label=f"Best (step {best_step})")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle("Overfitting Analysis — Training vs Validation Loss",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "comparative_overfitting.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_comparison_table(models: dict, output_dir: Path):
    """Generate a text comparison table of final metrics."""
    rows = []

    for label, data in models.items():
        summary = data["summary"]
        if not summary:
            continue

        final = summary.get("final_metrics", {})
        baseline = summary.get("baseline_metrics", {})
        rows.append({
            "Model": label.replace("\n", " "),
            "Params": f"{summary.get('total_params', 0) / 1e6:.0f}M",
            "Trainable": f"{summary.get('trainable_percent', 0):.2f}%",
            "Train Time": f"{summary.get('training_time_minutes', 0):.0f} min",
            "Train Loss": f"{summary.get('train_loss', 0):.4f}",
            "Val Loss": f"{final.get('eval_loss', 0):.4f}",
            "ROUGE-1": f"{final.get('eval_rouge1', 0):.4f}",
            "ROUGE-2": f"{final.get('eval_rouge2', 0):.4f}",
            "ROUGE-L": f"{final.get('eval_rougeL', 0):.4f}",
        })

    if not rows:
        print("  No summaries found for comparison table.")
        return

    df = pd.DataFrame(rows)
    table_str = df.to_string(index=False)

    output_path = output_dir / "model_comparison_table.txt"
    with open(output_path, "w") as f:
        f.write("MODEL COMPARISON TABLE\n")
        f.write("=" * 120 + "\n")
        f.write(table_str + "\n")
        f.write("=" * 120 + "\n")

    print(f"  Saved: {output_path}")
    print(f"\n{table_str}")


def main():
    # Auto-detect project root
    project_root = Path(__file__).resolve().parent.parent
    print(f"Project root: {project_root}")

    output_dir = project_root / "models" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading model logs...")
    models = load_model_logs(project_root)

    if not models:
        print("\nNo trained models found. Train at least one model first:")
        print("  python scripts/train_model.py --model flan-t5-small")
        sys.exit(1)

    print(f"\nFound {len(models)} trained model(s). Generating figures...")

    plot_comparative_training_loss(models, output_dir)
    plot_comparative_validation_loss(models, output_dir)
    plot_comparative_rougeL(models, output_dir)
    plot_overfitting_analysis(models, output_dir)
    generate_comparison_table(models, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
