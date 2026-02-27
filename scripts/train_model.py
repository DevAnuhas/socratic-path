#!/usr/bin/env python3
"""
Train a T5-family model with LoRA for Socratic Question Generation.

Supports three models for comparative analysis:
  1. FLAN-T5-small (77M)  — instruction-tuned, small
  2. FLAN-T5-base  (250M) — instruction-tuned, medium
  3. T5-base       (250M) — vanilla (no instruction tuning)

Usage:
  python scripts/train_model.py --model flan-t5-small
  python scripts/train_model.py --model flan-t5-base
  python scripts/train_model.py --model t5-base

All models use identical LoRA configuration and hyperparameters for fair
comparison. Training logs (CSV) and loss curves (PNG) are saved automatically.

Designed for EC2 g5.2xlarge (NVIDIA A10G, 24GB VRAM).
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════
# Each model is identified by a short name. The HuggingFace model ID, output
# directory name, and description are defined here.

MODEL_REGISTRY = {
    "flan-t5-small": {
        "hf_name": "google/flan-t5-small",
        "output_dir": "flan-t5-small-lora",
        "description": "FLAN-T5-small (77M params) — instruction-tuned, smallest variant",
    },
    "flan-t5-base": {
        "hf_name": "google/flan-t5-base",
        "output_dir": "flan-t5-base-lora",
        "description": "FLAN-T5-base (250M params) — instruction-tuned, medium variant",
    },
    "t5-base": {
        "hf_name": "t5-base",
        "output_dir": "t5-base-lora",
        "description": "T5-base (250M params) — vanilla, NO instruction tuning",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Identical across all models for fair comparison.

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    "modules_to_save": ["embed_tokens", "lm_head"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": TaskType.SEQ_2_SEQ_LM,
}

# Precision: bf16 on Ampere+ GPUs (A10G, A100), fp16 on older CUDA, fp32 on CPU.
# bf16 has the same exponent range as fp32 — avoids loss scaling issues that fp16 can hit.
if torch.cuda.is_available():
    _ampere_or_newer = torch.cuda.get_device_capability()[0] >= 8
    _USE_BF16 = _ampere_or_newer
    _USE_FP16 = not _ampere_or_newer
else:
    _USE_BF16 = False
    _USE_FP16 = False

TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 16,   # Smaller batch → 2x more gradient updates per epoch
    "per_device_eval_batch_size": 64,    # No gradients — max batch for fastest eval
    "gradient_accumulation_steps": 1,    # Effective batch = 16
    "num_train_epochs": 10,              # 10 epochs; cosine schedule needs headroom to learn
    "lr_scheduler_type": "cosine",
    "warmup_fraction": 0.06,             # ~6% warmup; converted to warmup_steps at runtime
    "weight_decay": 0.01,
    "max_target_length": 80,
    "fp16": _USE_FP16,
    "bf16": _USE_BF16,
    "seed": 42,
    "eval_num_beams": 4,
    "eval_do_sample": False,
    "logging_steps": 50,
    "eval_steps": 2000,
    "save_steps": 2000,
    "early_stopping_patience": 5,        # 5 × 2000 = 10000 steps grace (~2 epochs at batch=16)
    "early_stopping_threshold": 0.0,     # ANY improvement resets patience
}


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM TRAINING LOGGER CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════
# Logs training loss, validation loss, and ROUGE scores to CSV files for
# generating training curves in the thesis.

class TrainingLogger(TrainerCallback):
    """Logs training and evaluation metrics to CSV files."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.train_log_path = log_dir / "training_log.csv"
        self.eval_log_path = log_dir / "eval_log.csv"

        with open(self.train_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "loss", "learning_rate"])

        with open(self.eval_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "epoch", "eval_loss",
                "eval_rouge1", "eval_rouge2", "eval_rougeL",
            ])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step
        epoch = round(state.epoch, 4) if state.epoch else 0

        # Training loss (logged every logging_steps)
        if "loss" in logs and "eval_loss" not in logs:
            with open(self.train_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, epoch,
                    round(logs["loss"], 6),
                    logs.get("learning_rate", ""),
                ])

        # Evaluation metrics (logged every eval_steps)
        if "eval_loss" in logs:
            with open(self.eval_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, epoch,
                    round(logs["eval_loss"], 6),
                    round(logs.get("eval_rouge1", 0), 6),
                    round(logs.get("eval_rouge2", 0), 6),
                    round(logs.get("eval_rougeL", 0), 6),
                ])


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {device} — {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = "cpu"
        print(f"  Device: {device} (WARNING: training will be very slow)")
    return device


def plot_training_curves(log_dir: Path, model_name: str):
    """Generate training/validation loss curves from CSV logs."""
    train_log = log_dir / "training_log.csv"
    eval_log = log_dir / "eval_log.csv"

    if not train_log.exists() or not eval_log.exists():
        print("  Warning: Log files not found, skipping plot generation.")
        return

    import pandas as pd

    train_df = pd.read_csv(train_log)
    eval_df = pd.read_csv(eval_log)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")

    # Plot 1: Training Loss
    axes[0].plot(train_df["step"], train_df["loss"], color="steelblue", alpha=0.7, linewidth=0.8)
    # Add smoothed line (rolling average)
    if len(train_df) > 10:
        window = max(5, len(train_df) // 20)
        smoothed = train_df["loss"].rolling(window=window, center=True).mean()
        axes[0].plot(train_df["step"], smoothed, color="navy", linewidth=2, label=f"Smoothed (window={window})")
        axes[0].legend()
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Training Loss vs Validation Loss (overfitting detection)
    axes[1].plot(eval_df["step"], eval_df["eval_loss"], color="crimson", marker="o",
                 markersize=3, linewidth=1.5, label="Validation Loss")
    # Overlay training loss at eval steps (interpolated)
    if len(train_df) > 0 and len(eval_df) > 0:
        train_at_eval = []
        for eval_step in eval_df["step"]:
            nearby = train_df[train_df["step"] <= eval_step]
            if len(nearby) > 0:
                train_at_eval.append(nearby["loss"].iloc[-1])
            else:
                train_at_eval.append(None)
        axes[1].plot(eval_df["step"], train_at_eval, color="steelblue", marker="s",
                     markersize=3, linewidth=1.5, label="Training Loss")
    # Mark best checkpoint
    if len(eval_df) > 0:
        best_idx = eval_df["eval_loss"].idxmin()
        best_step = eval_df.loc[best_idx, "step"]
        best_loss = eval_df.loc[best_idx, "eval_loss"]
        axes[1].axvline(x=best_step, color="green", linestyle="--", alpha=0.7,
                        label=f"Best checkpoint (step {best_step})")
        axes[1].scatter([best_step], [best_loss], color="green", s=100, zorder=5)
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Overfitting Analysis\n(Train vs Validation Loss)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Validation ROUGE-L
    axes[2].plot(eval_df["step"], eval_df["eval_rougeL"], color="seagreen", marker="o",
                 markersize=3, linewidth=1.5)
    if len(eval_df) > 0:
        best_rouge_idx = eval_df["eval_rougeL"].idxmax()
        best_rouge_step = eval_df.loc[best_rouge_idx, "step"]
        best_rouge = eval_df.loc[best_rouge_idx, "eval_rougeL"]
        axes[2].axvline(x=best_rouge_step, color="green", linestyle="--", alpha=0.7,
                        label=f"Best ROUGE-L: {best_rouge:.4f} (step {best_rouge_step})")
        axes[2].scatter([best_rouge_step], [best_rouge], color="green", s=100, zorder=5)
        axes[2].legend()
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("ROUGE-L")
    axes[2].set_title("Validation ROUGE-L")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = log_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model_key: str, project_root: Path):
    """Train a single model with LoRA and save all artifacts."""

    if model_key not in MODEL_REGISTRY:
        print(f"Error: Unknown model '{model_key}'.")
        print(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    model_info = MODEL_REGISTRY[model_key]
    hf_name = model_info["hf_name"]
    output_dir_name = model_info["output_dir"]

    print("=" * 70)
    print(f"TRAINING: {model_info['description']}")
    print(f"HuggingFace ID: {hf_name}")
    print("=" * 70)

    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir = project_root / "datasets" / "processed"
    model_output_dir = project_root / "models" / output_dir_name
    adapter_path = model_output_dir / "adapter"
    merged_path = model_output_dir / "merged"
    checkpoint_path = model_output_dir / "checkpoints"
    log_dir = model_output_dir / "logs"

    for p in [adapter_path, merged_path, checkpoint_path, log_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    device = get_device()
    set_seed(TRAINING_CONFIG["seed"])

    # ── Dataset ───────────────────────────────────────────────────────────
    print("\nLoading tokenized dataset...")
    dataset = load_from_disk(str(data_dir / "soqg_tokenized"))
    print(f"  Train:      {len(dataset['train']):,} samples")
    print(f"  Validation: {len(dataset['validation']):,} samples")
    print(f"  Test:       {len(dataset['test']):,} samples")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer_path = data_dir / "tokenizer"
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    print(f"  Vocabulary size: {len(tokenizer)} (includes [Question] token)")

    # ── Base Model ────────────────────────────────────────────────────────
    print(f"\nLoading base model: {hf_name}")
    base_model = T5ForConditionalGeneration.from_pretrained(hf_name)
    base_model.resize_token_embeddings(len(tokenizer))
    base_params = base_model.num_parameters()
    print(f"  Base model parameters: {base_params:,}")

    # ── LoRA Configuration ────────────────────────────────────────────────
    print("\nApplying LoRA adapter...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = model.num_parameters()
    trainable_pct = trainable / total * 100
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({trainable_pct:.2f}%)")
    print(f"  Parameter reduction:  {total / trainable:.0f}x fewer trainable parameters")

    # ── Override generation config ────────────────────────────────────────
    model.generation_config.max_length = TRAINING_CONFIG["max_target_length"]
    model.generation_config.num_beams = TRAINING_CONFIG["eval_num_beams"]
    model.generation_config.do_sample = TRAINING_CONFIG["eval_do_sample"]
    model.generation_config.early_stopping = True
    print(f"  Generation config: max_length={model.generation_config.max_length}, "
          f"num_beams={model.generation_config.num_beams}")

    # ── Metrics ───────────────────────────────────────────────────────────
    rouge_metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.replace("[Question]", "").strip() for p in decoded_preds]
        decoded_labels = [l.replace("[Question]", "").strip() for l in decoded_labels]
        result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
        }

    # ── Data Collator ─────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # ── Training Arguments ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"{output_dir_name}-{timestamp}"

    # Compute warmup_steps from fraction
    train_samples = len(dataset["train"])
    batch = TRAINING_CONFIG["per_device_train_batch_size"]
    accum = TRAINING_CONFIG["gradient_accumulation_steps"]
    steps_per_epoch = (train_samples + batch - 1) // batch // accum
    total_steps = steps_per_epoch * TRAINING_CONFIG["num_train_epochs"]
    warmup_steps = int(total_steps * TRAINING_CONFIG["warmup_fraction"])
    print(f"\n  Total training steps:  {total_steps:,}")
    print(f"  Warmup steps:          {warmup_steps} ({TRAINING_CONFIG['warmup_fraction']:.0%} of total)")

    # Set tensorboard dir via env var
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(log_dir / "tensorboard")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_path),
        run_name=run_name,
        # Epochs & batch
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        # Optimiser
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_steps=warmup_steps,
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        # Precision
        fp16=TRAINING_CONFIG["fp16"],
        bf16=TRAINING_CONFIG["bf16"],
        # Checkpointing
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        # Logging
        logging_steps=TRAINING_CONFIG["logging_steps"],
        report_to="tensorboard",
        # Generation (evaluation only)
        predict_with_generate=True,
        generation_max_length=TRAINING_CONFIG["max_target_length"],
        generation_num_beams=TRAINING_CONFIG["eval_num_beams"],
        # Reproducibility
        seed=TRAINING_CONFIG["seed"],
        dataloader_num_workers=4,
        dataloader_pin_memory=device == "cuda",
        # Gradient checkpointing: only for base-size models
        gradient_checkpointing=(device == "cuda" and "small" not in model_key),
    )

    # ── GPU Optimisations ─────────────────────────────────────────────────
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        print("  TF32 enabled for faster training on Ampere GPUs")

    # ── Callbacks ─────────────────────────────────────────────────────────
    training_logger = TrainingLogger(log_dir)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"],
        early_stopping_threshold=TRAINING_CONFIG["early_stopping_threshold"],
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[training_logger, early_stopping],
    )

    print(f"\n  Run name:              {run_name}")
    print(f"  Effective batch size:  {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"  Learning rate:         {TRAINING_CONFIG['learning_rate']}")
    print(f"  LR scheduler:          {TRAINING_CONFIG['lr_scheduler_type']}")
    print(f"  Early stopping:        patience={TRAINING_CONFIG['early_stopping_patience']}")
    precision = "bf16" if TRAINING_CONFIG["bf16"] else "fp16" if TRAINING_CONFIG["fp16"] else "fp32"
    print(f"  Precision:             {precision}")

    # ── Baseline Evaluation ───────────────────────────────────────────────
    print("\nRunning baseline evaluation (before fine-tuning)...")
    baseline_results = trainer.evaluate()
    print("  Baseline metrics:")
    for key, value in baseline_results.items():
        if "rouge" in key or "loss" in key:
            print(f"    {key}: {value:.4f}")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\nStarting LoRA training for {model_info['description']}...")
    print("-" * 70)
    train_start = datetime.now()

    # Resume from last checkpoint if available
    last_checkpoint = None
    if checkpoint_path.exists():
        checkpoints = sorted(checkpoint_path.glob("checkpoint-*"))
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            print(f"  Resuming from checkpoint: {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    train_end = datetime.now()
    train_duration = (train_end - train_start).total_seconds()

    print("-" * 70)
    print(f"Training complete!")
    print(f"  Duration:           {train_duration / 60:.1f} minutes")
    print(f"  Samples/second:     {train_result.metrics['train_samples_per_second']:.2f}")
    print(f"  Final training loss: {train_result.metrics['train_loss']:.4f}")

    # ── Final Evaluation ──────────────────────────────────────────────────
    print("\nRunning final evaluation (best checkpoint)...")
    final_results = trainer.evaluate()
    print("  Final metrics:")
    for key, value in final_results.items():
        if "rouge" in key or "loss" in key:
            print(f"    {key}: {value:.4f}")

    print("\n  Improvement over baseline:")
    for metric in ["rouge1", "rouge2", "rougeL"]:
        baseline = baseline_results.get(f"eval_{metric}", 0)
        final = final_results.get(f"eval_{metric}", 0)
        improvement = final - baseline
        print(f"    {metric}: {baseline:.4f} -> {final:.4f} ({improvement:+.4f})")

    # ── Save Adapter ──────────────────────────────────────────────────────
    print(f"\nSaving LoRA adapter to: {adapter_path}")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    adapter_size = sum(
        os.path.getsize(adapter_path / f)
        for f in os.listdir(adapter_path)
        if os.path.isfile(adapter_path / f)
    ) / 1e6
    print(f"  Adapter size: ~{adapter_size:.1f} MB")

    # ── Save Merged Model ─────────────────────────────────────────────────
    print(f"Merging LoRA into base model and saving to: {merged_path}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

    # ── Save Training Summary ─────────────────────────────────────────────
    training_summary = {
        "model_key": model_key,
        "model_name": hf_name,
        "description": model_info["description"],
        "run_name": run_name,
        "device": device,
        "lora_config": {k: str(v) for k, v in LORA_CONFIG.items()},
        "training_config": {k: str(v) for k, v in TRAINING_CONFIG.items()},
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": round(trainable_pct, 4),
        "baseline_metrics": {k: float(v) for k, v in baseline_results.items() if isinstance(v, (int, float))},
        "final_metrics": {k: float(v) for k, v in final_results.items() if isinstance(v, (int, float))},
        "training_time_seconds": train_duration,
        "training_time_minutes": round(train_duration / 60, 1),
        "train_loss": train_result.metrics["train_loss"],
        "best_checkpoint_step": trainer.state.best_model_checkpoint,
        "total_steps": trainer.state.global_step,
        "timestamp": timestamp,
    }

    summary_path = adapter_path / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"  Training summary saved to: {summary_path}")

    # ── Generate Training Curves ──────────────────────────────────────────
    print("\nGenerating training curves...")
    plot_training_curves(log_dir, model_info["description"])

    # ── Sample Generation ─────────────────────────────────────────────────
    print("\nSample generations (beam search, deterministic):")
    print("-" * 65)
    test_cases = [
        ("reasons_evidence",
         "Climate change is not as serious as scientists claim because "
         "the weather has always changed throughout history."),
        ("clarity",
         "Social media is making teenagers more depressed and we should "
         "ban it for anyone under 18."),
        ("implication_consequences",
         "Artificial intelligence will eventually replace all human jobs "
         "and we need to prepare for universal basic income."),
    ]

    model.eval()
    for q_type, context in test_cases:
        input_text = f"Generate a Socratic question for this context: {q_type}: {context}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=400, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=80, num_beams=4, do_sample=False,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated.replace("[Question]", "").strip()
        print(f"  Type:     {q_type}")
        print(f"  Context:  {context[:80]}...")
        print(f"  Question: {generated}")
        print("-" * 65)

    print(f"\nAll artifacts saved to: {model_output_dir}")
    print("=" * 70)
    return training_summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train a T5-family model with LoRA for Socratic Question Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_model.py --model flan-t5-small
  python scripts/train_model.py --model flan-t5-base
  python scripts/train_model.py --model t5-base
  python scripts/train_model.py --model all    # Train all 3 models sequentially
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Model to train (or 'all' for all 3 models)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (auto-detected if not specified)",
    )
    args = parser.parse_args()

    # Auto-detect project root (script is at socratic-path/scripts/train_model.py)
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = Path(__file__).resolve().parent.parent
    print(f"Project root: {project_root}")

    # Validate paths
    data_dir = project_root / "datasets" / "processed" / "soqg_tokenized"
    if not data_dir.exists():
        print(f"Error: Tokenized dataset not found at {data_dir}")
        print("Please run notebook 02_preprocessing.ipynb first.")
        sys.exit(1)

    if args.model == "all":
        summaries = {}
        for model_key in MODEL_REGISTRY:
            print(f"\n{'#' * 70}")
            print(f"# Training model {list(MODEL_REGISTRY.keys()).index(model_key) + 1}/3: {model_key}")
            print(f"{'#' * 70}")
            summaries[model_key] = train_model(model_key, project_root)

        # Save combined summary
        combined_path = project_root / "models" / "training_comparison.json"
        with open(combined_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nCombined training summary saved to: {combined_path}")
    else:
        train_model(args.model, project_root)


if __name__ == "__main__":
    main()
