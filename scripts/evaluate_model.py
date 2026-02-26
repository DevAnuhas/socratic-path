#!/usr/bin/env python3
"""
Evaluate trained LoRA models on the SocratiQ test set.

Computes: ROUGE-1/2/L, BLEU-4, BERTScore F1, per-question-type ROUGE.
Generates score distribution plots and comparison tables.

Usage:
  python scripts/evaluate_model.py --model flan-t5-small
  python scripts/evaluate_model.py --model flan-t5-base
  python scripts/evaluate_model.py --model t5-base
  python scripts/evaluate_model.py --model all    # Evaluate all & compare

Output:
  evaluation_results/{model_name}/
    ├── evaluation_metrics.json
    ├── test_predictions.csv
    ├── score_distributions.png
    ├── per_type_rouge.png
    └── human_evaluation_samples.csv
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel
import evaluate as hf_evaluate
from rouge_score import rouge_scorer as rs_lib
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Model Registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "flan-t5-small": {
        "hf_name": "google/flan-t5-small",
        "output_dir": "flan-t5-small-lora",
        "label": "FLAN-T5-small + LoRA",
    },
    "flan-t5-base": {
        "hf_name": "google/flan-t5-base",
        "output_dir": "flan-t5-base-lora",
        "label": "FLAN-T5-base + LoRA",
    },
    "t5-base": {
        "hf_name": "t5-base",
        "output_dir": "t5-base-lora",
        "label": "T5-base + LoRA (no instruction tuning)",
    },
}

EVAL_GENERATION_CONFIG = {
    "max_length": 80,
    "num_beams": 4,
    "do_sample": False,
    "early_stopping": True,
}

# SocratiQ paper baselines (Ang et al., EACL 2023, Table 3)
PAPER_BASELINES = {
    "T5-p (paper, prompt-based)": {"rouge1": 0.172, "rouge2": 0.017, "rougeL": 0.211, "bleu4": 0.017, "bertscore": 0.632, "bleurt": 0.426},
    "ProphetNet-p (paper, prompt)": {"rouge1": 0.178, "rouge2": 0.018, "rougeL": 0.208, "bleu4": 0.018, "bertscore": 0.632, "bleurt": 0.425},
    "GPT-p (paper, prompt-based)": {"rouge1": 0.165, "rouge2": 0.013, "rougeL": 0.187, "bleu4": 0.013, "bertscore": 0.615, "bleurt": 0.423},
}


def load_model(model_key: str, project_root: Path, device: str):
    """Load a trained model following the critical load sequence."""
    model_info = MODEL_REGISTRY[model_key]
    adapter_path = project_root / "models" / model_info["output_dir"] / "adapter"

    if not adapter_path.exists():
        raise FileNotFoundError(f"No adapter found at {adapter_path}. Train the model first.")

    print(f"  Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

    print(f"  Loading base model: {model_info['hf_name']}")
    base_model = T5ForConditionalGeneration.from_pretrained(model_info["hf_name"])

    print(f"  Resizing embeddings to {len(tokenizer)}...")
    base_model.resize_token_embeddings(len(tokenizer))

    print(f"  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.to(device)
    model.eval()

    print(f"  Model loaded: {model.num_parameters():,} params on {device}")
    return model, tokenizer


def generate_predictions(model, tokenizer, test_df: pd.DataFrame, device: str,
                         batch_size: int = 32):
    """Generate predictions for the entire test set in batches."""
    all_predictions = []
    all_references = []

    input_texts = test_df["input_text"].tolist()
    ref_col = "original_target" if "original_target" in test_df.columns else "target_text"
    raw_refs = test_df[ref_col].tolist()

    # Pre-clean references once
    for ref in raw_refs:
        ref = ref.replace("[Question]", "").strip() if isinstance(ref, str) else ""
        all_references.append(ref)

    # Process in batches
    num_batches = (len(input_texts) + batch_size - 1) // batch_size
    use_amp = device == "cuda"

    for i in tqdm(range(0, len(input_texts), batch_size),
                  total=num_batches, desc="Generating"):
        batch_texts = input_texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", max_length=400,
            truncation=True, padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model.generate(**inputs, **EVAL_GENERATION_CONFIG)
            else:
                outputs = model.generate(**inputs, **EVAL_GENERATION_CONFIG)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded = [d.replace("[Question]", "").strip() for d in decoded]
        all_predictions.extend(decoded)

    return all_predictions, all_references


def compute_all_metrics(predictions: list, references: list, device: str):
    """Compute ROUGE, BLEU, BERTScore, and BLEURT."""
    clean_preds = [p.replace("[Question]", "").strip() for p in predictions]
    clean_refs = [r.replace("[Question]", "").strip() for r in references]

    # ROUGE (corpus-level)
    print("  Computing ROUGE...")
    rouge_metric = hf_evaluate.load("rouge")
    rouge_results = rouge_metric.compute(
        predictions=clean_preds,
        references=clean_refs,
        use_stemmer=True,
    )

    # BLEU-4 (corpus-level via sacrebleu)
    print("  Computing BLEU-4...")
    bleu_metric = hf_evaluate.load("sacrebleu")
    bleu_result = bleu_metric.compute(
        predictions=clean_preds,
        references=[[ref] for ref in clean_refs],
    )
    bleu4 = bleu_result["score"] / 100.0

    # BERTScore — try deberta-xlarge-mnli first (best correlation with human
    # judgements)
    bertscore_results = None
    F1 = None
    for bs_model in ["microsoft/deberta-xlarge-mnli", "roberta-large"]:
        try:
            print(f"  Computing BERTScore ({bs_model})...")
            P, R, F1 = bert_score_fn(
                clean_preds, clean_refs, lang="en", verbose=True, device=device,
                model_type=bs_model,
            )
            bertscore_results = {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F1.mean().item(),
                "model": bs_model,
            }
            break
        except (OverflowError, Exception) as e:
            print(f"    BERTScore with {bs_model} failed: {e}")
            if bs_model == "roberta-large":
                print("    Skipping BERTScore entirely.")
                F1 = torch.zeros(len(clean_preds))

    # BLEURT — learned metric that the paper reports alongside BERTScore.
    # Requires ~2GB checkpoint download on first run. Falls back gracefully
    # if the package is not installed (pip install bleurt-pytorch).
    bleurt_score = None
    try:
        print("  Computing BLEURT...")
        bleurt_metric = hf_evaluate.load("bleurt", "BLEURT-20")
        bleurt_result = bleurt_metric.compute(
            predictions=clean_preds,
            references=clean_refs,
        )
        bleurt_score = float(np.mean(bleurt_result["scores"]))
        print(f"    BLEURT mean: {bleurt_score:.4f}")
    except Exception as e:
        print(f"    BLEURT skipped ({e}). Install with: pip install bleurt-pytorch")
        print("    This is optional — BERTScore is the primary semantic metric.")

    # Per-sample scores for distribution plots and per-type analysis
    print("  Computing per-sample scores...")
    scorer = rs_lib.RougeScorer(["rougeL"], use_stemmer=True)
    sf = SmoothingFunction().method1
    per_sample_rougeL = []
    per_sample_bleu = []

    for pred, ref in zip(clean_preds, clean_refs):
        score = scorer.score(ref, pred)
        per_sample_rougeL.append(score["rougeL"].fmeasure)
        pred_toks = pred.split()
        ref_toks = ref.split()
        b = sentence_bleu([ref_toks], pred_toks, smoothing_function=sf) if pred_toks else 0.0
        per_sample_bleu.append(b)

    return {
        "rouge": rouge_results,
        "bleu4": bleu4,
        "bertscore": bertscore_results or {"precision": 0, "recall": 0, "f1": 0},
        "bleurt": bleurt_score,
        "per_sample_rougeL": np.array(per_sample_rougeL),
        "per_sample_bleu": np.array(per_sample_bleu),
        "per_sample_bertscore_f1": F1.numpy() if F1 is not None else np.zeros(len(clean_preds)),
    }


def compute_per_type_rouge(test_df: pd.DataFrame):
    """Compute ROUGE broken down by Socratic question type."""
    import re
    rouge_metric = hf_evaluate.load("rouge")

    # Input format: "Generate a Socratic question for this context: {type}: {context}"
    # The type token sits between the instruction prefix and the second colon.
    # Using a regex anchored to the prefix avoids breaking on colons in the context.
    _TYPE_RE = re.compile(
        r"Generate a Socratic question for this context:\s*([^:]+):",
        re.IGNORECASE,
    )

    def extract_question_type(input_text: str) -> str:
        m = _TYPE_RE.search(input_text)
        if m:
            return m.group(1).strip()
        # Fallback: try raw split for any non-standard formats
        parts = input_text.split(":")
        if len(parts) >= 2:
            return parts[-2].strip().split()[-1]  # last word before final colon
        return "unknown"

    test_df = test_df.copy()
    test_df["question_type"] = test_df["input_text"].apply(extract_question_type)

    type_results = {}
    for q_type, group in test_df.groupby("question_type"):
        type_preds = [p.replace("[Question]", "").strip() for p in group["prediction"].tolist()]
        type_refs = [r.replace("[Question]", "").strip() for r in group["reference"].tolist()]
        if not type_preds:
            continue

        res = rouge_metric.compute(predictions=type_preds, references=type_refs, use_stemmer=True)
        type_results[q_type] = {
            "count": len(group),
            "rouge1": res["rouge1"],
            "rouge2": res["rouge2"],
            "rougeL": res["rougeL"],
        }

    return type_results


def plot_score_distributions(metrics: dict, output_dir: Path, model_label: str):
    """Plot per-sample score distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Score Distributions — {model_label}", fontsize=13, fontweight="bold")

    data = [
        (metrics["per_sample_rougeL"], "ROUGE-L", "steelblue"),
        (metrics["per_sample_bleu"], "BLEU-4", "coral"),
        (metrics["per_sample_bertscore_f1"], "BERTScore F1", "seagreen"),
    ]

    for ax, (scores, name, color) in zip(axes, data):
        ax.hist(scores, bins=30, color=color, alpha=0.7, edgecolor="black")
        ax.axvline(scores.mean(), color="red", linestyle="--",
                   label=f"Mean: {scores.mean():.3f}")
        ax.set_title(f"{name} Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=150)
    plt.close()


def plot_per_type_rouge(type_results: dict, output_dir: Path, model_label: str):
    """Plot per-question-type ROUGE breakdown."""
    if not type_results:
        return

    types = sorted(type_results.keys())
    rouge1_scores = [type_results[t]["rouge1"] for t in types]
    rouge2_scores = [type_results[t]["rouge2"] for t in types]
    rougeL_scores = [type_results[t]["rougeL"] for t in types]
    counts = [type_results[t]["count"] for t in types]

    # Shorten type names for display
    short_types = [t.replace("_", " ").title()[:25] for t in types]

    x = np.arange(len(types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, rouge1_scores, width, label="ROUGE-1", color="steelblue")
    bars2 = ax.bar(x, rouge2_scores, width, label="ROUGE-2", color="coral")
    bars3 = ax.bar(x + width, rougeL_scores, width, label="ROUGE-L", color="seagreen")

    ax.set_xlabel("Socratic Question Type")
    ax.set_ylabel("ROUGE Score")
    ax.set_title(f"Per-Question-Type ROUGE Breakdown — {model_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(short_types, rotation=20, ha="right")
    ax.legend()

    # Add count labels above bars
    for i, count in enumerate(counts):
        ax.text(i, max(rouge1_scores[i], rougeL_scores[i]) + 0.01,
                f"n={count}", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(output_dir / "per_type_rouge.png", dpi=150)
    plt.close()


def evaluate_single_model(model_key: str, project_root: Path, device: str):
    """Run full evaluation for a single model."""
    model_info = MODEL_REGISTRY[model_key]
    model_label = model_info["label"]

    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {model_label}")
    print(f"{'=' * 70}")

    # Output directory
    output_dir = project_root / "evaluation_results" / model_info["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(model_key, project_root, device)

    # Load test data
    data_path = project_root / "datasets" / "processed"
    test_df = pd.read_parquet(data_path / "test_formatted.parquet")
    print(f"\n  Test samples: {len(test_df)}")

    # Generate predictions
    print("\n  Generating predictions...")
    predictions, references = generate_predictions(model, tokenizer, test_df, device)
    test_df["prediction"] = predictions
    test_df["reference"] = references

    # Free GPU memory before computing BERTScore
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute metrics
    print("\n  Computing metrics...")
    metrics = compute_all_metrics(predictions, references, device)

    # Per-question-type ROUGE
    print("\n  Computing per-question-type ROUGE...")
    type_results = compute_per_type_rouge(test_df)

    # Print results
    print(f"\n  {'=' * 50}")
    print(f"  RESULTS — {model_label}")
    print(f"  {'=' * 50}")
    print(f"  ROUGE-1:        {metrics['rouge']['rouge1']:.4f}")
    print(f"  ROUGE-2:        {metrics['rouge']['rouge2']:.4f}")
    print(f"  ROUGE-L:        {metrics['rouge']['rougeL']:.4f}")
    print(f"  BLEU-4:         {metrics['bleu4']:.4f}")
    print(f"  BERTScore P:    {metrics['bertscore']['precision']:.4f}")
    print(f"  BERTScore R:    {metrics['bertscore']['recall']:.4f}")
    print(f"  BERTScore F1:   {metrics['bertscore']['f1']:.4f}")
    if metrics['bleurt'] is not None:
        print(f"  BLEURT:         {metrics['bleurt']:.4f}")

    if type_results:
        print(f"\n  Per-Question-Type ROUGE-L:")
        for q_type, res in sorted(type_results.items()):
            print(f"    {q_type:<35} n={res['count']:<6} RL={res['rougeL']:.4f}")

    # Generate plots
    print("\n  Generating plots...")
    plot_score_distributions(metrics, output_dir, model_label)
    plot_per_type_rouge(type_results, output_dir, model_label)

    # Save results
    eval_results = {
        "model": model_key,
        "model_label": model_label,
        "test_samples": len(test_df),
        "rouge": {k: float(v) for k, v in metrics["rouge"].items()},
        "bleu4": float(metrics["bleu4"]),
        "bertscore": metrics["bertscore"],
        "bleurt": metrics["bleurt"],
        "per_type_rouge": type_results,
        "generation_config": EVAL_GENERATION_CONFIG,
    }

    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Attach per-sample scores and save predictions
    test_df["rougeL"] = metrics["per_sample_rougeL"]
    test_df["bleu"] = metrics["per_sample_bleu"]
    test_df["bertscore_f1"] = metrics["per_sample_bertscore_f1"]
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Human evaluation sample (50 items)
    human_eval = test_df.sample(50, random_state=42)[
        ["original_input", "reference", "prediction", "rougeL", "bertscore_f1"]
    ].copy()
    human_eval["fluency"] = None
    human_eval["relevance"] = None
    human_eval["is_socratic"] = None
    human_eval.to_csv(output_dir / "human_evaluation_samples.csv", index=False)

    print(f"\n  Saved results to: {output_dir}")
    return eval_results


def generate_comparison_report(all_results: dict, project_root: Path):
    """Generate a comprehensive comparison report across all models."""
    output_dir = project_root / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 90}")
    print("COMPARATIVE EVALUATION REPORT")
    print(f"{'=' * 90}")

    # Build comparison table
    header = f"{'Model':<45} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BLEU-4':>8} {'BERTScore':>10} {'BLEURT':>8}"
    print(header)
    print("-" * 100)

    # Paper baselines
    for name, scores in PAPER_BASELINES.items():
        bleurt_str = f"{scores['bleurt']:>8.4f}" if 'bleurt' in scores else f"{'N/A':>8}"
        print(f"{name:<45} {scores['rouge1']:>8.4f} {scores['rouge2']:>8.4f} "
              f"{scores['rougeL']:>8.4f} {scores['bleu4']:>8.4f} {scores['bertscore']:>10.4f} {bleurt_str}")

    print("-" * 100)

    # Our models
    rows = []
    for model_key, results in all_results.items():
        rouge = results["rouge"]
        bs = results["bertscore"]
        bleurt_val = results.get("bleurt")
        bleurt_str = f"{bleurt_val:>8.4f}" if bleurt_val is not None else f"{'N/A':>8}"
        row_str = (f"{results['model_label']:<45} {rouge['rouge1']:>8.4f} {rouge['rouge2']:>8.4f} "
                   f"{rouge['rougeL']:>8.4f} {results['bleu4']:>8.4f} {bs['f1']:>10.4f} {bleurt_str}")
        print(row_str)
        rows.append({
            "Model": results["model_label"],
            "ROUGE-1": rouge["rouge1"],
            "ROUGE-2": rouge["rouge2"],
            "ROUGE-L": rouge["rougeL"],
            "BLEU-4": results["bleu4"],
            "BERTScore F1": bs["f1"],
            "BLEURT": bleurt_val,
        })

    print(f"{'=' * 100}")

    # Save comparison
    comparison = {
        "paper_baselines": PAPER_BASELINES,
        "our_models": {k: {
            "rouge": v["rouge"],
            "bleu4": v["bleu4"],
            "bertscore": v["bertscore"],
            "bleurt": v.get("bleurt"),
            "per_type_rouge": v.get("per_type_rouge", {}),
        } for k, v in all_results.items()},
    }
    with open(output_dir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to: {output_dir / 'model_comparison.json'}")

    # Generate comparative per-type chart
    if len(all_results) > 1:
        _plot_comparative_per_type(all_results, output_dir)


def _plot_comparative_per_type(all_results: dict, output_dir: Path):
    """Plot per-question-type ROUGE-L across all models."""
    all_types = set()
    for results in all_results.values():
        all_types.update(results.get("per_type_rouge", {}).keys())

    if not all_types:
        return

    types = sorted(all_types)
    short_types = [t.replace("_", " ").title()[:25] for t in types]

    n_models = len(all_results)
    x = np.arange(len(types))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["steelblue", "coral", "seagreen", "goldenrod"]

    for i, (model_key, results) in enumerate(all_results.items()):
        type_rouge = results.get("per_type_rouge", {})
        scores = [type_rouge.get(t, {}).get("rougeL", 0) for t in types]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, scores, width, label=results["model_label"],
               color=colors[i % len(colors)])

    ax.set_xlabel("Socratic Question Type")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("Per-Question-Type ROUGE-L — Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(short_types, rotation=20, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "comparative_per_type_rouge.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'comparative_per_type_rouge.png'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LoRA models")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--project-root", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Project root: {project_root}")
    print(f"Device: {device}")

    if args.model == "all":
        all_results = {}
        for model_key in MODEL_REGISTRY:
            adapter_path = project_root / "models" / MODEL_REGISTRY[model_key]["output_dir"] / "adapter"
            if not adapter_path.exists():
                print(f"\nSkipping {model_key}: no adapter found at {adapter_path}")
                continue
            all_results[model_key] = evaluate_single_model(model_key, project_root, device)
        if all_results:
            generate_comparison_report(all_results, project_root)
    else:
        evaluate_single_model(args.model, project_root, device)


if __name__ == "__main__":
    main()
