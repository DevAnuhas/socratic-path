# Training Configuration Decisions & Issue Log

> Reference document for thesis writing. Chronicles all configuration changes, runtime
> issues, and rationale during model training on EC2 g5.2xlarge (NVIDIA A10G, 24 GB VRAM).

---

## 1. Model Selection

**Requirement:** Train and compare at least 3 models with training loss graphs.

| #   | Model                | HuggingFace ID         | Params | Rationale                                                                                                                     |
| --- | -------------------- | ---------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 1   | FLAN-T5-small + LoRA | `google/flan-t5-small` | 77 M   | Lightweight instruction-tuned baseline; trained first on Apple Silicon (274 min), then retrained on EC2                       |
| 2   | FLAN-T5-base + LoRA  | `google/flan-t5-base`  | 250 M  | Larger instruction-tuned variant; tests whether scale improves Socratic question quality                                      |
| 3   | T5-base + LoRA       | `t5-base`              | 250 M  | Same architecture as #2 but **no instruction tuning**; isolates the effect of FLAN instruction tuning on Socratic questioning |

**Why T5-base as the 3rd model (not GPT-2 or DistilGPT-2):**

- Same parameter count as FLAN-T5-base → fair comparison (controls for model size)
- Same T5 architecture → differences are purely from pre-training strategy (FLAN vs vanilla)
- Directly answers the research question: "Does instruction tuning improve Socratic question generation?"
- Avoids confounding variables (decoder-only vs encoder-decoder, different tokenizers, different model sizes)

---

## 2. LoRA Configuration

All three models use identical LoRA configuration for fair comparison:

```python
LORA_CONFIG = {
    "r": 16,                    # Rank — 16 is standard; 8 may underfit, 32 adds params with diminishing returns
    "lora_alpha": 32,           # Scale factor = alpha/r = 2.0
    "target_modules": [         # All attention + FFN projection layers in T5
        "q", "k", "v", "o",    # Attention: query, key, value, output
        "wi_0", "wi_1", "wo",  # FFN: input gate, input, output
    ],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "SEQ_2_SEQ_LM",
}
```

### Issue: `modules_to_save` Caused 46% Trainable Parameters

**Problem:** An earlier version included `modules_to_save: ["embed_tokens", "lm_head"]`, which
wraps the entire embedding and output projection matrices in PEFT's `ModulesToSaveWrapper`,
making them **fully trainable**. This inflated trainable parameters from the expected ~4% to
**46%** (51.8M / 112.3M for T5-small), effectively defeating the purpose of LoRA.

**Symptoms observed:**

- Eval loss stuck at ~655 (vs expected ~2-5 for a well-initialised model)
- ROUGE-L = 0.0000 across all evaluation steps
- Training loss was decreasing (671 → 555) but from a very high baseline

**Fix:** Removed `modules_to_save` entirely. After fix: **4.05% trainable** (2,555,904 / 63,053,696 for T5-small) — correct for LoRA with r=16 on 7 target modules.

**Thesis note:** The custom `[Question]` token added during preprocessing triggers
`resize_token_embeddings()`, which adds a single row to the embedding matrix. LoRA adapters on
attention/FFN layers handle this fine without `modules_to_save`.

---

## 3. Training Hyperparameters — Evolution

### Initial Configuration (Local, Apple Silicon MPS)

```
learning_rate:        1e-4
batch_size:           8 train / 8 eval
gradient_accum:       2 (effective batch = 16)
warmup_steps:         500 (fixed)
precision:            fp32 (fp16 causes NaN on MPS)
early_stopping:       patience=5, threshold=0.001
```

### EC2 Configuration — Run 2 (5 epochs, batch=32, UNDERFITTING)

```
learning_rate:        1e-4
batch_size:           32 train / 64 eval
gradient_accum:       1 (effective batch = 32)
warmup_fraction:      0.06
precision:            bf16
optimizer:            adamw_torch_fused
num_train_epochs:     5
eval_steps:           2000
early_stopping:       patience=3, threshold=0.0
gradient_checkpointing: T5-small=off, T5-base=on
```

**Problem:** Both models were UNDERFITTING. Evidence:

1. ROUGE-L still improving at final checkpoint (both models)
2. Val loss still decreasing at final checkpoint (both models)
3. Train loss > val loss (negative gap → regularisation working, no overfitting)
4. Cosine LR decayed to ~0 by epoch 4.7, killing learning before convergence
5. Only 13,215 gradient updates vs 52,860 in the MPS run (4x fewer)

**Results (Run 2):** FLAN-T5-small RL=0.143, FLAN-T5-base RL=0.149 — both below paper baseline (0.211).

### Final Configuration (EC2, NVIDIA A10G) — Run 3

```
learning_rate:        1e-4 (LoRA standard; paper's 5e-5 was for FULL fine-tuning)
batch_size:           16 train / 64 eval
gradient_accum:       1 (effective batch = 16)
warmup_fraction:      0.06 (~6% of total steps, computed to warmup_steps at runtime)
precision:            bf16 (Ampere GPU, compute capability ≥ 8)
optimizer:            adamw_torch_fused
num_train_epochs:     10 (cosine schedule needs headroom — LR at epoch 10 ≈ 0)
eval_steps:           2000 (was 500; eval beam search consumed 32% of total time)
early_stopping:       patience=5, threshold=0.0
gradient_checkpointing: T5-small=off, T5-base=on (saves memory for larger models)
```

### Change-by-Change Rationale

#### 3.1 Learning Rate: kept at 1e-4

- **Why:** Hu et al. (2022) LoRA paper uses 1e-4 to 4e-4 for LoRA fine-tuning. The paper's
  5e-5 was for FULL fine-tuning of t5-large (770M params, Appendix D). LoRA adapters need
  higher LR because only ~3% of parameters are trainable — the effective gradient signal
  per update is smaller, so a higher LR compensates.

#### 3.2 Batch Size: 8/8 → 32/64 → 16/64

- **Run 2 (batch=32):** A10G has 24 GB VRAM. With T5-small + LoRA + bf16, GPU memory usage
  was only ~3.7 GB (16% utilisation) at batch=16. Increasing to 32/64 improved utilisation.
- **Run 3 (batch=16):** Reverted to batch=16 because the larger batch size was causing
  underfitting — 2× fewer gradient updates per epoch means the cosine schedule exhausts the
  useful LR range before the model converges. At batch=16 with 10 epochs, we get 52,860
  total steps (matching the MPS run that achieved RL=0.27), versus only 13,215 at batch=32
  with 5 epochs. The A10G still handles batch=16 at ~98% GPU utilisation.

#### 3.3 Gradient Accumulation: 2 → 1

- **Why:** With batch=16 and no accumulation, effective batch = 16 (matching the successful
  MPS run). Accumulation was removed to avoid unnecessary overhead.

#### 3.4 Warmup: 500 steps → 6% ratio

- **Why:** Fixed warmup steps don't adapt when batch size changes. With batch=32,
  500 steps = 500 × 32 = 16,000 samples ≈ 19% of one epoch — far too much warmup.
  A ratio of 0.06 automatically scales: for T5-small at batch=32, this gives ~160 steps
  (5,120 samples ≈ 6% of epoch 1), which is standard.

#### 3.5 Precision: fp32/fp16 → bf16

- **Why:** bf16 has the same exponent range as fp32 (8 bits) but with 8-bit mantissa
  instead of 23-bit. This avoids the loss scaling issues that fp16 can encounter
  (gradient underflow). The A10G is an Ampere GPU (compute capability 8.6) which
  natively supports bf16. Free ~2x memory savings and ~1.5x throughput improvement.
- **MPS note:** Apple Silicon MPS does NOT support bf16 properly; the original MPS training
  used fp32 to avoid NaN losses caused by fp16.

#### 3.6 Early Stopping: patience=5/threshold=0.001 → patience=5/threshold=0.0

- **Why (threshold):** A threshold of 0.001 means any eval improvement < 0.001 ROUGE-L counts
  as "no improvement." For Socratic question generation, improvements are often gradual
  (e.g., +0.0005 per evaluation). The tight threshold was likely the cause of premature
  stopping in prior runs that showed "zero improvement." Setting to 0.0 means ANY improvement
  (however small) resets the patience counter.
- **Why (patience):** With eval_steps=2000, patience=5 gives 10,000 steps of grace
  (~2 epochs at batch=16). This allows the model to recover from learning rate schedule
  dips and noisy eval metrics without stopping prematurely.

#### 3.7 Optimizer: adamw_torch → adamw_torch_fused

- **Why:** `adamw_torch_fused` is a CUDA-optimised implementation that fuses the Adam update
  into a single kernel launch. Provides ~5-10% training speedup at zero cost. Only available
  on CUDA (falls back to `adamw_torch` on CPU).

#### 3.8 Gradient Checkpointing: off → on (CUDA only)

- **Why:** Trades ~20-30% slower training for ~40-60% memory savings by recomputing
  activations during backward pass instead of storing them. Not needed for T5-small
  (fits easily in 24 GB), but essential for T5-base at batch=32 with bf16.
- **Note:** Initially added both as a `TrainingArguments` flag AND a manual
  `model.gradient_checkpointing_enable()` call. The manual call was redundant (Trainer
  manages it via the arg) and was removed.

#### 3.9 TF32: enabled

- **Why:** TF32 (TensorFloat-32) uses the Tensor Cores on Ampere GPUs to perform fp32 matrix
  multiplications at near-bf16 speed with fp32-level precision for the exponent. Enabled via:

  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  ```

  This is a free performance boost with negligible precision loss.

---

## 4. Runtime Issues & Fixes

### 4.1 OverflowError in `compute_metrics`

**Error:** `OverflowError: out of range integral type conversion attempted` at line 360
during evaluation step.

**Root cause:** The `-100` ignore index labels (used by HuggingFace to mask padding tokens
in loss computation) were passed directly to `tokenizer.batch_decode()`. The tokenizer tried
to look up token ID -100 in the vocabulary, which overflows.

**Fix:** Replace -100 values with `pad_token_id` before decoding:

```python
predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
```

### 4.2 Stale Checkpoint Resume Crash (PyTorch CVE-2025-32434)

**Error:** `ValueError: torch < 2.6` when resuming training from a checkpoint saved by an
older PyTorch version.

**Root cause:** The EC2 environment had PyTorch < 2.6 installed. Stale checkpoints from a
failed run triggered the CVE safety check.

**Fix:**

1. Deleted stale checkpoints: `rm -rf models/flan-t5-small-lora/checkpoints/checkpoint-*`
2. Upgraded PyTorch: `/home/ubuntu/socratic-path/.venv/bin/pip install --upgrade torch`
3. Note: system pip was blocked by PEP 668 (`externally-managed-environment`); must use
   the venv pip directly.

### 4.3 fp16 NaN Loss on Apple Silicon MPS

**Error:** Training loss immediately became NaN when using `fp16=True` on Apple Silicon.

**Root cause:** MPS (Metal Performance Shaders) does not properly support fp16 mixed
precision training. The reduced precision causes gradient underflow in the T5 architecture.

**Fix:** Disabled fp16 on MPS; trained in fp32. On EC2 (CUDA), switched to bf16 which
has broader dynamic range and no such issues on Ampere GPUs.

### 4.4 Evaluation ROUGE=0.0000 Bug (torch.cuda.amp.autocast)

**Error:** `evaluate_model.py --model all` produced ROUGE-1/2/L = 0.0000 and BLEU-4 = 0.0000
for all models, but BERTScore F1 = 0.816 (normal range).

**Root cause:** The `generate_predictions()` function wrapped `model.generate()` in
`torch.cuda.amp.autocast()`, which defaults to **fp16** precision. The models were trained
with **bf16**. The fp16 autocast during inference caused numerical degradation of the LoRA
adapter weights, producing degenerate output (every prediction was just a comma `","`).

**Diagnosis:** Inspecting `test_predictions.csv` revealed all 10,573 predictions were the
single character `,`. The model was outputting `[Question],</s>` — the learned start token
followed by garbage. BERTScore still showed ~0.82 because it uses contextual embeddings where
even degenerate short text gets a non-trivial baseline similarity score.

**Fix (two-part):**

1. Removed `torch.cuda.amp.autocast()` from inference — generation runs in native precision.
2. Changed `load_model()` to prefer the merged model (`models/*/merged/`) over base+adapter
   loading. The merged model has LoRA weights baked in, avoiding PEFT adapter loading edge
   cases entirely. Falls back to adapter loading if no merged model exists.

### 4.5 GPU Underutilisation (45% UTL, 16% VRAM)

**Observation (via nvitop):** With T5-small at batch=16 + gradient checkpointing:

- GPU Utilisation: 45%
- GPU Memory: 3.7 GB / 22.5 GB (16%)

**Causes:**

1. Gradient checkpointing adds overhead (recomputing activations) that is unnecessary for
   T5-small — the model fits easily in memory without it.
2. Batch size 16 at the time had gradient checkpointing enabled, adding overhead.

**Fix:** Disabled gradient checkpointing for T5-small. GPU utilisation rose to 98%.
Final batch=16 (Run 3) is chosen for convergence reasons (more gradient updates) rather
than GPU utilisation — the trade-off is correct because the A10G still runs at ~98% UTL
without gradient checkpointing at batch=16.

---

## 5. Dependency Upgrades

### Key Version Changes (pyproject.toml)

| Package               | Old Minimum | New Minimum | Reason                                                 |
| --------------------- | ----------- | ----------- | ------------------------------------------------------ |
| torch                 | >=2.0.0     | >=2.6.0     | CVE-2025-32434 safe checkpoint loading                 |
| transformers          | >=4.36.0    | >=4.47.0    | `processing_class` in Trainer, modern Seq2Seq features |
| peft                  | >=0.7.0     | >=0.14.0    | Modern LoRA config, bug fixes                          |
| accelerate            | >=0.25.0    | >=1.0.0     | Major version with fused optimizer support             |
| datasets              | >=2.16.0    | >=3.0.0     | Modernised API, performance improvements               |
| numpy                 | >=1.24.0    | >=2.0.0     | Required by pandas 3.x                                 |
| sentence-transformers | >=2.2.0     | >=4.0.0     | Modern embedding models for KeyBERT                    |

### PyTorch CUDA Index

The project uses `https://download.pytorch.org/whl/cu124` for CUDA 12.4 wheels.
Originally used `cu121`, but PyTorch 2.6+ dropped cu121 builds — the minimum CUDA
index is now cu124. The cu124 wheels bundle their own CUDA runtime and are
backwards-compatible with the EC2 Deep Learning AMI's NVIDIA driver (535+).

---

## 6. Critical Load Sequence (for Inference / Evaluation)

When loading a trained LoRA adapter, the following sequence MUST be followed:

```python
# 1. Load tokenizer from adapter directory (contains custom [Question] token)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 2. Load base model from HuggingFace
base_model = T5ForConditionalGeneration.from_pretrained(hf_name)

# 3. Resize embeddings BEFORE loading adapter (vocab 32100 → 32128 after custom token)
base_model.resize_token_embeddings(len(tokenizer))

# 4. Load LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
```

**Why this order matters:** The LoRA adapter was saved with the resized embedding dimensions
(32,128 tokens). If step 3 is skipped, the adapter expects shape `(32128, d_model)` but
the base model still has shape `(32100, d_model)`, causing a `RuntimeError: shape mismatch`.

---

## 7. Underfitting Analysis (Run 2 — 5 epochs, batch=32)

### Evidence of Underfitting

Both FLAN-T5-small and FLAN-T5-base showed clear underfitting at the end of Run 2:

| Indicator                 | FLAN-T5-small           | FLAN-T5-base            | Verdict      |
| ------------------------- | ----------------------- | ----------------------- | ------------ |
| ROUGE-L at final eval     | 0.1430 (step 12000)     | 0.1490 (step 10000)     | Still rising |
| Val loss at final eval    | 4.866 (still decreasing)| 5.145 (still decreasing)| Still falling|
| Train–val loss gap        | −0.74 (train > val)     | −1.10 (train > val)     | No overfit   |
| LR at final useful step   | 2.36e-6 at step 12000  | 2.36e-6 at step 12000  | Near-zero    |
| LR effectively dead       | Step 12450 (epoch 4.7)  | Step 12450 (epoch 4.7)  | 0.3 ep wasted|

**Key insight:** The cosine LR schedule decays to zero at `num_train_epochs`. With only
5 epochs, the model had useful learning rate for ~4.7 epochs. But ROUGE-L was still
improving at epoch 4.5 — the LR died before convergence.

### Gradient Update Deficit

| Config       | Effective Batch | Epochs | Steps/Epoch | Total Steps | Useful Steps (LR > 1e-6) |
| ------------ | --------------- | ------ | ----------- | ----------- | ------------------------- |
| MPS (prior)  | 16              | 10     | 5,286       | 52,860      | ~37,000                   |
| EC2 Run 2    | 32              | 5      | 2,643       | 13,215      | ~12,450                   |
| EC2 Run 3    | 16              | 10     | 5,286       | 52,860      | ~37,000                   |

The MPS run had **3× more useful gradient updates** than EC2 Run 2. This directly explains
the ROUGE-L gap (0.27 vs 0.14).

### Fix for Run 3

1. `num_train_epochs`: 5 → 10 (match MPS, give cosine schedule room)
2. `per_device_train_batch_size`: 32 → 16 (2× more updates per epoch)
3. `early_stopping_patience`: 3 → 5 (10,000 steps grace ≈ 2 epochs at batch=16)

Combined effect: 52,860 total steps (4× more than Run 2), matching the MPS run that
achieved ROUGE-L = 0.27.

---

## 8. Evaluation Results (Run 2)

### Corpus-Level Metrics

| Model                       | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | BERTScore F1 |
| --------------------------- | ------- | ------- | ------- | ------ | ------------ |
| Paper: T5-p (prompt-based)  | 0.172   | 0.017   | 0.211   | 0.017  | 0.632        |
| Paper: ProphetNet-p         | 0.178   | 0.018   | 0.208   | 0.018  | 0.632        |
| Paper: GPT-p (prompt-based) | 0.165   | 0.013   | 0.187   | 0.013  | 0.615        |
| Our FLAN-T5-small + LoRA    | 0.150   | 0.029   | 0.145   | 0.012  | **0.849**    |
| Our FLAN-T5-base + LoRA     | 0.162   | 0.030   | 0.152   | 0.015  | **0.864**    |

**Observation:** ROUGE-L is below paper baselines (underfitting — see §7), but BERTScore
massively exceeds paper (0.85–0.86 vs 0.63). This means our models generate semantically
relevant questions but with different surface wording than the references.

### Per-Question-Type ROUGE-L

| Question Type                    | Count | Small RL | Base RL |
| -------------------------------- | ----- | -------- | ------- |
| assumptions                      | 75    | 0.188    | 0.167   |
| reasons_evidence                 | 3,746 | 0.180    | 0.170   |
| clarity                          | 3,593 | 0.150    | 0.149   |
| implication_consequences         | 2,517 | 0.096    | 0.131   |
| alternate_viewpoints_perspectives| 642   | 0.093    | 0.141   |

### Prediction Diversity Problem

| Model          | Unique Predictions | Total | % Unique |
| -------------- | ------------------ | ----- | -------- |
| FLAN-T5-small  | 508                | 10,573| 4.8%     |
| FLAN-T5-base   | 1,253              | 10,573| 11.8%    |

**Root cause:** Beam search (num_beams=4, do_sample=False) is deterministic. With
underfitted models that haven't learned context-specific patterns, the decoder repeatedly
selects the same high-probability generic templates (e.g., "What do you think...",
"What do you mean?"). Training for more epochs should increase diversity as the model
learns finer-grained context conditioning.

**Thesis note:** Low diversity is a symptom of underfitting, not a separate problem.
As the model trains longer, it learns to condition on the input context more specifically,
producing more varied outputs. This will be verified in Run 3.

---

## 9. Paper Baseline Comparison

From Ang et al. (2023, EACL) — SocratiQ dataset, Table 3:

| Model                                  | ROUGE-L   | BERTScore | BLEU-4 |
| -------------------------------------- | --------- | --------- | ------ |
| T5-p (prompt-based)                    | 0.211     | 0.632     | 0.017  |
| ProphetNet-p (prompt)                  | 0.208     | 0.632     | 0.018  |
| GPT-p (prompt-based)                   | 0.187     | 0.615     | 0.013  |
| **Our FLAN-T5-small LoRA (MPS, 10ep)** | **0.273** | —         | —      |

The MPS run (10 epochs, batch=16) exceeded the paper's best ROUGE-L by +0.062 (29%
relative improvement). The EC2 Run 3 config matches the MPS gradient update count and
is expected to recover this performance level.

---

## 10. Hardware & Environment

| Component       | Specification                                            |
| --------------- | -------------------------------------------------------- |
| EC2 Instance    | g5.2xlarge                                               |
| GPU             | NVIDIA A10G (24 GB VRAM, Ampere, compute capability 8.6) |
| vCPUs           | 8                                                        |
| RAM             | 32 GB                                                    |
| AMI             | Deep Learning AMI PyTorch 2.x (Ubuntu 22.04)             |
| CUDA            | 12.4 (wheels via cu124 index)                            |
| Python          | 3.10+ (venv at `/home/ubuntu/socratic-path/.venv/`)      |
| Package Manager | uv (with PyTorch CUDA 12.4 index)                        |

### Training Command

```bash
cd socratic-path
nohup bash scripts/run_all_training.sh > training_output.log 2>&1 &
tail -f training_output.log  # Monitor progress
```

### Abort Training

```bash
pgrep -f "train_model.py" | xargs kill
pgrep -f "run_all_training" | xargs kill
```
