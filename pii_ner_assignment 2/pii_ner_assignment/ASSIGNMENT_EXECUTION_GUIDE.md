# Assignment Execution Guide (90 Minutes)

This guide provides a step-by-step workflow for completing the assignment efficiently.

## Prerequisites
Ensure `huggingface.co` is accessible to download the DistilBERT model.

## Timeline Breakdown

### Minutes 0-10: Setup & Sanity Check
```bash
# Install dependencies
pip install -r requirements.txt

# Quick sanity check
python -c "from transformers import AutoTokenizer; print('✓ Dependencies OK')"
python -c "import src.labels; print('✓ Labels:', len(src.labels.LABELS), 'labels')"
```

### Minutes 10-20: Baseline Training (Run 1)
```bash
# Train with default parameters
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out
```

**Expected output:**
- 5 epochs × ~38 batches/epoch
- Training time: ~5-8 minutes on CPU
- Dev loss should decrease each epoch
- Best model auto-saved to `out/`

### Minutes 20-25: Baseline Evaluation
```bash
# Generate predictions on dev set
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json

# Evaluate span-level metrics
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

**What to note:**
- Overall Macro-F1 score
- PII-only Precision/Recall/F1
- **Target: PII Precision ≥ 0.80**
- Per-entity breakdown (especially CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE)

### Minutes 25-30: Latency Check
```bash
# Measure inference latency
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

**What to note:**
- p50 latency (median)
- p95 latency (95th percentile)
- **Target: p95 ≤ 20ms**

### Minutes 30-35: Analysis & Decision Point

Review your numbers:

**Scenario A: PII Precision ≥ 0.80 AND p95 ≤ 20ms**
→ ✓ Skip to minute 65 (test set prediction)

**Scenario B: PII Precision < 0.80**
→ Increase threshold or retrain (see Minute 35)

**Scenario C: p95 > 20ms**
→ Model is already optimal (DistilBERT), document trade-off

### Minutes 35-55: Hyperparameter Tuning (If Needed)

#### Option 1: Adjust Confidence Threshold (Quick)
```bash
# Try higher threshold for more precision
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred_thresh07.json \
  --pii_threshold 0.7

# Re-evaluate
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred_thresh07.json
```

If precision improves significantly, use this threshold going forward.

#### Option 2: Retrain with More Epochs (Slower)
```bash
# Only if baseline performance was poor
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out_v2 \
  --epochs 7 \
  --batch_size 16

# Re-evaluate
python src/predict.py --model_dir out_v2 --input data/dev.jsonl --output out_v2/dev_pred.json
python src/eval_span_f1.py --gold data/dev.jsonl --pred out_v2/dev_pred.json
```

### Minutes 55-65: Documentation

Create notes covering:
1. Model architecture: DistilBERT token classifier
2. Hyperparameters used (epochs, batch_size, lr, threshold)
3. Performance metrics:
   - PII Precision/Recall/F1
   - Overall Macro-F1
   - p50 and p95 latency
4. Design choices:
   - Why DistilBERT (speed/accuracy balance)
   - Why confidence threshold (ML-first, precision-focused)
   - BIO encoding via offset_mapping
5. Trade-offs observed (if any)

### Minutes 65-75: Test Set Prediction (Optional)
```bash
# Generate predictions on test set
python src/predict.py \
  --model_dir out \
  --input data/test.jsonl \
  --output out/test_pred.json
```

### Minutes 75-90: Final Review & Loom Recording

**Loom Script (~5 minutes):**

1. **Intro (30s)**: "Hi, I'm presenting my PII NER solution for STT transcripts."

2. **Architecture (1m)**: 
   - "I used DistilBERT token classifier with 13 BIO labels"
   - "Fast tokenizer with offset_mapping for char-level spans"
   - Show `src/` directory structure

3. **Training (1m)**:
   - "5 epochs, batch size 16, AdamW with warmup"
   - "Dev loss tracking, best model selection"
   - Show training command/logs

4. **Inference (1m)**:
   - "Key innovation: confidence threshold filtering"
   - "Bias towards PII precision as required"
   - Show `src/predict.py` key lines

5. **Results (1.5m)**:
   - "PII Precision: [your number]"
   - "PII Recall: [your number]"
   - "Latency p95: [your number]ms"
   - Show eval output

6. **Trade-offs (30s)**:
   - "Precision vs recall balance"
   - "DistilBERT for speed"
   - Any observed limitations

## Quick Troubleshooting

### Model download fails
```bash
# Check internet
ping huggingface.co

# Try with explicit online mode
export TRANSFORMERS_OFFLINE=0
```

### Out of memory during training
```bash
# Reduce batch size
python src/train.py ... --batch_size 8
```

### Low PII precision
```bash
# Increase threshold
python src/predict.py ... --pii_threshold 0.7
```

### High latency
- DistilBERT is already optimized
- Consider this a design constraint
- Document the trade-off

## Deliverables Checklist

- [ ] Code modifications (dataset.py, train.py, predict.py)
- [ ] `out/dev_pred.json` (dev set predictions)
- [ ] `out/test_pred.json` (test set predictions, if time permits)
- [ ] Loom video (~5 min covering architecture, metrics, trade-offs)
- [ ] Notes/documentation explaining approach

## Expected Final Numbers

Based on similar setups:
- **PII Precision**: 0.80-0.90
- **PII Recall**: 0.65-0.80
- **Overall Macro-F1**: 0.75-0.85
- **Latency p50**: 8-12 ms
- **Latency p95**: 15-20 ms

Good luck! 🚀
