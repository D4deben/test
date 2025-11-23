# PII NER Assignment - Solution Summary

## Overview
This solution implements a token-level Named Entity Recognition (NER) model for detecting PII in STT transcripts using DistilBERT with confidence-based precision optimization.

## Architecture

### Model
- **Base Model**: `distilbert-base-uncased`
- **Task**: Token Classification (13 labels: O + B/I tags for 7 entity types)
- **Rationale**: DistilBERT provides optimal balance between accuracy and latency (target p95 ≤ 20ms)

### Entity Types & PII Mapping
```
PII Entities (pii=true):
- CREDIT_CARD
- PHONE  
- EMAIL
- PERSON_NAME
- DATE

Non-PII Entities (pii=false):
- CITY
- LOCATION
```

## Key Implementation Details

### 1. Dataset (src/dataset.py)
- Uses HuggingFace Fast Tokenizer with `return_offsets_mapping=True`
- Converts character-level entity spans to BIO token labels
- Proper handling of special tokens (set label to -100)
- Robust collate function for batching

### 2. Training (src/train.py)
**Hyperparameters:**
- Epochs: 5 (increased from baseline 3)
- Batch Size: 16 
- Learning Rate: 5e-5
- Warmup Ratio: 0.1 (10% of total steps)
- Weight Decay: 0.01
- Max Length: 256 tokens

**Training Features:**
- AdamW optimizer with linear warmup schedule
- Gradient clipping (max_norm=1.0)
- Dev loss evaluation after each epoch
- Best model selection based on dev loss
- Reproducible with seed=42

### 3. Prediction (src/predict.py)
**Key Feature: Confidence-Based Filtering**
```python
pii_threshold = 0.6  # Tune between 0.5-0.7
```

For each token:
- Get softmax probabilities over all labels
- If `max_probability < pii_threshold`: treat as 'O' (non-entity)
- This biases towards **precision over recall** for PII entities

**BIO Decoding:**
- Convert token predictions to character spans using offset_mapping
- Handle B-/I- transitions properly
- Deduplicate overlapping spans
- Add PII flag based on entity type

## Expected Performance

### Quality Metrics (on dev set)
- **Overall F1**: ~0.75-0.85 (entity-level)
- **PII Precision**: ≥ 0.80 (primary goal)
- **PII Recall**: ~0.65-0.75 (acceptable trade-off)

### Latency (batch_size=1, CPU)
- **p50**: ~8-12 ms
- **p95**: ~15-20 ms
- Well within target of ≤ 20ms

## Commands

### Training
```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --epochs 5 \
  --batch_size 16
```

### Prediction
```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json \
  --pii_threshold 0.6
```

### Evaluation
```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

### Latency Measurement
```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

## Hyperparameter Tuning Guide

If initial results need improvement:

### To Increase PII Precision:
- Increase `pii_threshold` (0.6 → 0.7)
- Trade-off: Lower recall

### To Increase PII Recall:
- Decrease `pii_threshold` (0.6 → 0.5)  
- Trade-off: Lower precision

### To Improve Overall F1:
- Increase `epochs` (5 → 7)
- Increase `batch_size` (16 → 32) if memory allows
- Fine-tune `learning_rate` (try 3e-5 or 7e-5)

### To Reduce Latency:
- Model is already optimized (DistilBERT)
- Could try `distilbert-base-uncased` quantization
- Batch inference if requirements change

## Design Decisions

### 1. Why DistilBERT over BERT?
- 40% faster inference
- 40% smaller model size
- Retains 97% of BERT's performance
- Critical for p95 ≤ 20ms latency target

### 2. Why Confidence Threshold over Regex Validation?
- **ML-first approach**: Aligns with assignment requirements
- **Cleaner**: No complex regex maintenance
- **Flexible**: Single parameter to tune precision/recall trade-off
- **Robust**: Works across noisy STT variations

### 3. Why 5 Epochs?
- Dataset is small (600 train, 150 dev)
- More epochs allow better convergence
- Dev loss tracking prevents overfitting

## Files Modified

1. `src/dataset.py` - Complete rewrite for better BIO encoding
2. `src/train.py` - Enhanced training loop with dev evaluation
3. `src/predict.py` - Simplified with confidence-based filtering

## Limitations & Future Work

1. **Small Dataset**: Only 600 training examples
   - Could benefit from data augmentation
   - Transfer learning from larger NER datasets

2. **No Cross-Validation**: Single train/dev split
   - Could implement k-fold CV for robustness

3. **Fixed Threshold**: Single global confidence threshold
   - Could use per-entity-type thresholds
   - Could learn optimal thresholds from dev set

4. **No Ensemble**: Single model
   - Could ensemble multiple models for better precision

## Conclusion

This solution prioritizes **PII precision** (≥ 0.80) and **low latency** (≤ 20ms p95) as specified in the assignment requirements. The confidence-based filtering approach provides a simple, effective way to tune the precision/recall trade-off without relying on complex rule-based validation.
