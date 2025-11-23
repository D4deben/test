import json
import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from labels import ID2LABEL, label_is_pii


def bio_to_spans(
    text,
    offsets,
    label_ids,
    confidences,
    min_confidence: float = 0.6,
):
    """
    Convert BIO token labels + offsets into character-level spans.
    Tokens whose best-label confidence < min_confidence are treated as 'O'
    to bias towards higher precision on PII entities.
    """
    spans = []
    current = None  # current span: dict(start, end, label)

    for (start, end), lid, conf in zip(offsets, label_ids, confidences):
        # Skip special tokens
        if start == 0 and end == 0:
            continue

        tag = ID2LABEL.get(int(lid), "O")

        # Confidence-based filtering for precision
        if conf < min_confidence:
            tag = "O"

        if tag == "O":
            if current is not None:
                spans.append(current)
                current = None
            continue

        # tag like "B-EMAIL" or "I-EMAIL"
        if "-" in tag:
            prefix, ent_label = tag.split("-", 1)
        else:
            prefix, ent_label = "B", tag  # just in case

        if prefix == "B" or current is None or current["label"] != ent_label:
            # start a new span
            if current is not None:
                spans.append(current)
            current = {"start": start, "end": end, "label": ent_label}
        else:
            # continue the current span (I-)
            current["end"] = end

    if current is not None:
        spans.append(current)

    # Deduplicate spans just in case
    unique = {}
    for s in spans:
        key = (s["start"], s["end"], s["label"])
        unique[key] = s
    spans = list(unique.values())

    # Map to final output format with PII flag
    out = []
    for s in spans:
        label = s["label"]
        out.append(
            {
                "start": int(s["start"]),
                "end": int(s["end"]),
                "label": label,
                "pii": bool(label_is_pii(label)),
            }
        )
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    # Confidence threshold for non-O labels (tune if needed)
    ap.add_argument("--pii_threshold", type=float, default=0.6)
    return ap.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            uid = obj["id"]
            text = obj["text"]

            encoding = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            offsets = encoding["offset_mapping"][0].tolist()

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits[0]  # (seq_len, num_labels)
                probs = torch.softmax(logits, dim=-1)

            confidences, label_ids = torch.max(probs, dim=-1)
            label_ids = label_ids.tolist()
            confidences = confidences.tolist()

            spans = bio_to_spans(
                text,
                offsets,
                label_ids,
                confidences,
                min_confidence=args.pii_threshold,
            )
            results[uid] = spans

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
