import os
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from model import create_model
from labels import LABELS


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = create_model(args.model_name)
    model.to(device)

    # Datasets
    train_ds = PIIDataset(
        path=args.train,
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_train=True,
    )
    dev_ds = PIIDataset(
        path=args.dev,
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_train=True,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    # Optimizer & scheduler
    num_update_steps_per_epoch = max(1, len(train_dl))
    num_training_steps = args.epochs * num_update_steps_per_epoch

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )

    best_dev_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_dl, desc="Training", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_dl))
        print(f"  Train loss: {avg_train_loss:.4f}")

        # Simple dev loss eval (span F1 is evaluated by separate script)
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(dev_dl, desc="Dev", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                dev_loss += outputs.loss.item()

        dev_loss = dev_loss / max(1, len(dev_dl))
        print(f"  Dev loss:   {dev_loss:.4f}")

        # Save best model by dev loss
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            os.makedirs(args.out_dir, exist_ok=True)
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
            print(f"  New best model saved to {args.out_dir}")

    # In case dev loop never ran (edge case)
    if best_dev_loss == float("inf"):
        os.makedirs(args.out_dir, exist_ok=True)
        model.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)
        print(f"\nSaved model + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
