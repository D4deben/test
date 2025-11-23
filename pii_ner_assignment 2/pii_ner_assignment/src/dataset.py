import json
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from labels import LABELS, LABEL2ID


class PIIDataset(Dataset):
    """
    JSONL -> tokenized inputs with BIO labels, using character offsets.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 256,
        is_train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

        self.label_list = LABELS
        self.label2id = LABEL2ID
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        self.examples: List[Dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ex = {
                    "id": obj["id"],
                    "text": obj["text"],
                    # may be missing for test.jsonl
                    "entities": obj.get("entities", []),
                }
                self.examples.append(ex)

    def __len__(self) -> int:
        return len(self.examples)

    def _encode_labels(self, offsets, entities: List[Dict[str, Any]]) -> List[int]:
        """
        Convert character-level entity spans to BIO labels per token.
        offsets: list of (start, end) pairs from tokenizer
        entities: list of {"start": int, "end": int, "label": str}
        """
        labels: List[int] = []

        for start, end in offsets:
            # Special tokens have (0,0) offsets in HF fast tokenizers
            if start == 0 and end == 0:
                labels.append(-100)
                continue

            tag = "O"

            for ent in entities:
                ent_start = ent["start"]
                ent_end = ent["end"]

                # No overlap
                if end <= ent_start or start >= ent_end:
                    continue

                ent_label = ent["label"]
                if start == ent_start:
                    tag = f"B-{ent_label}"
                else:
                    tag = f"I-{ent_label}"
                break  # assume no overlapping entities

            labels.append(self.label2id.get(tag, self.label2id["O"]))

        return labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        text = ex["text"]

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )

        offsets = encoding["offset_mapping"]

        if self.is_train and ex.get("entities") is not None:
            labels = self._encode_labels(offsets, ex["entities"])
        else:
            # For test or unlabeled data
            labels = [-100] * len(offsets)

        return {
            "id": ex["id"],
            "text": text,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "offset_mapping": offsets,
            "labels": labels,
            "pad_token_id": self.pad_token_id,
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads a batch of variable-length sequences to tensors.
    Keeps ids/texts/offset_mapping around for possible analysis.
    """
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq, pad_value):
        return seq + [pad_value] * (max_len - len(seq))

    pad_token_id = batch[0].get("pad_token_id", 0)
    label_pad_id = -100

    input_ids = torch.tensor(
        [pad(x["input_ids"], pad_token_id) for x in batch], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [pad(x["attention_mask"], 0) for x in batch], dtype=torch.long
    )
    labels = torch.tensor(
        [pad(x["labels"], label_pad_id) for x in batch], dtype=torch.long
    )

    # offset_mapping is kept as list-of-lists (not used in training loss)
    offset_mapping = [pad(x["offset_mapping"], (0, 0)) for x in batch]

    out: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ids": [x["id"] for x in batch],
        "texts": [x["text"] for x in batch],
        "offset_mapping": offset_mapping,
    }
    return out
