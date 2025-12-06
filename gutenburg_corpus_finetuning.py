# Iteration 1 of finetuning using the Gutenberg poetry corpus
from google.colab import drive
import json, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from torch.utils.data import random_split
from math import exp
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import shutil
import os


class PoetryDataset(Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "labels": torch.tensor(self.input_ids[idx], dtype=torch.long),
        }


def collate_batch(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }


def evaluate_perplexity(model, dataset, max_batches=200):
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)
    total_loss, steps = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, total=min(len(loader), max_batches)):
            if steps >= max_batches:
                break
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
            steps += 1
    avg_loss = total_loss / steps
    ppl = exp(avg_loss)
    return avg_loss, ppl


def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


class PoetryDataset(Dataset):
    def __init__(self, ids, masks):
        self.ids, self.masks = ids, masks
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.masks[idx], dtype=torch.long),
            "labels": torch.tensor(self.ids[idx], dtype=torch.long)
        }


def collate(batch):
    ids = [x["input_ids"] for x in batch]
    masks = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]
    ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer_5.pad_token_id)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": ids, "attention_mask": masks, "labels": labels}


def compute_ppl(model):
    total_loss = 0
    steps = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to("cuda") for k,v in batch.items()}
        with torch.no_grad():
            loss = model(**batch).loss
        total_loss += loss.item()
        steps += 1
    avg_loss = total_loss / steps
    ppl = math.exp(avg_loss)
    return avg_loss, ppl
