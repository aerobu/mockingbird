# Finetuning GPT2 on parody dataset with dropout sweep
import os
import json
import torch
import random
import math
import csv
import time
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AdamW
from tqdm.auto import tqdm

modelSrc = "/home/ubuntu/mockingbird_transfer_s3/mockingbird_model_epoch7_l40_epoch1"
trainPath = "dataset_splits_top15/train.jsonl"
devPath   = "dataset_splits_top15/dev.jsonl"
lrs = [0.0001]
epochs = 1
batchSize = 4
gradAccum = 4
maxLen = 256
warmupFrac = 0.05
dropoutRates = [0.1, 0.2, 0.3]
resultsDir = "results_dropout_sweep"
os.makedirs(resultsDir, exist_ok=True)
logEvery = 500
samplePrompt = "Since you've gone, I've been lost without a trace\nI dream at night, I can only see your face\nI look around, but it's you I can't replace\nI feel so cold, and I long for your embrace\n"


class JsonlTextDataset(Dataset):
    def __init__(self, path):
        self.items = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.items.append(obj["text"])
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return {"text": self.items[idx]}


class GPT2Collator:
    def __init__(self, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.maxLen = max_len

    def __call__(self, batch):
        texts = [item["text"] for item in batch]

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.maxLen,
            return_tensors="pt",
            padding="longest",
        )

        input_ids = encoded["input_ids"]
        attn = encoded["attention_mask"]

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for r in rows:
            writer.writerow(r)


def evaluate_dev(model, dev_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    model.train()
    return total_loss / len(dev_loader)


def generate_sample(model, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def train_one_trial(lrValue, dropoutRate):
    trialDir = os.path.join(resultsDir, f"lr_{lrValue}_dr_{dropoutRate}") 
    modelDir = os.path.join(trialDir, "model")
    os.makedirs(trialDir, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(modelSrc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(
        modelSrc,
        resid_pdrop=dropoutRate,
        embd_pdrop=dropoutRate,
        attn_pdrop=dropoutRate,
    )
    model.resize_token_embeddings(len(tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    trainData = JsonlTextDataset(trainPath)
    devData   = JsonlTextDataset(devPath)
    train_loader = DataLoader(
        trainData,
        batch_size=batchSize,
        shuffle=True,
        collate_fn=GPT2Collator(tokenizer, maxLen),
    )
    dev_loader = DataLoader(
        devData,
        batch_size=batchSize,
        shuffle=False,
        collate_fn=GPT2Collator(tokenizer, maxLen),
    )
    optim = AdamW(model.parameters(), lr=lrValue)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    train_losses = []
    dev_losses = []
    global_step = 0
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        loop = tqdm(train_loader)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / gradAccum
            loss.backward()
            if (global_step + 1) % gradAccum == 0:
                optim.step()
                optim.zero_grad()
            if global_step % logEvery == 0:
                loop.set_postfix(train_loss=loss.item() * gradAccum) 
                train_losses.append((global_step, loss.item() * gradAccum))
            global_step += 1
        dev_loss = evaluate_dev(model, dev_loader, device)
        dev_losses.append((global_step, dev_loss))
        final_train_loss = train_losses[-1][1] if train_losses else 0.0
    write_csv(os.path.join(trialDir, "train_loss.csv"), train_losses)
    write_csv(os.path.join(trialDir, "dev_loss.csv"), dev_losses)
    sample = generate_sample(model, tokenizer, device, samplePrompt)
    with open(os.path.join(trialDir, "sample.txt"), "w") as f:
        f.write(sample)
    model.save_pretrained(modelDir)
    tokenizer.save_pretrained(modelDir)


if __name__ == "__main__":
    best_lr = lrs[0] 
    for dr in dropoutRates:
        train_one_trial(best_lr, dr)
