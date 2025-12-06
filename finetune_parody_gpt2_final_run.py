# Final finetuning script with best hyperparameters for parody production
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
lr = 0.0001
dropout = 0.1
epochs = 6
batchSize = 4
gradAccum = 4
maxLen = 256
warmupFrac = 0.05
resultsDir = "results_final_run"
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
            texts, truncation=True, max_length=self.maxLen,
            return_tensors="pt", padding="longest"
        )
        input_ids = encoded["input_ids"]
        attn = encoded["attention_mask"]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


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
            **inputs, max_new_tokens=120, temperature=0.8, top_p=0.9,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(modelSrc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(
        modelSrc,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
    )
    model.resize_token_embeddings(len(tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    train_loader = DataLoader(
        JsonlTextDataset(trainPath),
        batch_size=batchSize, shuffle=True,
        collate_fn=GPT2Collator(tokenizer, maxLen)
    )
    dev_loader = DataLoader(
        JsonlTextDataset(devPath),
        batch_size=batchSize, shuffle=False,
        collate_fn=GPT2Collator(tokenizer, maxLen)
    )
    optim = AdamW(model.parameters(), lr=lr)
    train_losses = []
    dev_losses = []
    best_dev_loss = float("inf")
    global_step = 0
    model.train()
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
                current_loss = loss.item() * gradAccum
                loop.set_postfix(train_loss=current_loss)
                train_losses.append((global_step, current_loss))
            global_step += 1
        dev_loss = evaluate_dev(model, dev_loader, device)
        dev_ppl = math.exp(dev_loss)
        dev_losses.append((epoch + 1, dev_loss))
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_path = os.path.join(resultsDir, "best_model")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            sample = generate_sample(model, tokenizer, device, samplePrompt)
            with open(os.path.join(resultsDir, "best_sample.txt"), "w") as f:
                f.write(f"Epoch {epoch+1} Sample\n{sample}")
    write_csv(os.path.join(resultsDir, "train_loss.csv"), train_losses)
    write_csv(os.path.join(resultsDir, "dev_loss.csv"), dev_losses)


if __name__ == "__main__":
    main()
