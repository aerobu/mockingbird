# Finetuning GPT2 using hte lyrics dataset:w

import os
import json
import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@dataclass
class TrainConfig:
    tokenizerDir: str = "mockingbird_transfer_s3/mockingbird_model"
    modelDir: str = "mockingbird_transfer_s3/mockingbird_model_epoch6"
    trainJson: str = "mockingbird_transfer_s3/processed_lyrics/lyrics_train_tokenized.json"
    outputBase: str = "mockingbird_transfer_s3"
    outputSubdir: str = "mockingbird_model_epoch7_l40"
    epochs: int = 1
    batchSize: int = 8
    gradAccumSteps: int = 8
    learningRate: float = 3e-5
    weightDecay: float = 0.01
    maxStepsLog: int = 1000
    useBf16: bool = True
    useFp16: bool = False
    numWorkers: int = 4
    pinMemory: bool = True


cfg = TrainConfig()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


tokenizer = GPT2TokenizerFast.from_pretrained(cfg.tokenizerDir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = GPT2LMHeadModel.from_pretrained(cfg.modelDir)
model.to(device)
model.train()


with open(cfg.trainJson, "r") as f:
    data = json.load(f)


inputIdsList = data["input_ids"]
attnMasksList = data["attention_mask"]


class LyricsDataset(Dataset):
    def __init__(self, ids, masks):
        self.ids = ids
        self.masks = masks

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ids = torch.tensor(self.ids[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": ids}


def collateFn(batch):
    ids = [b["input_ids"] for b in batch]
    masks = [b["attention_mask"] for b in batch]
    ids_padded = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(ids, batch_first=True, padding_value=-100)

    return {
        "input_ids": ids_padded,
        "attention_mask": masks_padded,
        "labels": labels_padded,
    }


dataset = LyricsDataset(inputIdsList, attnMasksList)

loader = DataLoader(
    dataset,
    batch_size=cfg.batchSize,
    shuffle=True,
    collate_fn=collateFn,
    num_workers=cfg.numWorkers,
    pin_memory=cfg.pinMemory,
    persistent_workers=True if cfg.numWorkers > 0 else False,
)


noDecay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in noDecay)],
        "weight_decay": cfg.weightDecay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in noDecay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=cfg.learningRate,
)


autocastDtype = None
if cfg.useBf16:
    autocastDtype = torch.bfloat16
elif cfg.useFp16:
    autocastDtype = torch.float16

useAutocast = autocastDtype is not None


scaler = torch.cuda.amp.GradScaler(enabled=cfg.useFp16)


os.makedirs(os.path.join(cfg.outputBase, cfg.outputSubdir), exist_ok=True)


globalStep = 0
for epoch in range(1, cfg.epochs + 1):
    epochLoss = 0.0
    numSteps = 0

    progressBar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progressBar, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        if useAutocast:
            with torch.autocast(device_type="cuda", dtype=autocastDtype):
                outputs = model(**batch)
                loss = outputs.loss
        else:
            outputs = model(**batch)
            loss = outputs.loss

        loss = loss / cfg.gradAccumSteps

        if cfg.useFp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % cfg.gradAccumSteps == 0:
            if cfg.useFp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            globalStep += 1

            currentLoss = loss.item() * cfg.gradAccumSteps
            epochLoss += currentLoss
            numSteps += 1

            if globalStep % cfg.maxStepsLog == 0:
                avgLoss = epochLoss / max(1, numSteps)

        if numSteps > 0:
            progressBar.set_postfix(
                loss=(epochLoss / numSteps),
                gs=globalStep,
            )

    avgEpochLoss = epochLoss / max(1, numSteps)

    saveDir = os.path.join(cfg.outputBase, f"{cfg.outputSubdir}_epoch{epoch}")
    os.makedirs(saveDir, exist_ok=True)

    model.save_pretrained(saveDir)
    tokenizer.save_pretrained(saveDir)
