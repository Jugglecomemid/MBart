#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 6:13 PM
# @Author  : Charles He
# @File    : mbart_train.py
# @Software: PyCharm

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import torch
import numpy as np
import turibolt as bolt
import torch.utils.data as Data
from transformers import MBartTokenizer, MBartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch import nn


def read_file(src_path, trg_path):
    en = []
    with open(src_path, 'r') as f:
        for line in f:
            en.append(line)

    yue = []
    with open(trg_path, 'r') as f:
        for line in f:
            yue.append(line)

    assert len(en) == len(yue)

    return en, yue


def token_(tokenizer, src, trg, src_lang="en_XX", trg_lang='zh_CN'):
    batch = tokenizer.prepare_seq2seq_batch(src, src_lang=src_lang, tgt_lang=trg_lang, tgt_texts=trg)
    ds = {'input_ids': batch["input_ids"],
          'decoder_input_ids': batch['labels'][:, :-1].contiguous(),
          'labels': batch['labels'][:, 1:].clone()
          }

    torch_dataset = Data.TensorDataset(ds["input_ids"], ds['decoder_input_ids'])

    return torch_dataset


def create_data_loader(df, batch_size):
    return DataLoader(df, batch_size=batch_size, shuffle=True, num_workers=4)


def train_epoch(model, loader, optimizer, scheduler, device):
    model = model.train().to(device)
    losses = []
    for b_x, b_y in loader:
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        labels = b_y.clone().to(device)
        loss = model(input_ids=b_x, decoder_input_ids=b_y, labels=labels)[0]

        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return np.mean(losses)


def evaluate_epoch(model, loader, device):
    model = model.eval().to(device)
    losses = []

    with torch.no_grad():
        for b_x, b_y in loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            labels = b_y.clone().to(device)
            loss = model(input_ids=b_x, decoder_input_ids=b_y, labels=labels)[0]
            losses.append(loss.item())
    return np.mean(losses)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')

    # example_english_phrase = ["I love you", 'you hate i']
    # expected_translation_chinese = ["我中意你", '你憎我']
    print("Loading and processing data")
    en, yue = read_file("../MARIAN/en2yue/train.en", "../MARIAN/en2yue/train.yue")
    val_en, val_yue = read_file("../MARIAN/en2yue/val.en", '../MARIAN/en2yue/val.yue')


    train_dataset = token_(tokenizer, en, yue)
    loader = create_data_loader(train_dataset, 8)

    val_dataset = token_(tokenizer, val_en, val_yue)
    val_loader = create_data_loader(val_dataset, 8)

    EPOCHS = 10
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    resultdir = bolt.ARTIFACT_DIR
    MODEL_SAVE_PATH = os.path.join(resultdir, 'MBart_translation.pt')

    print("Start training")

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 30)
        train_loss = train_epoch(model, loader, optimizer, scheduler, device)
        val_loss = evaluate_epoch(model, val_loader, device)
        print(f'Train_loss: {train_loss} | Val_loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        bolt.send_metrics({
            "Total_train_loss": train_loss,
            "Total_val_loss": val_loss
        })


if __name__ == '__main__':
    main()
