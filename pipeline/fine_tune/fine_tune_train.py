# -*- coding: utf-8 -*-
# @Time : 2022/12/01 16:00
# @File : fine_tune_test.py
# @Author: JW

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_scheduler
from tqdm.auto import tqdm
from tqdm import tqdm
from rouge import Rouge
import numpy as np
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Text Summarization')
parser.add_argument('--model_checkpoint', type=str, default='facebook/bart-base')
parser.add_argument('--model_save', type=str, default='bart_retrain')
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--epoch_num', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--train_path', type=str, default='./CXR_train.json')
parser.add_argument('--test_path', type=str, default='./MEDIQA2021_RRS_Test_Set_Full.json')
parser.add_argument('--val_path_1', type=str, default='./CXR_test.json')
parser.add_argument('--val_path_2', type=str, default='./indiana_dev.json')
parser.add_argument('--max_dataset_size', type=int, default=2000)
parser.add_argument('--max_input_length', type=int, default=1024)
parser.add_argument('--max_target_length', type=int, default=128)
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--no_repeat_ngram_size', type=int, default=2)
args = parser.parse_args()



def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['content'])
        batch_targets.append(sample['title'])
    batch_data = tokenizer(
        batch_inputs, 
        padding=True, 
        max_length=args.max_input_length,
        truncation=True, 
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets, 
            padding=True, 
            max_length=args.max_target_length,
            truncation=True, 
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data


if __name__ == "__main__":


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    seed_everything(5)
    train_data = TrainData(args.train_path, args.max_dataset_size)
    valid_data = ValidData(args.val_path_1, args.val_path_2,args.max_dataset_size)
    test_data = TestData(args.test_path,args.max_dataset_size)
    print(" train: ", len(list(train_data.data.keys())), "val: ", len(list(valid_data.data.keys())),"test: ",len(list(test_data.data.keys())))

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    model = model.to(device)

    train_dataloader = DataLoader(train_data.data, batch_size=args.batch_size, shuffle=True, collate_fn=collote_fn)
    valid_dataloader = DataLoader(valid_data.data, batch_size=args.batch_size, shuffle=False, collate_fn=collote_fn)
    test_dataloader = DataLoader(test_data.data, batch_size=args.batch_size, shuffle=False, collate_fn=collote_fn)

    rouge = Rouge()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=args.epoch_num*len(train_dataloader),
    )

    total_loss = 0.
    best_avg_rouge = 0.
    for t in range(args.epoch_num):
        print(f"Epoch {t+1}/{args.epoch_num}\n-------------------------------")
        total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss, device = device)
        valid_rouge = test_loop(valid_dataloader, model,tokenizer, max_target_length = args.max_target_length,beam_size = args.beam_size, no_repeat_ngram_size = args.no_repeat_ngram_size, rouge = rouge, device = device)
        rouge_avg = valid_rouge['avg']
        if rouge_avg > best_avg_rouge:
            best_avg_rouge = rouge_avg
            print('saving new weights...\n')
            torch.save(model.state_dict(), f'{args.model_save}_epoch_{t+1}_valid_rouge_{rouge_avg:0.4f}_model_weights.bin')
    print("Done!")
