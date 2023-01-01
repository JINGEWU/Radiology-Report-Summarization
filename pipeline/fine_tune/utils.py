# -*- coding: utf-8 -*-
# @Time : 2022/11/01 16:00
# @File : utils.py
# @Author: JW

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from rouge import Rouge
import random
import numpy as np
import os
import json
import codecs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_json_data(file_path):
    data = None
    with codecs.open(file_path, encoding='utf-8') as rf:
        data = json.load(rf)
    return data

def save_json_array(lst, file_path, encoding='utf-8'):
    with codecs.open(file_path, 'w') as wf:
        json.dump(lst, wf)

def covert_data(data):
    output_data = []
    for key, value in data.items():
        input_text = value['content']
        summary_text = value['title']
        output_data.append({'input_text': input_text, 'summary_text':summary_text})
    return output_data

class TrainData(Dataset):
    def __init__(self, data_file, max_dataset_size):
        self.data = self.load_data(data_file, max_dataset_size)
    
    def load_data(self, data_file, max_dataset_size):
        Data = {}
        idx = 0
        data_load = load_json_data(data_file)
        for key, value in data_load.items():
            if idx >= max_dataset_size:
                break
            if data_load[key]!=None:
                Data[idx] = {
                        'title': data_load[key]['impression'],
                        'content': data_load[key]['findings']
                    }
                idx+=1
        return Data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TestData(Dataset):
    def __init__(self, data_file, max_dataset_size):
        self.data = self.load_data(data_file, max_dataset_size)
    
    def load_data(self, data_file, max_dataset_size):
        Data = {}
        idx = 0
        data_load = load_json_data(data_file)
        for data in data_load:
            if idx >= max_dataset_size:
                break
            if data!=None:
                Data[idx] = {
                        'title': data['impression'],
                        'content': data['findings']
                    }
                idx+=1
        return Data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ValidData(Dataset): 
    def __init__(self, data_file1,data_file2, max_dataset_size):
        self.data = self.load_data(data_file1, data_file2,max_dataset_size)
        


    def load_data(self,data_file1, data_file2,max_dataset_size):
        data_1 = {}
        idx1 = 0
        data_load1 = load_json_data(data_file1)
        for key, value in data_load1.items():
            if data_load1[key]!=None:
                data_1[idx1] = {
                        'title': data_load1[key]['impression'],
                        'content': data_load1[key]['findings']
                    }
                idx1+=1
        data_2 = {}
        idx2 = 0
        data_load2 = load_json_data(data_file2)
        for data in data_load2:
            if data!=None:
                data_2[idx2] = {
                        'title': data['impression'],
                        'content': data['findings']
                    }
                idx2+=1
        Data = {}
        idx = 0
        for key, value in data_1.items():
            if idx >= max_dataset_size:
                break
            if data_1[key]!=None:
                Data[idx] = value
                idx+=1
        for key2, value2 in data_2.items():
            if idx >= max_dataset_size:
                break
            if data_2[key2]!=None:
                Data[idx] = value2
                idx+=1
        return Data
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_val_split(train, rate = 0.8):
    idx = 1
    train_num = int(len(list(train.keys()))*rate)
    train_data = {}
    val_data = {}
    for key, value in train.items():
        if idx<=train_num:
            train_data[key] = train[key]
            idx+=1
        else:
            val_data[key-train_num] = train[key]
            idx+=1
    return train_data, val_data


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss, device = device):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model,tokenizer,max_target_length = 32,beam_size = 4, no_repeat_ngram_size = 2, rouge = Rouge(),device = device):
    preds, labels = [], []
    
    model.eval()
    
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length,
                num_beams=beam_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
            ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
            
        preds += [pred.strip() for pred in decoded_preds]
        labels += [label.strip() for label in decoded_labels]

    scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
    result = {key: value['f'] * 100 for key, value in scores.items()}
    result['avg'] = np.mean(list(result.values()))
    print(f"Rouge1: {result['rouge-1']:>0.2f} Rouge2: {result['rouge-2']:>0.2f} RougeL: {result['rouge-l']:>0.2f}\n")
    return result

def test_loop2(dataloader, model,tokenizer,max_target_length = 32,beam_size = 4, no_repeat_ngram_size = 2, rouge = Rouge(),device = device):
    preds, labels = [], []
    
    model.eval()
    
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length,
                num_beams=beam_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
            ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
            
        preds += [pred.strip() for pred in decoded_preds]
        labels += [label.strip() for label in decoded_labels]

    scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
    roughes = {key: value['f'] * 100 for key, value in scores.items()}
    roughes['avg'] = np.mean(list(roughes.values()))
    print(f"Rouge1: {roughes['rouge-1']:>0.2f} Rouge2: {roughes['rouge-2']:>0.2f} RougeL: {roughes['rouge-l']:>0.2f}\n")
    results = {}
    print('saving predicted results...')
    idx = 0
    for pred, label in zip(preds, labels):
        results[idx] = {
            "prediction": pred, 
            "summarization": label
        }
        idx+=1
    return results


