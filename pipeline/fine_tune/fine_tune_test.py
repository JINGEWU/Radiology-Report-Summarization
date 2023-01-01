# -*- coding: utf-8 -*-
# @Time : 2022/12/01 16:00
# @File : fine_tune_train.py
# @Author: JW

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from tqdm.auto import tqdm
from tqdm import tqdm
from rouge import Rouge
import numpy as np
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Text Summarization:testing')
parser.add_argument('--model_checkpoint', type=str, default='facebook/bart-base')
parser.add_argument('--model_save', type=str, default='re-train')
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--test_path', type=str, default='./MEDIQA2021_RRS_Test_Set_Full.json')
parser.add_argument('--max_dataset_size', type=int, default=2000)
parser.add_argument('--max_input_length', type=int, default=1024)
parser.add_argument('--max_target_length', type=int, default=128)
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--no_repeat_ngram_size', type=int, default=2)
parser.add_argument('--model_checkpoint_path', type=str, default='./bart_retrain_epoch_2_valid_rouge_22.22_model_weights.bin')
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
    test_data = TestData(args.test_path,args.max_dataset_size)
    print(" test: ", len(list(test_data.data.keys())))


    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    model = model.to(device)

    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collote_fn)

    rouge = Rouge()


    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # load the best model for testing from your local path 
    model.load_state_dict(torch.load(args.model_checkpoint_path))  

    sources, preds, labels = [], [], []
    model.eval()
    with torch.no_grad():
        print('evaluating on test set...')
        for batch_data in tqdm(test_dataloader):
            batch_data = batch_data.to(device)
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=args.max_target_length,
                num_beams=args.beam_size,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            ).cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            label_tokens = batch_data["labels"].cpu().numpy()

            decoded_sources = tokenizer.batch_decode(
                batch_data["input_ids"].cpu().numpy(), 
                skip_special_tokens=True, 
                use_source_tokenizer=True
            )
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

            sources += [source.strip() for source in decoded_sources]
            preds += [pred.strip() for pred in decoded_preds]
            labels += [label.strip() for label in decoded_labels]
        
        scores = rouge.get_scores(
            hyps= preds, 
            refs=labels, 
            avg=True
        )
        rouges = {key: value['f'] * 100 for key, value in scores.items()}
        rouges['avg'] = np.mean(list(rouges.values()))
        print(f"Test Rouge1: {rouges['rouge-1']:>0.2f} Rouge2: {rouges['rouge-2']:>0.2f} RougeL: {rouges['rouge-l']:>0.2f}\n")
        results = {}
        print('saving predicted results...')
        # save the prediction results into files
        idx = 0
        for source, pred, label in zip(sources, preds, labels):
            results[idx] = {
                "document": source, 
                "prediction": pred, 
                "summarization": label
            }
            idx+=1
        save_json_array(results, f'{args.model_save}_test_data_pred.json')

   