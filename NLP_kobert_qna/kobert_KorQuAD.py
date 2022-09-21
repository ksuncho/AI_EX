#import package
import os, json
import numpy as np
from tqdm.notebook import tqdm

import argparse
import glob
import logging
import os
import random
import timeit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features
)

# exploring data
##raw data check
train_data = json.load(open('./kobert/data/KorQuAD_v1.0_train.json', 'r'))
dev_data = json.load(open('./kobert/data/KorQuAD_v1.0_dev.json', 'r'))

train_data["data"] = train_data["data"][:150]
with open('./kobert/data/KorQuAD_v1.0_train.json', 'w') as fout:
  json.dump(train_data, fout, indent=2)

dev_data["data"] = dev_data["data"][:50]
with open('./kobert/data/KorQuAD_v1.0_dev.json', 'w') as fout:
  json.dump(dev_data, fout, indent=2)

# train_data["data"]
print("Nb of data: ", len(train_data["data"]))
print()
# print(train_data["data"][0].keys())
# print(len(train_data["data"][0]["paragraphs"]))
print("QA example: ")
for k, v in train_data["data"][0]["paragraphs"][0]["qas"][0].items():
    print(k, v)
print()
print("Context example: ")
train_data["data"][0]["paragraphs"][0]["context"]

from transformers.data.processors.squad import SquadProcessor, SquadV1Processor
from transformers.data.processors.squad import squad_convert_examples_to_features

processor = SquadV1Processor()
train_examples = processor.get_train_examples('./kobert/data', 'KorQuAD_v1.0_train.json')

from transformers import BertModel, BertConfig, AdamW
from kobert.tokenization_kobert import KoBertTokenizer

tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

if __name__ == '__main__':
    features, train_dataset = squad_convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=512,
            doc_stride=128, 
            max_query_length=64,
            is_training=False,
            return_dataset="pt", 
            tqdm_enabled=False)

    len(train_dataset[200])
    print(train_dataset[200])
    # all_input_ids,
    # all_attention_masks,
    # all_token_type_ids,
    # all_start_positions,
    # all_end_positions,
    # all_cls_index,
    # all_p_mask,
    # all_is_impossible,

    " ".join(tokenizer.convert_ids_to_tokens(train_dataset[200][0]))
    print(train_dataset[200][3], train_dataset[200][4])

    #Load Model
    #import torch
    torch.cuda.empty_cache()
    tokenizer.tokenize("가장 위대한 기념물을 짓기로 결정했습니다")
    model = BertForQuestionAnswering.from_pretrained('monologg/kobert')
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    config = BertConfig.from_pretrained('monologg/kobert')

    device = torch.device("cuda" )
    model.to(device) 

    # training 을 위한 파라미터 설정
    params = {
    'max_seq_length': 512,
    'doc_stride': 128,
    'max_query_length': 64,
    'do_lower_case': False,

    'num_train_epochs': 2,
    'per_gpu_train_batch_size': 6,
    'per_gpu_eval_batch_size': 6,
    #'per_gpu_train_batch_size': 8,
    #'per_gpu_eval_batch_size': 8,

    'learning_rate': 5e-05,
    'gradient_accumulation_steps': 2,
    'weight_decay': 0.0,
    'adam_epsilon': 1e-08,
    'max_grad_norm': 1.0,
    'warmup_steps': 0,

    'save_steps': 200,
    'output_dir': 'models',
    'max_answer_length': 30,
    'n_best_size': 20,
    'threads': 1 
    }

    from argparse import Namespace
    args = argparse.Namespace()
    args = vars(args)

    args.update(params)
    args = Namespace(**args)

    #Data Preparation
    train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    #Start Training
    # Optimizer 설정
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    # Scheduler 사용할 경우
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    # t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )
    global_step = 1

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    model.train()
    for _ in range(args.num_train_epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0)
        for step, batch in enumerate(epoch_iterator):      
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)
            loss = outputs[0]
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            global_step += 1

            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                # scheduler.step() 
                model.zero_grad()
                
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                print('Global Step: {} | Train Loss: {:.3f} '.format(global_step, (tr_loss-logging_loss)/args.save_steps))

                logging_loss = tr_loss

                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint{}.pth".format(global_step)))
                tokenizer.save_pretrained(args.output_dir)

    eval_examples = processor.get_dev_examples('./kobert/data', 'KorQuAD_v1.0_dev.json')
    eval_features, eval_dataset = squad_convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False,
            return_dataset="pt",
            tqdm_enabled=False,
            threads=args.threads,
        )

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, 
                             batch_size=args.eval_batch_size)

    all_results = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)
            
        for i, example_index in enumerate(example_indices):
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [(output[i]).detach().cpu().tolist() for output in outputs]
            
            start_logits, end_logits = output
            
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)
    
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    predictions = compute_predictions_logits(
        eval_examples,
        eval_features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file=None,
        output_null_log_odds_file=None,
        verbose_logging=False,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
        tokenizer=tokenizer
    )

    results = squad_evaluate(eval_examples, predictions)
    results
    predictions