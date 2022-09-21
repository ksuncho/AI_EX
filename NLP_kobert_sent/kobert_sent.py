import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from argparse import Namespace
import gluonnlp as nlp
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from transformers import (AutoTokenizer, AutoConfig, BertPreTrainedModel, BertModel,
                          AdamW, get_linear_schedule_with_warmup)

# Raw Data Exploration
raw_data = open('./nsmc/ratings_train.txt', 'r', encoding="UTF-8").readlines()
raw_data = [ele.strip().split("\t") for ele in raw_data]
pd.DataFrame(raw_data).head()

# import gluonnlp as nlp 라이브러리 사용
train_data = nlp.data.TSVDataset("./nsmc/ratings_train.txt", field_indices=[1,2],  num_discard_samples=1)
test_data = nlp.data.TSVDataset("./nsmc/ratings_test.txt", field_indices=[1,2],  num_discard_samples=1)

#print(train_data[:5])

#Data preparation
##Tokenizer
from kobert_tokenizer import KoBERTTokenizer
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

idx = 200
print(train_data[idx][0])
print(tokenizer.encode(train_data[idx][0]))
print(tokenizer.decode(tokenizer.encode(train_data[idx][0])))

