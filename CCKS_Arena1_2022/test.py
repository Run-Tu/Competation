# coding: UTF-8
import time, os
from tokenize import Number
import numpy as np
from train_eval import train, test
import random
from bert import *
import argparse
from tqdm import tqdm
from utils import create_optimizer, get_time_dif, load_dataset, gettoken,set_random_seed
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from transformers import get_cosine_schedule_with_warmup
import torch
import json
import gc
gc.enable()
PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'
NUM_FOLDS = 5

start_time = time.time()
print("Loading data...")

def load_dataset(path):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            line_dict = json.loads(lin)
            subject = line_dict["subject"]
            object = line_dict["object"]
            predicate = line_dict["predicate"]
            triple_id = line_dict["triple_id"]
            raw_sent = SEP.join([subject, predicate, object])
            if "salience" in line_dict.keys():
                salience = line_dict["salience"]
                contents.append([raw_sent, triple_id, int(salience)])
            else:
                contents.append([raw_sent, triple_id, 0])
                
    return contents

train_data_all = np.array(load_dataset('./data/train_triple.jsonl'))
kfold = KFold(n_splits=5, random_state=1024, shuffle=True)
Fold_index = np.array([x for x in range(len(train_data_all))])
for fold, (train_indices, val_indices) in enumerate(kfold.split(Fold_index)):    
        print(f"\nFold {fold + 1}/{NUM_FOLDS}")
            
        set_random_seed(1024 + fold)
        print(train_indices[:20])
        train_dataset = train_data_all[np.array(train_indices)]
        val_dataset = train_data_all[np.array(val_indices)]
        print(train_dataset[12])
        print("len train_dataset is", len(train_dataset))
        print("len val_dataset is", len(val_dataset))