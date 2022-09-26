# coding: UTF-8
from torch.utils.data import TensorDataset, DataLoader
import csv
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import random
import numpy as np
import time, os
from datetime import timedelta
from transformers import BertTokenizer
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader
from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig
import json

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'


class OpenKGDataSet(Dataset):
    def __init__(self, contents):
        super().__init__()

        self.raw_sent = contents[0].tolist()
        self.triple_id = contents[1]
        self.label = contents[2]

    def __len__(self):
        return len()

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True


def load_dataset(path, config) -> list:
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


def build_dataset(config):
    train = load_dataset(config.train_path, config)
    dev = load_dataset(config.dev_path, config)
    test = load_dataset(config.test_path, config)

    return train, dev, test


def build_iterator(dataset, config, istrain):
    sent = torch.LongTensor([temp[0] for temp in dataset])
    labels = torch.FloatTensor([temp[1] for temp in dataset])
    train_dataset = TensorDataset(sent, labels)
    if istrain:
        train_loader = DataLoader(dataset, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers,
                                  drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=config.batch_size,
                                  num_workers=config.num_workers, drop_last=True)
    
    return train_loader


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time

    return timedelta(seconds=int(round(time_dif)))


def gettoken(config, sent):
    tokenizer = config.tokenizer
    encode_result = tokenizer(sent, padding='max_length', truncation=True, max_length=config.max_length)
    input_ids = torch.tensor(encode_result['input_ids'])
    attention_mask = torch.tensor(encode_result['attention_mask'])
    type_ids = torch.tensor(encode_result['token_type_ids'])
    position_ids = []
    for j, ids in enumerate(input_ids):
        position_id = list(range(config.max_length))
        position_ids.append(position_id)
    position_ids = torch.tensor(position_ids)

    return input_ids, attention_mask, type_ids, position_ids


def getCNtoken(config, sent):
    """
        使用chineseBERT
    """
    tokenizer = config.CNtokenizer
    encode_result = tokenizer(sent, padding='max_length', truncation=True, max_length=config.max_length,return_tensors='pt')
    input_ids = torch.tensor(encode_result['input_ids'])
    attention_mask = torch.tensor(encode_result['attention_mask'])
    type_ids = torch.tensor(encode_result['token_type_ids'])
    pinyin_ids = torch.tensor(encode_result['pinyin_ids'])

    position_ids = []
    for j, ids in enumerate(input_ids):
        position_id = list(range(config.max_length))
        position_ids.append(position_id)
    position_ids = torch.tensor(position_ids)

    return input_ids, attention_mask, type_ids, pinyin_ids,position_ids


def create_optimizer(model):
    """
        传入参数时可以不用带layer_name,AdamW会有一个默认的group识别号,按照识别号使用自定义参数或默认参数来进行优化
    """
    named_parameters = list(model.named_parameters())# 这里不变成list无法下标切片
    
    uerBert_parameters = named_parameters[:197]    
    dense_parameters = named_parameters[-6:] # 最后4个是全连接层参数
    # only fc one group
    dense_group = [params for (name, params) in dense_parameters]

    parameters = []
    parameters.append({"params": dense_group})

    for layer_num, (name, params) in enumerate(uerBert_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01

        lr = 3e-5

        if layer_num >= 60:        
            lr = 4e-5

        if layer_num >= 180:
            lr = 5e-5

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return AdamW(parameters)

if __name__ == "__main__":
    print("")
