# coding: UTF-8
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from sklearn import metrics
import time
import json
from utils import get_time_dif, gettoken, create_optimizer, getCNtoken
from transformers import AdamW
# 设置微信提醒
import requests

# 日志模块 
TODAY = datetime.date.today()
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y/%m/%d %H:%M:%S %p"
logging.basicConfig(filename=f"./output/{TODAY}.log", level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

# 模型存储路径
import os
Model_Save_Path = f'./output/{TODAY}'
if not os.path.exists(Model_Save_Path):
    os.mkdir(Model_Save_Path)

def train(config, model, train_iter, dev_iter, iter, scheduler=None):
    logging.info(f"CurrentModel is {model.name}")
    start_time = time.time()
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = create_optimizer(model)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = 1e12
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, batches in enumerate(train_iter):
            model.zero_grad()
            # sent, _, labels = batches # _表示的是triple_id训练不用id所以给消音了
            if config.use_CNBERT:
                sent, _, labels = batches
                input_ids, attention_mask, type_ids, pinyin_ids,position_ids = getCNtoken(config, list(sent))
                input_ids, attention_mask, type_ids, pinyin_ids, labels = \
                    input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), pinyin_ids.to(config.device), labels.to(config.device)
                position_ids = position_ids.to(config.device)

                pmi = model(input_ids, pinyin_ids, attention_mask, type_ids)
            else:
                sent, _, labels = batches
                input_ids, attention_mask, type_ids, position_ids = gettoken(config, sent)
                input_ids, attention_mask, type_ids, labels = \
                    input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), labels.to(config.device)
                position_ids = position_ids.to(config.device)

                pmi = model(input_ids, attention_mask, type_ids, position_ids)
            loss = F.binary_cross_entropy(pmi, labels.float(), reduction='sum')
            loss.backward()
            optimizer.step()
            if scheduler:
                    scheduler.step()
            total_batch += 1
            if total_batch % config.test_batch == 1:
                time_dif = get_time_dif(start_time)
                print("test:")
                f1, _, dev_loss, predict, ground = evaluate(config, model, dev_iter, test=False)
                # 微信提醒
                # resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                #      json={
                #          "token": "1ff3eac8fd3b",
                #          "title": "chineseBERT_3L",
                #          "name": f"CCKS_1_{model.name}",
                #          "content": f"Epoch={epoch}, F1={f1}"
                #      })
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Time: {2}'
                logging.info(msg.format(total_batch, loss.item(), time_dif))
                print(msg.format(total_batch, loss.item(), time_dif))
                print("loss", total_batch, loss.item(), dev_loss)
                if dev_loss < dev_best_loss:
                    logging.info(f"save {dev_loss}")
                    print("save", dev_loss)
                    model_path = f"{model.name}_{iter}_Fold.ckpt"
                    torch.save(model.state_dict(), Model_Save_Path + model_path)
                    dev_best_loss = dev_loss
                model.train()


def evaluate(config, model, data_iter, test=True):
    # model.eval()
    loss_total = 0
    predicts, grounds, all_bires = [], [], []
    with torch.no_grad():
       for i, batches in enumerate(data_iter):
            if config.use_CNBERT:
                sent, _, labels = batches
                input_ids, attention_mask, type_ids, pinyin_ids,position_ids = getCNtoken(config, list(sent))
                input_ids, attention_mask, type_ids, pinyin_ids, labels = \
                    input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), pinyin_ids.to(config.device), labels.to(config.device)
                position_ids = position_ids.to(config.device)

                pmi = model(input_ids, pinyin_ids, attention_mask, type_ids)
            else:
                sent, _, labels = batches
                input_ids, attention_mask, type_ids, position_ids = gettoken(config, sent)
                input_ids, attention_mask, type_ids, labels = \
                    input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), labels.to(config.device)
                position_ids = position_ids.to(config.device)

                pmi = model(input_ids, attention_mask, type_ids, position_ids)
            loss = F.binary_cross_entropy(pmi, labels.float(), reduction='sum')
            loss_total += loss.item()
            bires = torch.where(pmi > 0.45, torch.tensor([1]).to(config.device), torch.tensor([0]).to(config.device))
            for b, g, p in zip(bires, labels, pmi):
                all_bires.append(b.item())
                predicts.append(p.item())
                grounds.append(g.item())
            print("test set size:", len(grounds))
            accuracy = metrics.accuracy_score(grounds, all_bires)
            p = metrics.precision_score(grounds, all_bires, zero_division=0)
            r = metrics.recall_score(grounds, all_bires, zero_division=0)
            f1 = metrics.f1_score(grounds, all_bires, zero_division=0)
            logging.info("f1:{},p:{},r,{}, accuracy:{}".format(f1, p, r, accuracy))
            print("f1:{},p:{},r,{}, accuracy:{}".format(f1, p, r, accuracy))

    return f1, pmi, loss_total / len(data_iter), predicts, grounds


def predict(config, model, data_iter):
    # model.eval()
    predicts = []
    with torch.no_grad():
        for i, batches in enumerate(data_iter):
            if config.use_CNBERT:
                sent, triple_id, labels = batches
                input_ids, attention_mask, type_ids, pinyin_ids,position_ids = getCNtoken(config, list(sent))
                input_ids, attention_mask, type_ids, pinyin_ids, labels = \
                    input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), pinyin_ids.to(config.device), labels.to(config.device)
                position_ids = position_ids.to(config.device)

                pmi = model(input_ids, pinyin_ids, attention_mask, type_ids)
            else:
                sent, triple_id, labels = batches
                input_ids, attention_mask, type_ids, position_ids = gettoken(config, sent)
                input_ids, attention_mask, type_ids, labels = \
                    input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), labels.to(config.device)
                position_ids = position_ids.to(config.device)

                pmi = model(input_ids, attention_mask, type_ids, position_ids)

            bires = torch.where(pmi > 0.45, torch.tensor([1]).to(config.device), torch.tensor([0]).to(config.device))
            for b, t in zip(bires, triple_id):
                predicts.append({"salience": b.item(), "triple_id": t})
    with open(config.save_path + f"{model.name}_{TODAY}_result.jsonl", "w") as f:
        for t in predicts:
            f.write(json.dumps(t, ensure_ascii=False)+"\n")


def test(config, model, test_iter, iter):
    # test
    model_path = f"{model.name}_{iter}_Fold.ckpt"
    model.load_state_dict(torch.load(Model_Save_Path + model_path))
    model.eval()
    start_time = time.time()
    predict(config, model, test_iter)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)