# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertForMaskedLM
from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig

__call__ = ['Config','Model','Best_Model']

class Config(object):

    """配置参数"""
    def __init__(self, args):
        self.model_name = 'bert'
        self.rank = -1
        self.local_rank = -1
        self.train_path = args.data_dir + '/train_triple.jsonl'  # 训练集
        self.test_path = args.data_dir + '/dev_triple.jsonl'  # 测试集
        self.save_path = args.output_dir  # 模型训练结果
        self.bert_path = args.model_dir
        self.test_batch = args.test_batch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.num_workers = 1
        self.local_rank = -1
        self.num_classes = 2                         # 类别数
        self.num_epochs = args.epochs                                            # epoch数
        self.batch_size = args.batch_size                                           # mini-batch大小
        self.learning_rate = args.learning_rate                                     # 学习率
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path, do_lower_case=True)
        self.CNtokenizer = ChineseBertTokenizerFast.from_pretrained('junnyu/ChineseBERT-large')
        self.hidden_size = args.hidden_size
        self.max_length = args.max_length
        self.use_CNBERT = False
        self.use_Representation = False
        self.sub_max_lenth = 10
        self.obj_max_lenth = 10
        self.pre_max_lenth = 8


class Model(nn.Module):
    name = 'uerBert_base_Model'
    def __init__(self, config, ):
        super(Model, self).__init__()
        self.num_labels = 1
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.dropout)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, type_ids, position_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=type_ids, position_ids=position_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        procuct_ouput = torch.mean(sequence_output, 1)
        x = self.dense_1(procuct_ouput)
        x = torch.sigmoid(x).squeeze(-1)
        return x


class Best_Model(nn.Module):
    name = 'uer_roberta_3l'
    def __init__(self, config, ):
        super(Best_Model, self).__init__()
        self.num_labels = 1
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1)
        )


    def forward(self, input_ids, attention_mask, type_ids, position_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=type_ids, position_ids=position_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        procuct_ouput = torch.mean(sequence_output, 1)
        x = self.fc(procuct_ouput)
        x = torch.sigmoid(x).squeeze(-1)
        return x


class Model_v6(nn.Module):
    """
        1、使用uer_roberta
        2、接连1层卷积层
    """
    name = 'uer_roberta_conv'
    def __init__(self, config, ):
        super(Model_v6, self).__init__()
        self.num_labels = 1
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(config.hidden_size, 256, 2, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(256, 1, 2, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros'),
        )


    def forward(self, input_ids, attention_mask, token_type_ids, position_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        procuct_ouput = torch.mean(sequence_output,1,keepdim=True)
        procuct_ouput = procuct_ouput.transpose(1, 2)
        x = self.conv(procuct_ouput).transpose(1,2)
        x = torch.sigmoid(torch.mean(x,1)).squeeze()

        return x


class Representation_Model(nn.Module):
    """
      1、采用表征学习的思路，将subject、object和predicate分别用BERT进行表征
      2、将表征后的结果concanate接入全连接层
    """
    name = 'uerBERT_Representation_Model_3l'
    def __init__(self, config, ):
        super(Representation_Model, self).__init__()
        self.num_labels = 1
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size*3, 512),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1)
        )


    def forward(self, sub_input_ids, sub_attention_mask, sub_type_ids,
           obj_input_ids, obj_attention_mask, obj_type_ids,
           pre_input_ids, pre_attention_mask, pre_type_ids):
      
        sub_outputs = self.bert(input_ids=sub_input_ids, attention_mask=sub_attention_mask, token_type_ids=sub_type_ids)
        obj_outputs = self.bert(input_ids=obj_input_ids, attention_mask=obj_attention_mask, token_type_ids=obj_type_ids)
        pre_outputs = self.bert(input_ids=pre_input_ids, attention_mask=pre_attention_mask, token_type_ids=pre_type_ids)
        # subject
        sub_output = sub_outputs[0]
        sub_output = self.dropout(sub_output)
        sub_output = torch.mean(sub_output, 1)
        # object_
        obj_output = obj_outputs[0]
        obj_output = self.dropout(obj_output)
        obj_output = torch.mean(obj_output, 1)
        # predicate
        pre_output = pre_outputs[0]
        pre_output = self.dropout(pre_output)
        pre_output = torch.mean(pre_output, 1)

        allInput_output = torch.cat([sub_output, obj_output, pre_output], dim=1)
        x = self.fc(allInput_output)
        x = torch.sigmoid(x).squeeze(-1)
        return x