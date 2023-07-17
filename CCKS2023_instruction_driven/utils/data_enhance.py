"""
    GPT-4推理模式可用多进程处理,数据增强时多进程报错:
    openai.error.ServiceUnavailableError:The server is overloaded or not ready yet
    
    解决思路:time.sleep()
"""
import openai
openai.api_key = "YOUR_OWN_KEY"
import json
import random
import string
import logging
import datetime
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Manager

#logging
TODAY = datetime.date.today()
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y/%m/%d %H:%M:%S %P"
logging.basicConfig(filename=f"../data_enhance_logs/{TODAY}_data_enhanced.log", level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def GPT4_data_enhance_one_query(data_input:str):
    """Use GPT4 for one query inference

    Args:
        prompt (str): prompt to GPT

    Returns:
        result: result from GPT
    """
    prompt_relation_list = f"""
        作为一名专业的数据标注员，你需要对如下对话生成候选关系列表。现有例子：
        输入数据:"浅草神社位于日本东京都台东区浅草的浅草寺本堂东侧,供奉的是土师真中知、桧田浜成、桧前武成,三位对于浅草寺创立有密切关联的人，每年5月17日都会举行三社祭。现在被指定为重要文化财产。",
        候选关系列表:{['事件', '位于', '名称由来']},
        接下来需要你对下面的数据生成候选关系列表,输入数据:\"{data_input}\"。
        回答时请按照如下格式作答：
        {{ \"候选关系列表\": [\"候选关系1\", \"候选关系2\"] }}
    """

    relation_list_response = openai.ChatCompletion.create(
        model = 'gpt-4',
        messages = [
                    {"role" : "user", 
                    "content" : prompt_relation_list}
                ],
        max_tokens = 1024,
        n = 1 
    )
    relation_list_result = relation_list_response.choices[0]["message"]["content"].strip()
    relation_list_result = relation_list_result.replace('\n','').replace(' ','')

    relation_list = json.loads(relation_list_result)["候选关系列表"]

    instruction = f"""已知候选的关系列表：{relation_list},请你根据关系列表，从以下输入中抽取出可能存在的头实体(Subject)与尾实体(Object)，并给出对应的关系三元组。请按照 (Subject,Relation,Object) 的格式回答。"""
    
    prompt_answer = f"""已知候选的关系列表：{relation_list},请你根据关系列表，从以下输入中抽取出可能存在的头实体(Subject)与尾实体(Object)，并给出对应的关系三元组。请按照 (Subject,Relation,Object) 的格式回答。存在多个三元组时用逗号分隔开\'{data_input}\'"""

    gpt4_label_response = openai.ChatCompletion.create(
        model = 'gpt-4',
        messages = [
                    {"role" : "user", 
                    "content" : prompt_answer}
                ],
        max_tokens = 1024,
        n = 1 
    )
    gpt4_label_result = gpt4_label_response.choices[0]["message"]["content"].strip()
    gpt4_label_result = gpt4_label_result.replace('\n','').replace(' ','')

    return gpt4_label_result, instruction


def GPT4_build_new_data(train_df, shared_list, result_file_path):
    """GPT按领域进行数据增强

    Args:
        df (train_df): train Dataframe
        shared_list (list): 记录数据增强结果
        result_file_path (str): 结果文件路径
    """
    cates = ['建筑', '运输', '人物', '天文对象', '医学', '事件', 
             '作品', '自然科学', '人造物件', '生物','组织', '地理地区']
    for cate in cates:
        for _,row in tqdm(train_df[train_df["cate"]==cate].iterrows()):
            dict_ = {}
            gpt4_result, instruction = GPT4_data_enhance_one_query(row['input'])
            dict_["id"] = row["id"]*2
            dict_["cate"] = row["cate"]
            dict_["instruction"] = instruction
            dict_["output"] = gpt4_result
            dict_["kg"] = "["+gpt4_result.replace('(','[').replace(')',']')+"]"
            with open(result_file_path, 'a') as f:
                f.write(str(dict_) + '\n')
            shared_list.append(dict_)

        break
    return shared_list


def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    
    return result_str


def main():
    train_df = pd.read_csv("../data/train.csv", encoding="UTF-8")
    temp_result_file_path = f"../data/temp_file/{generate_random_string(4)}.txt"
    shared_list = GPT4_build_new_data(train_df, shared_list=[], result_file_path=temp_result_file_path)
    train_enhanced_df = pd.DataFrame(shared_list)
    train_enhanced_df.to_csv(f"../data/result/{TODAY}_train_enhanced_{generate_random_string(4)}_df.csv", index=False, encoding="UTF-8")


if __name__ == "__main__":
    main()