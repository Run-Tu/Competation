import openai
openai.api_key = "YOUR_OWN_KEY"
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
logging.basicConfig(filename=f"../logs/{TODAY}.log", level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def GPT4_one_query_inference(prompt:str):
    """Use GPT4 for one query inference

    Args:
        prompt (str): prompt to GPT

    Returns:
        result: result from GPT
    """
    response = openai.ChatCompletion.create(
        model = 'gpt-4',
        messages = [
                    {"role" : "user", 
                    "content" : prompt}
                ],
        max_tokens = 1024,
        n = 1 
    )
    result = response.choices[0]["message"]["content"].strip()
    result = result.replace('\n','').replace(' ','')

    return result


def GPT4_multiprocess_inference(df, shared_list, result_file_path):
    """Use multiprocess to inference

    Args:
        df (DataFrame): The valid dataframe's chunk
        shared_list (list): list of shared memory between processes
        sub_pid (int): sub process id for save each query result
    """
    for _,row in tqdm(df.iterrows()):
        prompt = "你是一个十分精准的信息抽取模型" + row["instruction"] + '\n' + '"""' + row["input"] + '"""'
        result = GPT4_one_query_inference(prompt)
        row["result"] = result
        with open(result_file_path, 'a') as f:
            f.write(str(row.to_dict()) + '\n')
        current_id = row["id"]
        logging.info(f"当前处理的query的id是{current_id},处理进度{_+1}/{6*len(df)}")
        shared_list.append(row.to_dict())


def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    
    return result_str


def main():
    valid_df = pd.read_csv("../../data/valid.csv", encoding="UTF-8")
    # 使用Manager创建共享内存对象()
    manager = Manager()
    shared_list = manager.list()
    # 按照数量平均分配任务到不同的进程中
    n_process = 8
    chunk_size = len(valid_df) // n_process
    chunked_df = [valid_df[i:i+chunk_size] for i in range(0, len(valid_df), chunk_size)]
    # 启动多个进程并发执行任务
    processes = []
    for chunk in chunked_df:
        result_file_path = f"../../data/temp_file/{generate_random_string(6)}.txt"
        p = Process(target=GPT4_multiprocess_inference, args=(chunk, shared_list, result_file_path))
        processes.append(p)
        p.start()
    # 等待所有的任务完成
    for p in processes:
        p.join()
    # 将共享内存结果转换成dataframe
    train_enhanced_df = pd.DataFrame(shared_list)
    train_enhanced_df.to_csv(f"../../data/result/{TODAY}_train_enhanced_df.csv", index=False, encoding="UTF-8")


if __name__ == "__main__":
    main()

