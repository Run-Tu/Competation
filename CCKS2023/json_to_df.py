import json
import pandas as pd

def json_to_df(json_path:str, train_file=True):
    """
        convert json to dataframe
    """
    content = []

    with open(json_path, 'r', encoding="UTF-8") as f:
        if train_file:
            for json_content in f.readlines():
                json_content = json.loads(json_content)
                content.append([json_content["id"],
                                json_content["cate"],
                                json_content["instruction"],
                                json_content["input"],
                                json_content["output"],
                                json_content["kg"]])
                
            train = pd.DataFrame(content, columns=["id","cate", "instruction", "input", "output", "kg"])
            train.to_csv("./data/train.csv", index=False, encoding="UTF-8")
        else:
            for json_content in f.readlines():
                json_content = json.loads(json_content)
                content.append([json_content["id"],
                                json_content["cate"],
                                json_content["instruction"],
                                json_content["input"]])
            
            valid = pd.DataFrame(content, columns=["id", "cate", "instruction", "input"])
            valid.to_csv("./data/valid.csv", index=False, encoding="UTF-8")


if __name__ == "__main__":
    json_to_df("./data/train.json")
    json_to_df("./data/valid1.json", train_file=False)
