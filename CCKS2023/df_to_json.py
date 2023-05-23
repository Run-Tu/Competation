import json
import datetime
import pandas as pd

TODAY = datetime.date.today()

def df_to_json(df_path:str):
    result_df = pd.read_csv(df_path, encoding="utf-8")
    result_df["lst_result"] = result_df["result"].apply(lambda x: [i.split(",") for i in x.replace("),(", "|").replace("(", "").replace(")", "").split("|")])
    column_lst = ["id", "cate", "instruction"]
    for _, row in result_df.iterrows():
        dict_ = dict()
        for lst in column_lst:
            dict_[lst] = row[lst]
        dict_["output"] = row["result"]
        dict_["kg"] = row["lst_result"]
        with open(f"./data/result/{TODAY}_result.json", 'a', encoding="utf-8-sig") as f:
            f.write(json.dumps(dict_, ensure_ascii=False)+'\n')


if __name__ == "__main__":
    df_to_json("./data/result/2023_0523_result.csv")