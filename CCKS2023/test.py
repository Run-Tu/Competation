import openai
import json

# 设置openAI API密钥
openai.api_key = "YOUR_OWN_KEY"

# 定义输入文本和指令
input_text = "兑换桥（法语：Pont au Change），法国巴黎桥梁，横跨塞纳河以及河流中间的西岱岛。桥梁连接第一区的正义宫和巴黎古监狱以及右岸第四区的沙特莱剧院。"
instruction = "你是一个强大的建筑领域的信息抽取模型，请帮我完成下面的信息抽取工作。\n给你举个例子:已知候选的关系列表是：['位于', '名称由来', '机场服务地区']，从'广岛西飞行场是一个位于日本广岛县广岛市西区的飞行场。现已废止，并改建为直升机机场。'中信息抽取的结果为'(广岛西飞行场,位于,广岛市),(西区,位于,广岛县),(西区,位于,广岛市),(广岛西飞行场,名称由来,广岛市),(广岛西飞行场,机场服务地区,广岛市)'\n    \
              已知候选的关系列表：['临近', '位于']，请你根据关系列表，从以下输入中抽取出可能存在的头实体(Subject)与尾实体(Object)，并给出对应的关系三元组。请按照 (Subject,Relation,Object) 的格式回答。"

# 调用openai.ChatCompletion.create()方法
response = openai.ChatCompletion.create(
    model = 'gpt-4',
    messages = [
                {"role" : "user", 
                 "content" : f"{instruction}\n{input_text}\n"}
              ],
    max_tokens = 1024,
    n = 1 
)

result = response.choices[0]["message"]["content"].strip()
print(result.replace('\n','').replace(' ',''))