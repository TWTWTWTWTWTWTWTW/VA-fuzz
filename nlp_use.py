import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# 加载模型和Tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./model')
tokenizer = DistilBertTokenizer.from_pretrained('./model')

# 加载意图到ID的映射
with open('intent_to_id.json', 'r') as f:
    intent_to_id = json.load(f)

# 创建ID到意图的映射
id_to_intent = {v: k for k, v in intent_to_id.items()}


def predict_intent(text):
    # 对输入字符串进行Token化
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=64)

    # 模型推断
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取logits并找到预测的标签ID
    logits = outputs.logits
    predicted_label_id = torch.argmax(logits, dim=1).item()

    # 将预测的ID转换为相应的意图名称
    predicted_intent = id_to_intent[predicted_label_id]

    return predicted_intent



