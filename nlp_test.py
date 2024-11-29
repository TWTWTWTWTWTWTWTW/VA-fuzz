import json
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

# 读取数据
with open('dataset.json') as f:
    data = json.load(f)

intents = data['intents']

texts = []
labels = []

# 创建意图到ID的映射
intent_to_id = {intent['intent']: i for i, intent in enumerate(intents)}

# 保存意图映射
with open('intent_to_id.json', 'w') as f:
    json.dump(intent_to_id, f, ensure_ascii=False, indent=4)

# 准备训练数据
for intent in intents:
    for example in intent['examples']:
        texts.append(example)
        labels.append(intent_to_id[intent['intent']])

# 分割数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

# 加载Tokenizer和模型
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(intent_to_id))

# 自定义Dataset类
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len)
        return {
            'input_ids': torch.tensor(encodings['input_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask']),
            'labels': torch.tensor(label)
        }

# 创建训练和验证数据集
train_dataset = IntentDataset(train_texts, train_labels, tokenizer, max_len=64)
val_dataset = IntentDataset(val_texts, val_labels, tokenizer, max_len=64)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 创建Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 训练模型
trainer.train()

# 评估模型
results = trainer.evaluate()
print("Evaluation results:", results)

# 保存模型和tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

# 保存意图映射已在前面进行
print("Model, tokenizer, and intent_to_id mapping saved to './model'")
