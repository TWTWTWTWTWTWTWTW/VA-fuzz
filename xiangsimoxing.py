import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import nltk

# 下载 CMU Pronouncing Dictionary
nltk.download('cmudict')
from nltk.corpus import cmudict


# 定义相似辅音和元音的字典
similar_consonants_nltk = {
    'P': ['B', 'M'], 'B': ['P', 'M'], 'M': ['B', 'P'],
    'F': ['V'], 'V': ['F'],
    'TH': ['DH'], 'DH': ['TH'],
    'T': ['D', 'S'], 'D': ['T', 'Z'], 'S': ['Z', 'SH'], 'Z': ['S', 'ZH'], 'N': ['D', 'T'],
    'SH': ['ZH', 'S'], 'ZH': ['SH', 'Z'],
    'K': ['G', 'NG'], 'G': ['K', 'NG'], 'NG': ['K', 'G'],
    'HH': ['']
}

similar_vowels_nltk = {
    'IY1': ['IH1', 'EY1', 'EH1'], 'IH1': ['IY1', 'EH1', 'EY1'],
    'EY1': ['EH1', 'IY1', 'AE1'], 'EH1': ['AE1', 'IH1', 'EY1'],
    'AE1': ['EH1', 'AY1', 'AA1'], 'AA1': ['AE1', 'AO1', 'AH1'],
    'AO1': ['AA1', 'AW1', 'OW1'], 'AH1': ['AA1', 'EH1', 'AO1'],
    'UH1': ['UW1', 'AH1', 'OW1'], 'UW1': ['UH1', 'OW1', 'AO1'],
    'OW1': ['AO1', 'UH1', 'UW1']
}



# 加载音标字典
d = cmudict.dict()

# 假设 phoneme_vectors 包含音标对应的音素特征
phoneme_to_feature = {
    'P': np.random.rand(2), 'B': np.random.rand(2), 'M': np.random.rand(2),
    'F': np.random.rand(2), 'V': np.random.rand(2),
    'TH': np.random.rand(2), 'DH': np.random.rand(2),
    'T': np.random.rand(2), 'D': np.random.rand(2), 'S': np.random.rand(2),
    'Z': np.random.rand(2), 'N': np.random.rand(2),
    'SH': np.random.rand(2), 'ZH': np.random.rand(2),
    'K': np.random.rand(2), 'G': np.random.rand(2), 'NG': np.random.rand(2),
    'HH': np.random.rand(2),
    # 可以继续添加音标及其特征
}


def get_sentence_features(sentence):
    words = sentence.split()
    features = []

    for word in words:
        # 获取单词的音标
        phonemes_list = d.get(word.lower())
        if phonemes_list:
            # 选择音标的第一个发音
            phonemes = phonemes_list[0]
            for phoneme in phonemes:
                if phoneme in phoneme_to_feature:
                    features.append(phoneme_to_feature[phoneme])
                    if phoneme in similar_consonants_nltk:
                        for similar in similar_consonants_nltk[phoneme]:
                            if similar in phoneme_to_feature:
                                features.append(phoneme_to_feature[similar])
                    if phoneme in similar_vowels_nltk:
                        for similar in similar_vowels_nltk[phoneme]:
                            if similar in phoneme_to_feature:
                                features.append(phoneme_to_feature[similar])

    return np.mean(features, axis=0) if features else np.zeros(2)


# 定义模糊句子和非模糊句子
fuzzy_sentences = [
    "Route t. moo the airport",
    "moot re to the airport",
    "Lead tee moo the supermarket",
    "tet directions goo the mall",
    "met directions to the gaulle",
]

non_fuzzy_sentences = [
    "Activate cross traffic alert",
    "Turn on cross traffic warning",
    "Enable cross traffic detection",
    "Start cross traffic alert system",
    "Engage cross traffic assist",
    "Switch on cross traffic alert",
    "Turn on cross traffic feature",
    "Activate the cross traffic warning",
    "Enable cross traffic monitoring",
    "Start the cross traffic alert system"
]

# 获取特征
X_fuzzy = np.array([get_sentence_features(sentence) for sentence in fuzzy_sentences])
print(X_fuzzy)
X_non_fuzzy = np.array([get_sentence_features(sentence) for sentence in non_fuzzy_sentences])

# 创建标签
y_fuzzy = np.ones(len(fuzzy_sentences))
y_non_fuzzy = np.zeros(len(non_fuzzy_sentences))

# 合并特征和标签
X = np.concatenate([X_fuzzy, X_non_fuzzy], axis=0)
y = np.concatenate([y_fuzzy, y_non_fuzzy], axis=0)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练梯度提升分类器
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_classifier.fit(X_train, y_train)

# 保存模型
joblib.dump(gb_classifier, 'gradient_boosting_model.pkl')

# 加载模型
gb_classifier = joblib.load('gradient_boosting_model.pkl')

# 新句子进行预测
new_sentences = [
    "Activate traffic alert",
    "moot reeh to the gaul",
    "Engage traffic assist",
    "Route t. moo the airport",
    "Activate cross traffic alert"
]

# 提取特征
X_new = np.array([get_sentence_features(sentence) for sentence in new_sentences])

# 进行预测
predictions = gb_classifier.predict(X_new)
probabilities = gb_classifier.predict_proba(X_new)  # 获取置信度

# 输出预测结果和置信度
for sentence, prediction, probability in zip(new_sentences, predictions, probabilities):
    label = "Fuzzy Sentence" if prediction == 1 else "Non-Fuzzy Sentence"
    confidence = probability[1] if prediction == 1 else probability[0]  # 置信度
    print(probability)
    print(f"Sentence: '{sentence}' - Predicted Label: {label} - Confidence: {confidence:.2f}")
