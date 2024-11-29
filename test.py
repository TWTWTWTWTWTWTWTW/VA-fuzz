import nltk
nltk.download('cmudict')


# 加载音标字典
d = nltk.corpus.cmudict.dict()

# 定义一个函数来提取音标
def get_pronunciation(sentence):
    words = sentence.split()
    pronunciations = {}
    for word in words:
        word = word.lower()
        if word in d:
            # 选择第一个发音
            pronunciations[word] = d[word][0]
        else:
            pronunciations[word] = None  # 如果没有找到音标
    return pronunciations

# 示例句子
sentence = "Hello world"
pronunciation = get_pronunciation(sentence)
print(pronunciation)
