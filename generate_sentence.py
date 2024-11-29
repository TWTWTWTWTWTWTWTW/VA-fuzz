import nltk
from nltk.corpus import wordnet
import random

# 确保 NLTK 资源已下载
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')


def generate_sentence_variants(sentence):
    """
    每次只替换句子中的一个单词，并生成多个替代结果。
    如果某个单词有多个同义词，则分别生成所有可能的替代句子。

    :param sentence: 输入的句子 (str)
    :return: 替代后的多个句子列表 (list of str)
    """
    # 分词
    words = nltk.word_tokenize(sentence)
    variants = []

    for i, word in enumerate(words):
        # 获取单词的同义词
        synonyms = wordnet.synsets(word)

        if synonyms:
            # 获取所有相关的同义词
            lemmas = [lemma.name() for syn in synonyms for lemma in syn.lemmas()]
            # 去重并排除与原词相同的同义词
            lemmas = list(set(lemmas) - {word})

            for lemma in lemmas:
                # 替换当前单词为同义词
                new_words = words[:]
                new_words[i] = lemma.replace('_', ' ')
                variants.append(' '.join(new_words))

    return variants

#
# # 示例
# sentence = "Turn on the air conditioning"
# variants = generate_sentence_variants(sentence)
#
# # 打印结果
# print("Original Sentence:", sentence)
# print("Generated Variants:")
# for v in variants:
#     print(v)
