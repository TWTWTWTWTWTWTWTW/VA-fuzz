import nltk

# 确保 NLTK 资源已下载
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def swap_nouns_and_verbs(sentence):
    """
    将句子中的名词和动词互换。

    :param sentence: 输入的句子 (str)
    :return: 互换后的句子 (str)
    """
    # 分词和词性标注
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    # 提取名词和动词
    nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
    verbs = [word for word, pos in pos_tags if pos.startswith('VB')]

    # 如果没有名词或动词，返回原句
    if not nouns or not verbs:
        return sentence

    # 将名词和动词分别替换
    swapped_words = []
    noun_idx, verb_idx = 0, 0  # 分别追踪替换到的名词和动词索引

    for word, pos in pos_tags:
        if pos.startswith('NN') and verb_idx < len(verbs):
            # 用动词替换名词
            swapped_words.append(verbs[verb_idx])
            verb_idx += 1
        elif pos.startswith('VB') and noun_idx < len(nouns):
            # 用名词替换动词
            swapped_words.append(nouns[noun_idx])
            noun_idx += 1
        else:
            # 保持其他词语不变
            swapped_words.append(word)

    # 拼接成句子
    return ' '.join(swapped_words)


# # 示例
# sentence = "open the air conditioning"
# swapped_sentence = swap_nouns_and_verbs(sentence)
# print("Original Sentence:", sentence)
# print("Swapped Sentence:", swapped_sentence)
