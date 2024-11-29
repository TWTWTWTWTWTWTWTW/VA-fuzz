from Similar_nltk import *
import nltk
from nltk.corpus import cmudict

# 加载CMU字典
nltk.download('cmudict')
cmu_dict = cmudict.dict()


# 将单词转换为音素
def word_to_phonemes(word):
    word = word.lower()
    if word in cmu_dict:
        return cmu_dict[word][0]  # 取第一个发音
    return None  # 如果CMU字典中没有该单词

# 定义计算音素相似度的函数
def get_phoneme_similarity(phoneme1, phoneme2):
    # 如果两个音素完全相同，距离为0
    if phoneme1 == phoneme2:
        return 0
    # 检查辅音相似表
    if phoneme1 in similar_consonants_nltk and phoneme2 in similar_consonants_nltk[phoneme1]:
        return 0.5  # 相似辅音替换的代价为0.5
    # 检查元音相似表
    if phoneme1 in similar_vowels_nltk and phoneme2 in similar_vowels_nltk[phoneme1]:
        return 0.5  # 相似元音替换的代价为0.5
    # 如果不相似，代价为1
    return 1

# 计算音素Levenshtein距离
def phonetic_levenshtein(phonemes1, phonemes2):
    len1, len2 = len(phonemes1), len(phonemes2)

    # 创建一个动态规划表
    dp = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]

    # 初始化边界条件
    for i in range(1, len1 + 1):
        dp[i][0] = i
    for j in range(1, len2 + 1):
        dp[0][j] = j

    # 填充动态规划表
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            insert_cost = dp[i][j - 1] + 1  # 插入代价
            delete_cost = dp[i - 1][j] + 1  # 删除代价
            replace_cost = dp[i - 1][j - 1] + get_phoneme_similarity(phonemes1[i - 1], phonemes2[j - 1])  # 替换代价

            dp[i][j] = min(insert_cost, delete_cost, replace_cost)

    # 返回最终编辑距离
    return dp[len1][len2]


# 将句子转换为音素序列
def sentence_to_phonemes(sentence):
    words = sentence.split()
    phoneme_sentence = []
    for word in words:
        phonemes = word_to_phonemes(word)
        if phonemes:
            phoneme_sentence.extend(phonemes)
            phoneme_sentence.append(" ")  # 添加空格分隔单词
        else:
            # 如果找不到该单词的音素表示，将单词作为音素处理（降级方案）
            phoneme_sentence.extend(list(word))
            phoneme_sentence.append(" ")  # 同样在末尾加空格
    return phoneme_sentence[:-1]  # 去除末尾多余的空格

# 计算两句话的相似度（基于音素序列）
def sentence_similarity(sentence1, sentence2):
    phoneme_sentence1 = sentence_to_phonemes(sentence1)
    phoneme_sentence2 = sentence_to_phonemes(sentence2)
    # print(phoneme_sentence1)
    # 计算两句子对应音素序列的Levenshtein距离
    distance = phonetic_levenshtein(phoneme_sentence1, phoneme_sentence2)

    # 正常化距离，使之成为相似度
    max_len = max(len(phoneme_sentence1), len(phoneme_sentence2))
    similarity = 1 - (distance / max_len) if max_len > 0 else 1

    return similarity

# # 测试示例
# sentence1 = "hello world"
# sentence2 = "hallo word"
#
# similarity = sentence_similarity(sentence1, sentence2)
# print(f"Sentence similarity: {similarity}")