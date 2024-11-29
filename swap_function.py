import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# 确保下载了相关的nltk资源
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_nouns_and_verbs(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith('NN')]
    verbs = [word for word, pos in tagged if pos.startswith('VB')]
    return nouns, verbs

def swap_nouns_and_verbs(sentence1, sentence2):
    nouns1, verbs1 = extract_nouns_and_verbs(sentence1)
    nouns2, verbs2 = extract_nouns_and_verbs(sentence2)

    new_sentences = []

    # 确保都有名词可用
    if nouns1 and nouns2:
        # 互换第一个名词
        swapped_sentence1 = sentence1.replace(nouns1[0], nouns2[0])
        swapped_sentence2 = sentence2.replace(nouns2[0], nouns1[0])
        new_sentences.append(swapped_sentence1)
        new_sentences.append(swapped_sentence2)

    # 确保都有动词可用
    if verbs1 and verbs2:
        # 互换第一个动词
        swapped_sentence1 = sentence1.replace(verbs1[0], verbs2[0])
        swapped_sentence2 = sentence2.replace(verbs2[0], verbs1[0])
        new_sentences.append(swapped_sentence1)
        new_sentences.append(swapped_sentence2)

    return new_sentences

# # 示例句子
# sentence1 = "The cat sleeps on the mat."
# sentence2 = "The dog runs in the park."
#
# new_sentences = swap_nouns_and_verbs(sentence1, sentence2)
# for sentence in new_sentences:
#     print(sentence)
