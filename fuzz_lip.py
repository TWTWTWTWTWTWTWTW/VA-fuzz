import json
import nltk
from nltk.corpus import cmudict
import random
from nlp_use import *
from tr import  *
from generate_sentence import generate_sentence_variants
from swap_nouns_and_verbs import swap_nouns_and_verbs
import copy  # 用于创建浅拷贝
from Similar_nltk import *
# 确保下载必要的NLTK数据
nltk.download('cmudict')

# 获取发音词典
d = cmudict.dict()

with open('dataset.json', 'r') as f:
    data = json.load(f)



def get_phonemes(word):
    """获取音标，返回第一个发音的音标列表"""
    return d.get(word.lower(), [])[0] if word.lower() in d else None
def phonemes_to_words(phoneme_sequence):
    """将音标序列转换为可能的单词。"""
    possible_words = []

    # 将音标序列转换为字符串
    phoneme_string = " ".join(phoneme_sequence)

    # 遍历字典，查找与给定音标序列匹配的单词
    for word, pronunciations in d.items():
        for pron in pronunciations:
            if " ".join(pron) == phoneme_string:  # 确保音标序列匹配
                possible_words.append(word)

    return possible_words if possible_words else [""]  # 如果没有找到，则返回一个包含空字符串的列表
def swap_consonants(phoneme1, phoneme2):
    """交换两个音标的元音，保持原始 phoneme 不变"""
    swapped = False  # 标记是否发生了交换

    # 确保 phoneme1 和 phoneme2 是有效的音标
    if phoneme1 and phoneme2:
        # 创建两个音标的浅拷贝
        phoneme1_copy = phoneme1.copy() if phoneme1 else phoneme1  # 检查是否为 None
        phoneme2_copy = phoneme2.copy() if phoneme2 else phoneme2  # 检查是否为 None

        for i in range(min(len(phoneme1_copy), len(phoneme2_copy))):
            if phoneme1_copy[i] in similar_vowels_nltk and phoneme2_copy[i] in similar_vowels_nltk:
                # 找到元音相似的音标
                c1 = phoneme1_copy[i] if i > 0 else None
                c2 = phoneme2_copy[i] if i > 0 else None

                if c1 and c2 and c1 != c2:  # 确保存在有效元音并且不同
                    # 交换元音
                    phoneme1_copy[i], phoneme2_copy[i] = c2, c1
                    swapped = True  # 设置标记为 True
                    return phoneme1_copy, phoneme2_copy, swapped

    # 如果没有进行交换，仍然返回拷贝和交换标记
    return phoneme1, phoneme2, swapped
def swap_vowels(phoneme1, phoneme2):
    """交换两个音标的元音前辅音，保持原始 phoneme 不变"""
    swapped = False  # 标记是否发生了交换

    # 确保 phoneme1 和 phoneme2 是有效的音标
    if phoneme1 and phoneme2:
        # 创建两个音标的浅拷贝
        phoneme1_copy = phoneme1.copy() if phoneme1 else phoneme1  # 检查是否为 None
        phoneme2_copy = phoneme2.copy() if phoneme2 else phoneme2  # 检查是否为 None

        for i in range(min(len(phoneme1_copy), len(phoneme2_copy))):
            if phoneme1_copy[i] in similar_vowels_nltk and phoneme2_copy[i] in similar_vowels_nltk:
                # 找到元音相似的音标
                c1 = phoneme1_copy[i - 1] if i > 0 else None
                c2 = phoneme2_copy[i - 1] if i > 0 else None

                if c1 and c2 and c1 != c2:  # 确保存在有效辅音并且不同
                    # 交换辅音
                    phoneme1_copy[i - 1], phoneme2_copy[i - 1] = c2, c1
                    swapped = True  # 设置标记为 True
                    return phoneme1_copy, phoneme2_copy, swapped

    # 如果没有进行交换，仍然返回拷贝和交换标记
    return phoneme1, phoneme2, swapped
def process_swapped_words(words, i, j, swapped_phonemes1, swapped_phonemes2, example, examples):
    # 生成新的单词
    new_words1 = phonemes_to_words(swapped_phonemes1)
    new_words2 = phonemes_to_words(swapped_phonemes2)

    # 保存原始单词
    word_i = words[i]
    word_j = words[j]

    for new_word1 in new_words1:
        if new_word1:  # 只处理非空单词
            words[i] = new_word1
            # 循环生成与第二个新单词的例子
            for new_word2 in new_words2:
                if new_word2:  # 只处理非空单词
                    words[j] = new_word2
                    final_example = " ".join(words)
                    # 调用 tr 和 predict_intent 函数
                    txt = tr(final_example)
                    prediction = predict_intent(txt)

                    if original_prediction != prediction:
                        examples.append({
                            # 用ASR识别出的还是用原来的我们以为的
                            'intent': prediction,
                            'example': final_example
                        })
                        original = " ".join([word_i, word_j])
                        swapped = " ".join([new_word1, new_word2])
                        print(
                            f"Original words: '{original}', Swapped words: '{swapped}',Original example:'{example}' Final example: '{final_example}', ASR output: '{txt}', Prediction: {prediction}")
                        # print(examples)
            words[i] = word_i
            words[j] = word_j
def swap_in_words(words, phoneme_lists):
    """交换单词的辅音并生成新句子，返回生成的所有组合"""
    examples = []
    example = " ".join(words)
    for i in range(len(phoneme_lists)):
        for j in range(i + 1, len(phoneme_lists)):
            # 交换元音前的辅音
            swapped_phonemes1, swapped_phonemes2, swapped = swap_consonants(phoneme_lists[i], phoneme_lists[j])

            if swapped:
                process_swapped_words(words, i, j, swapped_phonemes1, swapped_phonemes2, example, examples)
                print(examples)

            # 交换相似的元音
            swapped_phonemes1, swapped_phonemes2, swapped = swap_vowels(phoneme_lists[i], phoneme_lists[j])

            if swapped:
                process_swapped_words(words, i, j, swapped_phonemes1, swapped_phonemes2, example, examples)
                print(examples)
    return examples

def insert_examples(examples):
    # 将生成的结果加入到最终的数据集中
    for example in examples:
        # 去重并合并同一intent的例子
        intent = example['intent']
        example_text = example['example']

        if intent not in generated_data:
            generated_data[intent] = set()  # 使用集合以避免重复

        if example_text not in seen_examples:
            generated_data[intent].add(example_text)  # 只添加唯一的例子
            seen_examples.add(example_text)  # 记录已处理的例子


def final_generate_sentence(example):
    generate_examples=generate_sentence_variants(example)
    examples=[]
    for generate_example in generate_examples:
        print("   ",generate_example)
        txt = tr(generate_example)
        prediction = predict_intent(txt)

        if original_prediction != prediction:
            examples.append({
                # 用ASR识别出的还是用原来的我们以为的
                'intent': prediction,
                'example': example
            })

            print(f"Original words: Original example:'{example}' Final example: '{generate_example}', ASR output: '{txt}', Prediction: {prediction}")
            print(examples)
    return examples

def final_swap(example):
    swap_example=swap_nouns_and_verbs(example)
    examples=[]

    print("swap   ",swap_example)
    txt = tr(swap_example)
    prediction = predict_intent(txt)

    if original_prediction != prediction:
        examples.append({
            # 用ASR识别出的还是用原来的我们以为的
            'intent': prediction,
            'example': example
        })

        print(f"Original words: Original example:'{example}' Final example: '{swap_example}', ASR output: '{txt}', Prediction: {prediction}")
        print(examples)
    return examples


# 用于存储生成的例子
generated_data = {}
seen_examples = set()  # 用于去重的集合

# 遍历数据集中的所有例子
for intent_data in data['intents']:
    for example in intent_data['examples']:
        print(example)
        original_prediction = predict_intent(example)


        # 分割成单词
        words = example.split()
        # 获取所有单词的音标
        phoneme_lists = [get_phonemes(word) for word in words]
        # 调用封装好的交换函数
        swapped_examples = swap_in_words(words, phoneme_lists)
        insert_examples(swapped_examples)

        # 同义词替换
        generate_examples = final_generate_sentence(example)
        insert_examples(generate_examples)

        # 名词和动词
        swap_examples = final_swap(example)
        insert_examples(swap_examples)


# 将合并后的数据结构转化为所需的格式
final_data = {
    "intents": []
}
for intent, examples in generated_data.items():
    final_data.append({
        'intent': intent,
        'examples': list(examples)  # 转换为列表
    })

# 将生成的数据写入 JSON 文件
output_file = 'generated_dataset.json'
with open(output_file, 'w') as f:
    json.dump(final_data, f, indent=4)

print(f"Data saved to {output_file}")
