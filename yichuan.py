import json
import random
from tr import *
import numpy as np
from swap_function import swap_nouns_and_verbs
from generate_sentence import generate_sentence_variants
from pareto import *
def initialize_population(real_dataset_path, generated_dataset_path):
    """
    初始化种群，结合真实的和生成的唤醒词。

    参数:
    - real_dataset_path: 真实数据集的文件路径 (dataset.json)
    - generated_dataset_path: 生成的模糊词数据集的文件路径 (generated_dataset.json)

    返回:
    - initial_population: 初始种群，包含真实和模糊的 examples
    - real_examples_backup: 存储真实的 examples 用于相似度计算
    """
    # 读取 dataset.json 和 generated_dataset.json
    with open(real_dataset_path, 'r') as f:
        dataset = json.load(f)

    with open(generated_dataset_path, 'r') as f:
        generated_dataset = json.load(f)

    # 提取数据
    real_examples = {}
    generated_examples = {}

    # 处理 dataset.json
    for item in dataset['intents']:
        intent = item['intent']
        examples = item['examples']
        real_examples[intent] = examples

    # 处理 generated_dataset.json
    for item in generated_dataset['intents']:
        intent = item['intent']
        examples = item['examples']
        generated_examples[intent] = examples

    # 创建初始种群，结合真实和模糊词
    initial_population = {}
    for intent in generated_examples:
        if intent in real_examples:
            initial_population[intent] = real_examples[intent] + generated_examples[intent]
        else:
            initial_population[intent] = real_examples[intent]

    # 保存真实的 examples 以备相似度计算
    real_examples_backup = real_examples.copy()

    return initial_population, real_examples_backup


def crossover(parent1, parent2):
    """
    从两个父代个体中生成一个子代个体，使用单点交叉方法。

    参数:
    - parent1: 第一个父代个体（字符串形式的 example）
    - parent2: 第二个父代个体（字符串形式的 example）

    返回:
    - child: 新生成的子代个体
    """
    # 将句子拆分成单词
    words1 = parent1.split()
    words2 = parent2.split()

    # 检查两个父代的单词长度，避免长度小于1的情况
    if len(words1) < 2 or len(words2) < 2:
        # 如果任何一个父代的长度小于2，直接返回其中一个父代
        return random.choice([parent1, parent2])

    # 随机选择交叉点
    crossover_point = np.random.randint(1, min(len(words1), len(words2)))

    # 生成子代，结合父代的部分单词
    child = ' '.join(words1[:crossover_point] + words2[crossover_point:])

    return child


def genetic_algorithm(initial_population, real_examples_backup, num_generations=10, mutation_rate=0.1):
    # 遍历每个意图，进行独立遗传选择
    new_fuzzy_words = {}
    for intent, examples in initial_population.items():
        selected_examples = similarity_based_selection(intent, examples, real_examples_backup[intent])

        current_population=selected_examples
        for generation in range(num_generations):
            next_population = []
            print("11111111111111111111111111111111111111111111111111111111111111111")

            for i in range(len(current_population)):
                for j in range(i + 1, len(current_population)):
                    child = crossover(current_population[i],current_population[j])

                    txt=tr(child)
                    pre_intent = predict_intent(txt)

                    if pre_intent == intent:
                        next_population.append(child)
                        print(generation,",child:",child,"txt",txt,",pre_intent:",intent,",parent:",current_population[i],current_population[j])

                    children = swap_nouns_and_verbs(current_population[i],current_population[j])
                    for child in children:

                        txt = tr(child)
                        pre_intent = predict_intent(txt)

                        if pre_intent == intent:
                            next_population.append(child)
                            print(generation, ",child:", child, "txt", txt, ",pre_intent:", intent, ",parent:",current_population[i], current_population[j])

            for example in current_population:
                children=generate_sentence_variants(example)
                for child in children:

                    txt = tr(child)
                    pre_intent = predict_intent(txt)

                    if pre_intent == intent:
                        next_population.append(child)
                        print(generation, ",child:", child, "txt", txt, ",pre_intent:", intent, ",parent:",example)

            current_population = next_population
            # print(generation, intent, current_population)
        # 将每个意图的新的模糊词存储到字典中
        new_fuzzy_words[intent] = current_population

        print("------------------------------------------------------------", intent, current_population)

    return new_fuzzy_words


# 使用示例
real_dataset_path = 'dataset1.json'
generated_dataset_path = 'generated_dataset.json'
initial_population, real_examples_backup = initialize_population(real_dataset_path, generated_dataset_path)

# 打印初始种群信息
print("Initial Population:", initial_population)
# print(real_examples_backup)
# print(real_examples_backup['Turn on defrost'])
#
# # 使用示例
# pareto_fronts = pareto_front_selection(initial_population, real_examples_backup)
#
#
# # 打印每个 intent 的帕累托前沿结果
# for intent, front in pareto_fronts.items():
#     print(f"Intent: {intent}")
#     for example, wake_up_rate, similarity in front:
#         print(f"  Example: {example}, Wake-Up Rate: {wake_up_rate}, Similarity: {similarity}")
#

# 进行遗传算法
new_fuzzy_words = genetic_algorithm(initial_population,real_examples_backup)

# 打印生成的新模糊词
for intent, words in new_fuzzy_words.items():
    print(f"Intent: {intent}")
    for word in words:
        print(f"  Fuzzy Word: {word}")