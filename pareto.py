import random

from tr import *
from nlp_use import predict_intent
from word_similar import sentence_similarity


def calculate_wake_up_rate(example, intent, num_trials=10):
    """
    计算唤醒率，通过调用tr和predict_intent函数。

    参数:
    - example: 要测试的唤醒词例子
    - intent: 当前 example 所对应的 intent
    - num_trials: 调用 predict_intent 函数的次数

    返回:
    - 唤醒率（成功唤醒次数/总次数）
    """
    success_count = 0
    for _ in range(num_trials):
        transformed_example = tr(example)  # 调用 tr 函数进行处理
        predicted_intent = predict_intent(transformed_example)  # 调用 predict_intent 函数得到预测的 intent
        if predicted_intent == intent:  # 如果预测的 intent 与当前 example 对应的 intent 相匹配
            success_count += 1

    return success_count / num_trials  # 返回成功唤醒的比例

def calculate_similarity(example, real_examples):
    """
    计算与所有 real_examples 的平均相似度。

    参数:
    - example: 要比较的例子
    - real_examples: 真实 examples 列表

    返回:
    - 平均相似度
    """
    total_similarity = 0
    for real_example in real_examples:
        total_similarity += sentence_similarity(example, real_example)

    return total_similarity / len(real_examples)
    #
    # index=random.randrange(0,len(real_examples))
    # similarity = sentence_similarity(example, real_examples[index])
    # return similarity


def pareto_front_selection(intent,examples, real_examples):
    """
    对每个 intent 的 population 进行帕累托前沿选择。

    参数:
    - population: 初始种群，包含多个 intents 的例子
    - real_examples: 真实 examples 用于相似度计算

    返回:
    - pareto_fronts: 每个 intent 的非支配个体集合
    """

    population_metrics = []  # 每个 intent 的个体性能指标列表

    # 计算每个 example 的唤醒率和相似度
    for example in examples:
        wake_up_rate=1
        # wake_up_rate = calculate_wake_up_rate(example, intent)  # 传递当前 intent
        similarity = calculate_similarity(example, real_examples)  # 计算相似度
        similarity = round(similarity, 1)
        population_metrics.append((example, wake_up_rate, similarity))
        # print(population_metrics)

    # 帕累托前沿选择
    pareto_front = []
    # 帕累托前沿选择
    for i, (example1, wake_up_rate1, similarity1) in enumerate(population_metrics):

        dominated = False
        for j, (example2, wake_up_rate2, similarity2) in enumerate(population_metrics):
            if i != j and wake_up_rate2 >= wake_up_rate1 and similarity2 <= similarity1 and (
                    wake_up_rate2 > wake_up_rate1 or similarity2 < similarity1):
                dominated = True
                break
        if not dominated:
            pareto_front.append((example1, wake_up_rate1, similarity1))

        # 限制每个 intent 选择的数量最多为 10 个
        if len(pareto_front) >= 10:
            break

    return pareto_front


def similarity_based_selection(intent,examples, real_examples):
    """
    对每个 intent 的 population 进行相似度排序选择前十个。

    参数:
    - population: 初始种群，包含多个 intents 的例子
    - real_examples: 真实 examples 用于相似度计算

    返回:
    - selected_examples: 每个 intent 的选择个体集合
    """
    print("run Similarity")
    population_metrics = []  # 每个 intent 的个体性能指标列表
    select_examples=[]
    # 计算每个 example 的唤醒率和相似度
    for example in examples:
        wake_up_rate = 1
        # wake_up_rate = calculate_wake_up_rate(example, intent)  # 传递当前 intent
        similarity = calculate_similarity(example, real_examples)  # 计算相似度
        similarity = round(similarity, 1)
        population_metrics.append((example,similarity))
        select_examples.append(example)
    # 根据相似度排序并选择前十个
    # 按相似度升序排序，最小的相似度优先
    sorted_metrics = sorted(population_metrics, key=lambda x: x[1])

    print("sorted_metrics:",sorted_metrics)
    # 只提取前十个相似度最低的 example
    selected_examples = [example for example, similarity in sorted_metrics[:10]]


    return selected_examples

