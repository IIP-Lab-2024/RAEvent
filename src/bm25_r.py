import json
import os
import torch
from collections import Counter
from math import log
import jieba
from collections import defaultdict
import math


# 假设已有的函数：分词、计算文档长度平均值、计算IDF值
# tokenize(text): 返回分词后的列表
def tokenize(text):
    # 使用精确模式分词
    words = jieba.cut(text, cut_all=False)
    return list(words)

# avgdl: 文档平均长度
def calculate_avgdl(documents):
    total_length = sum(len(tokenize(doc)) for doc in documents)
    avgdl = total_length / len(documents)
    return avgdl
# idf(word): 返回给定单词的IDF值
def compute_idf(documents):
    idf_values = defaultdict(lambda: 0)
    total_documents = len(documents)
    for document in documents.values():
        seen_words = set()
        for word in tokenize(document):
            if word not in seen_words:
                idf_values[word] += 1
                seen_words.add(word)
    for word, df in idf_values.items():
        idf_values[word] = math.log((total_documents + 1) / (df + 1)) + 1  # 加1平滑
    return idf_values

def idf(word):
    return idf_values.get(word, 0)

def bm25(idf, freq, doc_len, avgdl, k1, b):
    """
    计算BM25得分。
    """
    score = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (doc_len / avgdl)))
    return score

def compute_scores(query, documents, avgdl, idf_values, k1, b):
    """
    对给定查询计算所有文档的BM25得分。
    """
    query_words = Counter(tokenize(query))
    scores = {}
    for doc_id, doc_text in documents.items():
        doc_words = tokenize(doc_text)
        doc_len = len(doc_words)
        doc_word_counts = Counter(doc_words)  # 文档中每个词的频率
        score = 0
        for word, freq in query_words.items():
            if word in doc_word_counts:
                word_idf = idf_values.get(word, 0)  # 从预计算的IDF值中获取
                word_freq = doc_word_counts[word]  # 获取文档中词的频率
                score += bm25(word_idf, word_freq, doc_len, avgdl, k1, b)
        scores[doc_id] = score
    return scores

if __name__ == "__main__":
    # 加载查询
    k1 = 1.5
    b = 0.001
    queries = []
    with open('LeCard/query/query.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            queries.append(data)
    results = {}  # 用于存储每个查询的结果
    # 遍历查询，对每个查询检索并排序案件
    for query in queries:
        ridx = query['ridx']
        q_text = query['q']
        
        # 加载对应的待查案例
        candidate_folder = os.path.join('LeCard/candidates', str(ridx))
        documents = {}
        for file_name in os.listdir(candidate_folder):
            file_id = file_name.split('.')[0]
            with open(os.path.join(candidate_folder, file_name), 'r') as f:
                case = json.load(f)
                documents[file_id] = case['ajjbqk']
        sample_text = documents[list(documents.keys())[0]]  # 获取一个示例文档的文本
        # 计算平均文档长度
        doc_lengths = [len(tokenize(doc)) for doc in documents.values()]
        avgdl = sum(doc_lengths) / len(doc_lengths)
        
        idf_values = compute_idf(documents)

        # 计算得分并排序
        scores = compute_scores(q_text, documents, avgdl, idf_values, k1, b)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 输出或保存结果
        # print(f"Query {ridx} results:", sorted_scores[:10])
        # 将排序后的案件ID存储在结果字典中，与查询ID相关联
        results[ridx] = [int(score[0]) for score in sorted_scores]

    with open("SCR-Experiment/result/event/submit1.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 如果需要打印出来查看
    for query_id, sorted_case_ids in results.items():
        print(f"Query {query_id} results:", sorted_case_ids[:10])
