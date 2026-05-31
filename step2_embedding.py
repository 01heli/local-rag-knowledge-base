import math
from openai import OpenAI

client = OpenAI(
    base_url = "http://127.0.0.1:1234/v1",
    api_key = "not-needed"
)

response = client.embeddings.create(
    model="text-embedding-bge-m3",
    input="苹果是一种水果"
)

vec = response.data[0].embedding


def dot_product(a, b):
    """向量 a 和 b 的点积"""
    return sum(a[i] * b[i] for i in range(len(a)))


def magnitude(a):
    """向量 a 的模长（长度）"""
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a, b):
    """返回 -1 到 1 之间的值，1 表示方向完全相同"""
    return dot_product(a, b) / (magnitude(a) * magnitude(b))

sim = cosine_similarity(vec, vec)

# print(f"向量长度（维度）: {len(vec)}")
# print(f"前 5 个元素: {vec[:5]}")
# print(f"自己和自己的余弦相似度: {sim}")


texts = [
    ("苹果很好吃", "香蕉很好吃"),
    ("苹果很好吃", "苹果是一种水果"),
    ("苹果很好吃", "今天天气不错"),
]

for a, b in texts:
    # 1. 分别获取两个文本的向量
    va = client.embeddings.create(model="text-embedding-bge-m3", input=a).data[0].embedding
    vb = client.embeddings.create(model="text-embedding-bge-m3", input=b).data[0].embedding
    
    # 2. 计算余弦相似度
    sim = cosine_similarity(va, vb)
    
    # 3. 打印结果
    print(f"「{a}」vs「{b}」→ {sim:.4f}")


