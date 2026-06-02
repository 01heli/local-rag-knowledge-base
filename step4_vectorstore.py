import json
import math
from step2_embedding import cosine_similarity


class VectorStore:
    """一个最简向量库——存文本块和对应向量，支持暴搜。

    chunks[i] 和 vectors[i] 通过同一个索引 i 对应，
    这是刻意设计的数据布局：数据结构透明，方便调试和理解。
    """

    def __init__(self):
        self._chunks: list[str] = []
        self._vectors: list[list[float]] = []

    def add(self, text: str, vector: list[float]) -> None:
        """存入一个文本块和它的向量。"""
        self._chunks.append(text)
        self._vectors.append(vector)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """检索最相似的 top_k 个文本块。

        暴搜所有向量，逐一计算余弦相似度，返回按相似度降序排列的结果。

        返回：
            [(文本块, 相似度), ...]  list of tuples，按相似度从高到低排序
            如果库为空，返回空列表
        """
        if not self._vectors:
            return []

        scores = []
        for i, vec in enumerate(self._vectors):
            sim = cosine_similarity(query_vector, vec)
            scores.append((sim, i))

        # 按相似度降序
        scores.sort(key=lambda x: x[0], reverse=True)

        # 取 top_k
        results = []
        for sim, idx in scores[:top_k]:
            results.append((self._chunks[idx], sim))

        return results

    def __len__(self) -> int:
        """库中存储的文本块数量。"""
        return len(self._chunks)

    def save(self, filepath: str) -> None:
        """将向量库存为 JSON 文件——方便调试和检查数据质量。"""
        data = {
            "chunks": self._chunks,
            "vectors": self._vectors,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(filepath: str) -> "VectorStore":
        """从 JSON 文件加载向量库。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        store = VectorStore()
        store._chunks = data["chunks"]
        store._vectors = data["vectors"]
        return store


# ============================================================
# 测试（不需要 LM Studio，纯本地跑）
# ============================================================
if __name__ == "__main__":
    # 用随机小数模拟 768 维向量（避免依赖 LM Studio）
    import random
    random.seed(42)

    def fake_vector(seed: int) -> list[float]:
        """根据 seed 生成一个确定性的假向量，用于测试。"""
        r = random.Random(seed)
        return [r.random() for _ in range(768)]

    # --- 测试 1: 基本存取 ---
    print("=" * 60)
    print("测试 1: 基本存取")
    store = VectorStore()
    store.add("苹果是一种很好吃的水果", fake_vector(1))
    store.add("香蕉也是一种很好吃的水果", fake_vector(2))
    store.add("今天天气真不错", fake_vector(3))

    print(f"库大小: {len(store)}")
    results = store.search(fake_vector(1), top_k=3)
    for text, sim in results:
        print(f"  {sim:.4f} | {text[:40]}...")
    assert len(results) == 3, "应该返回 3 条结果"

    # --- 测试 2: top_k 限制 ---
    print("\n测试 2: top_k 限制")
    # 先多塞几条
    for i in range(4, 12):
        store.add(f"文档片段 {i}", fake_vector(i))

    print(f"库大小: {len(store)}")
    results = store.search(fake_vector(1), top_k=3)
    assert len(results) == 3, "top_k=3 应该只返回 3 条"
    for text, sim in results:
        print(f"  {sim:.4f} | {text[:40]}...")

    # --- 测试 3: 相似度排序 ---
    print("\n测试 3: 相似度排序")
    # 向量 1 和向量 2 相似（seed 接近）→ 1.0, 0.9996, ...
    # 向量 1 和向量 100 差异大 → 低分
    store.add("完全不相关的文本", fake_vector(100))
    results = store.search(fake_vector(1), top_k=5)
    print("向量 1 的检索结果（降序）:")
    for i, (text, sim) in enumerate(results):
        marker = " <-- 第一名应该是自己"
        print(f"  #{i+1}: {sim:.4f} | {text[:40]}...{marker if i == 0 else ''}")

    # --- 测试 4: save/load 往返 ---
    print("\n测试 4: save/load 往返")
    store.save("_test_vectorstore.json")
    loaded = VectorStore.load("_test_vectorstore.json")
    assert len(loaded) == len(store), "加载后大小应一致"
    assert loaded._chunks == store._chunks, "文本块应一致"
    assert loaded._vectors == store._vectors, "向量应一致"
    print(f"保存 → 加载成功，数据一致 ({len(loaded)} 条)")

    # 用加载的库再做一次检索，结果应一致
    results_loaded = loaded.search(fake_vector(1), top_k=3)
    results_orig = store.search(fake_vector(1), top_k=3)
    assert results_loaded == results_orig, "加载后的检索结果应一致"
    print("检索结果也一致")

    # 清理
    import os
    os.remove("_test_vectorstore.json")

    # --- 测试 5: 空库 ---
    print("\n测试 5: 空库")
    empty_store = VectorStore()
    assert len(empty_store) == 0
    assert empty_store.search(fake_vector(1)) == []
    print("空库 search 返回 []，没有报错")

    print("\n" + "=" * 60)
    print("5 个测试全部通过")
