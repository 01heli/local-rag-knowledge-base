"""
第 8 步: 优化与扩展——四个模块让 RAG 系统更实用。

模块:
  1. 多文档类型支持 —— HTML / JSON / CSV
  2. 检索质量评估 —— MRR, Precision@k, Recall@k
  3. 重排序 —— 关键词增强（默认） + LLM 重排序（可选）
  4. 对话历史 —— 多轮对话管理

用法:
  from step8_optimizations import (
      load_single_extended,      # 模块 1
      EvalMetrics, TestSet,      # 模块 2
      rerank_results,            # 模块 3
      ConversationHistory,       # 模块 4
      build_prompt_with_history, # 辅助: 拼接历史的 prompt
  )

自测 (不需要 LM Studio):
  python step8_optimizations.py
"""

import os
import re
import csv
import json
import math
import random
from html.parser import HTMLParser

# ================================================================
# 模块 1: 多文档类型支持
# ================================================================


class _HTMLStripper(HTMLParser):
    """从 HTML 中提取纯文本——跳过脚本/样式，保留段落结构。"""

    def __init__(self):
        super().__init__()
        self._text: list[str] = []
        self._skip = False  # 是否正在跳过 script/style 内容

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        elif tag in ("p", "br", "li", "tr", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._text.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        elif tag in ("p", "li", "tr", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._text.append("\n")

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self._text.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._text)


def read_html(filepath: str) -> str:
    """读取 HTML 文件，提取纯文本。"""
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    # 先用正则粗略去掉 <script> 和 <style> 块，作为保险
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    stripper = _HTMLStripper()
    try:
        stripper.feed(html)
    except Exception:
        # HTMLParser 对极不规范的 HTML 可能抛异常，回退到纯正则
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    text = stripper.get_text()
    # 合并多余空白
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _flatten_json(obj, prefix="", depth=0, max_depth=3, max_value_len=500) -> str:
    """递归展平 JSON 对象为可读文本。"""
    lines = []
    if depth > max_depth:
        return ""

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                lines.append(_flatten_json(value, full_key, depth + 1, max_depth, max_value_len))
            else:
                val_str = str(value)
                if len(val_str) > max_value_len:
                    val_str = val_str[:max_value_len] + "..."
                lines.append(f"{full_key}: {val_str}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            full_key = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                lines.append(_flatten_json(item, full_key, depth + 1, max_depth, max_value_len))
            else:
                val_str = str(item)
                if len(val_str) > max_value_len:
                    val_str = val_str[:max_value_len] + "..."
                lines.append(f"{full_key}: {val_str}")
    else:
        val_str = str(obj)
        if len(val_str) > max_value_len:
            val_str = val_str[:max_value_len] + "..."
        lines.append(f"{prefix}: {val_str}" if prefix else val_str)

    return "\n".join(line for line in lines if line)


def read_json(filepath: str, encoding: str = "utf-8") -> str:
    """读取 JSON 文件，展平为人类可读的文本。"""
    with open(filepath, "r", encoding=encoding) as f:
        data = json.load(f)
    return _flatten_json(data)


def read_csv(filepath: str, encoding: str = "utf-8") -> str:
    """读取 CSV 文件，转为制表符分隔的文本。"""
    text = ""
    try:
        with open(filepath, "r", encoding=encoding) as f:
            reader = csv.reader(f)
            rows = ["\t".join(row) for row in reader]
            text = "\n".join(rows)
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="gbk") as f:
            reader = csv.reader(f)
            rows = ["\t".join(row) for row in reader]
            text = "\n".join(rows)
    return text


# ---- 导入 step5 的原版加载器（本地导入，避免循环依赖）----

def load_single_extended(filepath: str) -> dict | None:
    """扩展版单文件加载器——在 step5 基础上增加 HTML/JSON/CSV。

    返回格式与 step5_loader.load_single() 一致:
        {"path": ..., "name": ..., "type": ..., "content": ...}
    如果格式不支持或读取失败，返回 None。
    """
    from step5_loader import load_single

    if not os.path.isfile(filepath):
        print(f"  [跳过] 文件不存在: {filepath}")
        return None

    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    try:
        if ext in (".html", ".htm"):
            content = read_html(filepath)
            return {
                "path": filepath,
                "name": os.path.basename(filepath),
                "type": "html",
                "content": content,
            }

        elif ext == ".json":
            content = read_json(filepath)
            return {
                "path": filepath,
                "name": os.path.basename(filepath),
                "type": "json",
                "content": content,
            }

        elif ext == ".csv":
            content = read_csv(filepath)
            return {
                "path": filepath,
                "name": os.path.basename(filepath),
                "type": "csv",
                "content": content,
            }

        else:
            # 委托给 step5 原版（TXT/MD/PDF）
            return load_single(filepath)

    except Exception as e:
        print(f"  [跳过] 读取失败: {filepath} ({e})")
        return None


# ================================================================
# 模块 2: 检索质量评估
# ================================================================

class EvalMetrics:
    """检索质量指标——MRR, Precision@k, Recall@k。

    所有方法均为静态方法，不依赖外部状态。
    """

    @staticmethod
    def compute_mrr(retrieved_ranks: list[int], relevant_ids: set[int]) -> float:
        """Mean Reciprocal Rank——第一个相关结果的排名的倒数。

        retrieved_ranks: 检索结果中每个 chunk 在库中的索引，按排名顺序排列
        relevant_ids: 标注为相关的 chunk 索引集合

        返回: 0~1，越高越好。无命中时返回 0。
        """
        for rank, chunk_id in enumerate(retrieved_ranks, 1):
            if chunk_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def compute_precision_at_k(
        retrieved_ranks: list[int], relevant_ids: set[int], k: int
    ) -> float:
        """Precision@k——前 k 个结果中相关结果的占比。"""
        if k <= 0:
            return 0.0
        top_k = set(retrieved_ranks[:k])
        if not top_k:
            return 0.0
        hits = len(top_k & relevant_ids)
        return hits / min(k, len(top_k))

    @staticmethod
    def compute_recall_at_k(
        retrieved_ranks: list[int], relevant_ids: set[int], k: int
    ) -> float:
        """Recall@k——所有相关结果中，前 k 个命中了多少。"""
        if not relevant_ids:
            return 0.0
        top_k = set(retrieved_ranks[:k])
        hits = len(top_k & relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def compute_all(
        retrieved_ranks: list[int],
        relevant_ids: set[int],
        k_values: list[int] | None = None,
    ) -> dict:
        """一次性计算全部指标，返回 dict。"""
        if k_values is None:
            k_values = [1, 3, 5, 10]
        metrics = {"MRR": EvalMetrics.compute_mrr(retrieved_ranks, relevant_ids)}
        for k in k_values:
            metrics[f"P@{k}"] = EvalMetrics.compute_precision_at_k(
                retrieved_ranks, relevant_ids, k
            )
            metrics[f"R@{k}"] = EvalMetrics.compute_recall_at_k(
                retrieved_ranks, relevant_ids, k
            )
        return metrics


class TestSet:
    """检索测试集——管理一组标注好正确答案的查询。

    文件格式 (JSON):
      {
        "name": "测试集名称",
        "queries": [
          {"id": "q1", "question": "...", "relevant_chunk_ids": [3, 7]},
          ...
        ]
      }
    """

    def __init__(self, filepath: str = ""):
        self.name = "unnamed"
        self._queries: list[dict] = []
        self._filepath = filepath
        if filepath and os.path.exists(filepath):
            self.load(filepath)

    def add_query(self, qid: str, question: str, relevant_ids: list[int], notes: str = ""):
        """添加一条标注好的查询。"""
        # 如果已存在同 id，先移除
        self._queries = [q for q in self._queries if q["id"] != qid]
        self._queries.append(
            {"id": qid, "question": question, "relevant_chunk_ids": list(relevant_ids)}
        )
        if notes:
            self._queries[-1]["notes"] = notes

    def remove_query(self, qid: str):
        """按 id 移除一条查询。"""
        self._queries = [q for q in self._queries if q["id"] != qid]

    @property
    def queries(self) -> list[dict]:
        return list(self._queries)

    def __len__(self) -> int:
        return len(self._queries)

    def save(self, filepath: str | None = None):
        """保存测试集到 JSON 文件。"""
        path = filepath or self._filepath
        if not path:
            raise ValueError("未指定保存路径")
        data = {"name": self.name, "queries": self._queries}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """从 JSON 文件加载测试集。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.name = data.get("name", "unnamed")
        self._queries = data.get("queries", [])
        self._filepath = filepath

    @staticmethod
    def interactive_annotate(vectorstore, questions_file: str) -> "TestSet":
        """交互式标注模式——用户手动标记检索结果是否相关。

        vectorstore: 已加载的 VectorStore 实例
        questions_file: 包含问题的文本文件（一行一个问题）

        流程:
          1. 读取问题列表
          2. 每个问题: embed → search → 展示 top-10 预览
          3. 用户输入相关序号（如 "1,3,5"），或 "skip" 跳过
          4. 生成 TestSet

        注意: 此函数需要 embedding_fn 来向量化问题，但它是交互式的，
              所以由外部（step7_cli.py）调用时传入 embedding_fn。
              这里只提供静态方法骨架——实际交互由 CLI 驱动。
        """
        ts = TestSet()
        ts.name = f"annotated_{os.path.basename(questions_file)}"

        if not os.path.exists(questions_file):
            print(f"[错误] 问题文件不存在: {questions_file}")
            return ts

        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]

        print(f"已加载 {len(questions)} 个问题")
        print(f"知识库大小: {len(vectorstore)} 个块")
        print()
        print("交互式标注说明:")
        print("  - 输入相关块的编号，逗号分隔（如 1,3,5）")
        print("  - 输入 'all' 表示全部相关")
        print("  - 输入 'none' 或直接回车表示没有相关块")
        print("  - 输入 'skip' 跳过此问题")
        print()

        # 注意: 交互标注需要 embedding_fn，这里返回空 TestSet 作为骨架
        # 实际标注流程由 step7_cli.py 中的 /annotate 命令驱动
        return ts


class EvalRunner:
    """运行评估——遍历测试集，计算指标，生成报告。

    用法:
      runner = EvalRunner("knowledge_base.json", "test_questions.json", my_embed_fn)
      report = runner.run()
      print(runner.generate_report())
    """

    def __init__(
        self,
        kb_path: str,
        testset_path: str,
        embedding_fn,
        top_k: int = 5,
    ):
        from step4_vectorstore import VectorStore

        self.store = VectorStore.load(kb_path)
        self.testset = TestSet(testset_path)
        self.embed = embedding_fn
        self.top_k = top_k
        self._per_query: list[dict] = []

    def run(self, k_values: list[int] | None = None) -> dict:
        """运行评估，返回聚合指标。"""
        if k_values is None:
            k_values = [1, 3, 5, 10]

        self._per_query = []
        all_metrics: list[dict] = []

        for q in self.testset.queries:
            try:
                q_vec = self.embed(q["question"])
                results = self.store.search(q_vec, top_k=self.top_k)

                # 获取检索到的 chunk 在 VectorStore 中的索引
                # search 返回 [(text, similarity), ...]——需要反查索引
                retrieved_indices = []
                for text, sim in results:
                    try:
                        idx = self.store._chunks.index(text)
                        retrieved_indices.append(idx)
                    except ValueError:
                        pass

                relevant = set(q["relevant_chunk_ids"])
                metrics = EvalMetrics.compute_all(
                    retrieved_indices, relevant, k_values
                )
                metrics["question"] = q["question"]
                metrics["id"] = q["id"]
                all_metrics.append(metrics)
                self._per_query.append(
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "retrieved_indices": retrieved_indices,
                        "relevant_ids": list(relevant),
                        "metrics": metrics,
                    }
                )
            except Exception as e:
                print(f"  [警告] 评估 '{q['id']}' 失败: {e}")

        # 聚合
        agg = {"num_queries": len(all_metrics)}
        if not all_metrics:
            return agg

        for key in all_metrics[0]:
            if key in ("question", "id"):
                continue
            values = [m[key] for m in all_metrics if key in m]
            agg[f"avg_{key}"] = sum(values) / len(values) if values else 0.0

        return agg

    def generate_report(self) -> str:
        """生成人类可读的评估报告。"""
        if not self._per_query:
            return "没有评估数据，请先调用 run()。"

        lines = [
            "=" * 60,
            "  检索质量评估报告",
            "=" * 60,
            f"  测试集: {self.testset.name}",
            f"  知识库: {len(self.store)} 个块",
            f"  查询数: {len(self._per_query)}",
            f"  top_k:  {self.top_k}",
            "",
            "─" * 40,
            "  逐查询明细",
            "─" * 40,
        ]

        for pq in self._per_query:
            m = pq["metrics"]
            lines.append(f"  [{pq['id']}] {pq['question'][:60]}")
            lines.append(f"    相关块: {pq['relevant_ids']}")
            lines.append(f"    检索排名: {pq['retrieved_indices'][:self.top_k]}")
            lines.append(
                f"    MRR={m.get('MRR', 0):.4f}  "
                f"P@3={m.get('P@3', 0):.4f}  "
                f"R@3={m.get('R@3', 0):.4f}"
            )
            lines.append("")

        # 汇总
        if len(self._per_query) > 1:
            lines.append("─" * 40)
            lines.append("  汇总指标")
            lines.append("─" * 40)
            # 从 per_query 重新聚合
            agg = {}
            for pq in self._per_query:
                for key, val in pq["metrics"].items():
                    if key in ("question", "id"):
                        continue
                    agg.setdefault(key, []).append(val)
            for key in sorted(agg):
                avg = sum(agg[key]) / len(agg[key])
                lines.append(f"    avg_{key}: {avg:.4f}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ================================================================
# 模块 3: 重排序
# ================================================================

# 内置停用词表（避免引入 jieba 等额外依赖）
STOP_WORDS_ZH: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
    "们", "那", "什么", "怎么", "如何", "可以", "因为", "所以",
    "但是", "如果", "虽然", "而且", "还是", "只是", "已经", "这个",
    "那个", "哪些", "哪里", "为什么", "怎么", "怎么样",
}

STOP_WORDS_EN: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some",
    "such", "no", "only", "own", "same", "than", "too", "very", "just",
    "that", "this", "it", "its", "what", "which", "who", "whom", "how",
    "when", "where", "why",
}


def tokenize(text: str) -> set[str]:
    """简单分词——切分中英文混合文本，返回去停用词后的 token 集合。

    中文: 按 CJK 标点边界切分，保留长度 >= 1 的片段
    英文: 按空白和标点切分，保留长度 >= 2 的单词
    """
    tokens: set[str] = set()

    # 切分：在标点/空白处断开
    # 保留中文字符、英文字母、数字
    raw = re.findall(r"[一-鿿]+|[a-zA-Z]{2,}|\d+", text.lower())

    for token in raw:
        token = token.strip()
        if not token:
            continue
        # 停用词过滤
        if token in STOP_WORDS_ZH or token in STOP_WORDS_EN:
            continue
        # 单个英文字母跳过
        if len(token) == 1 and token.isascii() and token.isalpha():
            continue
        tokens.add(token)

    return tokens


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard 相似度——交集大小 / 并集大小。"""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class KeywordReranker:
    """基于关键词匹配的重排序器——不需要任何 API 调用。

    算法:
      1. 对查询和每个 chunk 分别分词
      2. 计算 Jaccard 相似度
      3. 额外奖励: 完整短语命中（查询中的连续词组在 chunk 中出现）
      4. 最终分 = α × 原始向量相似度 + (1-α) × 关键词分

    参数:
      alpha: 向量相似度权重 (0~1)，默认 0.7。
             越高越信任向量检索结果。
    """

    def __init__(self, alpha: float = 0.7):
        self.alpha = max(0.0, min(1.0, alpha))

    def rerank(
        self, query: str, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """重排序——返回按新分数降序排列的结果列表。"""
        if not results:
            return []

        query_tokens = tokenize(query)

        # 如果查询分词后为空（纯停用词），回退原序
        if not query_tokens:
            return results

        scored = []
        for text, vec_sim in results:
            chunk_tokens = tokenize(text)
            keyword_score = _jaccard(query_tokens, chunk_tokens)

            # 短语奖励: 查询中的连续 2-4 词序列在 chunk 中出现过
            phrase_bonus = self._phrase_bonus(query, text)

            combined_keyword = min(1.0, keyword_score + phrase_bonus)
            final_score = self.alpha * vec_sim + (1 - self.alpha) * combined_keyword

            scored.append((text, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _phrase_bonus(query: str, text: str, max_len: int = 4) -> float:
        """检测查询中的连续词序列是否在 chunk 中原样出现。

        返回: 0~0.2 的奖励值（每个命中短语 +0.05，上限 0.2）。
        """
        # 提取查询中的中文连续字符和英文连续字符
        q_tokens = re.findall(r"[一-鿿]+|[a-zA-Z]+", query.lower())
        text_lower = text.lower()

        bonus = 0.0
        for token in q_tokens:
            if len(token) >= 3 and token in text_lower:
                bonus += 0.05
            # 对于更长的中文 token，检查 3-4 字子串
            if len(token) >= 4:
                for i in range(len(token) - 2):
                    sub = token[i : i + 3]
                    if sub in text_lower:
                        bonus += 0.03
                        break

        return min(0.2, bonus)


class LLMReranker:
    """基于 LLM 的重排序器——让模型判断每个 chunk 与问题的相关度。

    需要 LM Studio 加载对话模型。如果 LLM 调用失败，回退到原始顺序。
    """

    def __init__(self, get_client_fn, model: str = "qwen/qwen3.5-9b"):
        self._get_client = get_client_fn
        self.model = model

    def rerank(
        self, query: str, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """调用 LLM 重排序。"""
        if len(results) <= 1:
            return results

        # 构建 prompt: 让 LLM 按相关度排序
        chunks_text = []
        for i, (text, _) in enumerate(results, 1):
            preview = text.replace("\n", " ")[:200]
            chunks_text.append(f"[{i}] {preview}")

        prompt = f"""请根据以下文本块与问题的相关度，从高到低排序。
输出格式: 只需输出排序后的编号，逗号分隔。例如: 3,1,5,2,4

问题: {query}

文本块:
{chr(10).join(chunks_text)}

排序结果（只输出编号）:"""

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            msg = response.choices[0].message
            answer = (msg.content or "").strip()

            # 解析返回的编号
            nums = re.findall(r"\d+", answer)
            order = [int(n) - 1 for n in nums if 1 <= int(n) <= len(results)]

            if not order:
                return results

            # 按 LLM 给出的顺序重排
            reordered = [results[i] for i in order if i < len(results)]

            # 补充未被 LLM 提到的结果（排在末尾）
            mentioned = set(order)
            for i in range(len(results)):
                if i not in mentioned:
                    reordered.append(results[i])

            return reordered

        except Exception as e:
            print(f"  [警告] LLM 重排序失败: {e}，回退到原始顺序")
            return results


def rerank_results(
    query: str,
    results: list[tuple[str, float]],
    method: str = "keyword",
    **kwargs,
) -> list[tuple[str, float]]:
    """统一重排序接口。

    参数:
      query: 用户问题
      results: [(文本块, 相似度), ...] —— VectorStore.search() 的返回
      method: "keyword" | "llm" | "none"
      **kwargs: 传递给具体 reranker 的参数（如 alpha）

    返回: 重排后的 [(文本块, 分数), ...]
    """
    if method == "none" or not results:
        return results

    if method == "keyword":
        alpha = kwargs.get("alpha", 0.7)
        reranker = KeywordReranker(alpha=alpha)
        return reranker.rerank(query, results)

    if method == "llm":
        get_client_fn = kwargs.get("get_client_fn")
        if get_client_fn is None:
            from openai import OpenAI

            def get_client_fn():
                return OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")

        reranker = LLMReranker(get_client_fn)
        return reranker.rerank(query, results)

    print(f"  [警告] 未知重排序方法: {method}，保持原序")
    return results


# ================================================================
# 模块 4: 对话历史
# ================================================================


class ConversationHistory:
    """多轮对话管理器——记录本轮会话中的问答，让模型"记住"之前的上下文。

    每轮对话存储:
      - question: 用户问题
      - answer: 助手回答
      - references: 该轮检索到的文档块列表 [(text, score), ...]

    特性:
      - 容量限制: 最多保存 max_turns 轮，超出自动丢弃最早的
      - get_context_for_prompt(): 生成拼接好的历史上下文字符串
      - format_history_for_display(): 格式化输出给用户查看
    """

    def __init__(self, max_turns: int = 20):
        self._turns: list[dict] = []
        self.max_turns = max_turns

    def add_turn(
        self,
        question: str,
        answer: str,
        references: list[tuple[str, float]] | None = None,
    ):
        """记录一轮对话。"""
        self._turns.append(
            {
                "question": question,
                "answer": answer,
                "ref_count": len(references) if references else 0,
                "references": references or [],
            }
        )
        # 超出容量 → 丢弃最早的
        while len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def clear(self):
        """清空全部对话历史。"""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def get_context_for_prompt(self, max_context_turns: int = 5) -> str:
        """生成一段文本，嵌入到新的 RAG prompt 中以提供历史上下文。

        只取最近 max_context_turns 轮，避免 prompt 过长。
        """
        if not self._turns:
            return ""

        recent = self._turns[-max_context_turns:]

        lines = ["[对话历史]"]
        for i, turn in enumerate(recent, 1):
            lines.append(f"用户: {turn['question']}")
            # 截断太长的回答
            ans = turn["answer"]
            if len(ans) > 300:
                ans = ans[:300] + "..."
            lines.append(f"助手: {ans}")
            lines.append("")

        lines.append("[当前对话]")
        return "\n".join(lines)

    def format_history_for_display(self, n: int = 10) -> str:
        """格式化输出对话历史，用于 /history 命令展示。"""
        if not self._turns:
            return "（暂无对话历史）"

        recent = self._turns[-n:]
        lines = []
        for i, turn in enumerate(recent, 1):
            actual_idx = len(self._turns) - len(recent) + i
            lines.append(f"[{actual_idx}] 用户: {turn['question']}")
            ans = turn["answer"]
            if len(ans) > 200:
                ans = ans[:200] + "..."
            lines.append(f"[{actual_idx}] 助手: {ans}")
            if turn["ref_count"] > 0:
                lines.append(f"[{actual_idx}] (参考 {turn['ref_count']} 个文档块)")
            lines.append("")

        return "\n".join(lines).strip()


def build_prompt_with_history(
    question: str,
    references: list[str],
    history_context: str = "",
) -> str:
    """拼接最终的 RAG 提示词——在标准 prompt 前面加上对话历史。

    参数:
      question: 当前用户问题
      references: 检索到的资料文本列表（已编号格式化）
      history_context: ConversationHistory.get_context_for_prompt() 的返回值
    """
    ref_text = "\n\n".join(references)

    history_section = ""
    if history_context:
        history_section = f"""{history_context}

---

"""

    return f"""{history_section}你是一个知识助手，请根据以下参考资料回答用户的问题。

要求:
- 如果参考资料中有答案，请基于资料回答，并在引用处标注 [编号]
- 如果参考资料中没有相关信息，请如实说"根据现有资料，无法回答这个问题"，不要编造
- 用中文回答，语言简洁清晰
- 如果对话历史中有相关上下文，请结合上下文理解用户意图

参考资料:
---
{ref_text}
---

用户问题: {question}

请回答:"""


# ================================================================
# 自测代码（不需要 LM Studio，纯本地跑）
# ================================================================

if __name__ == "__main__":
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="step8_test_")
    print(f"测试临时目录: {tmpdir}\n")

    passed = 0
    failed = 0

    def check(condition, msg):
        global passed, failed
        if condition:
            passed += 1
            print(f"  [OK] {msg}")
        else:
            failed += 1
            print(f"  [FAIL] {msg}")

    # ============================================================
    # 测试 1: 多文档类型支持
    # ============================================================
    print("=" * 60)
    print("模块 1: 多文档类型支持")
    print("=" * 60)

    # 1a: HTML
    print("\n1a: HTML 提取")
    html_content = """<html><head><title>Test</title></head><body>
    <h1>标题</h1>
    <p>这是第一段内容。</p>
    <script>console.log('should be hidden')</script>
    <p>这是第二段内容，包含<b>加粗</b>文字。</p>
    </body></html>"""
    html_path = os.path.join(tmpdir, "test.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    result = load_single_extended(html_path)
    check(result is not None, "load_single_extended 应返回非 None")
    check(result["type"] == "html", f"类型应为 html，实际: {result['type']}")
    check("第一段内容" in result["content"], "应包含第一段内容")
    check("console.log" not in result["content"], "不应包含 script 内容")
    check("加粗" in result["content"], "应包含加粗文字")

    # 1b: CSV
    print("\n1b: CSV 读取")
    csv_content = "姓名,年龄,城市\n张三,25,北京\n李四,30,上海"
    csv_path = os.path.join(tmpdir, "test.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    result = load_single_extended(csv_path)
    check(result is not None, "load_single_extended 应返回非 None")
    check(result["type"] == "csv", f"类型应为 csv，实际: {result['type']}")
    check("张三" in result["content"], "应包含张三")
    check("25" in result["content"], "应包含 25")

    # 1c: JSON
    print("\n1c: JSON 扁平化")
    json_content = json.dumps(
        {
            "title": "测试文档",
            "author": {"name": "张三", "email": "zhang@test.com"},
            "tags": ["python", "rag", "ai"],
        },
        ensure_ascii=False,
    )
    json_path = os.path.join(tmpdir, "test.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_content)

    result = load_single_extended(json_path)
    check(result is not None, "load_single_extended 应返回非 None")
    check(result["type"] == "json", f"类型应为 json，实际: {result['type']}")
    check("python" in result["content"], "应包含 python")
    check("title" in result["content"], "应包含 title 键")

    # 1d: 委托给 step5（TXT）
    print("\n1d: TXT 委托给 step5")
    txt_path = os.path.join(tmpdir, "test.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("纯文本测试内容")
    result = load_single_extended(txt_path)
    check(result is not None, "TXT 应返回非 None")
    check(result["type"] == "txt", f"类型应为 txt，实际: {result['type']}")
    check("纯文本测试内容" in result["content"], "应包含文本内容")

    # 1e: 不存在文件
    print("\n1e: 不存在文件")
    result = load_single_extended(os.path.join(tmpdir, "nope.xyz"))
    check(result is None, "不存在文件应返回 None")

    # ============================================================
    # 测试 2: 检索质量评估
    # ============================================================
    print("\n" + "=" * 60)
    print("模块 2: 检索质量评估")
    print("=" * 60)

    # 2a: 基础指标 — 完美命中（前3名全是相关结果）
    print("\n2a: 基础指标 — 完美命中（前3名全是相关结果）")
    retrieved = [3, 7, 1, 5, 9]
    relevant = {3, 7, 1}
    metrics = EvalMetrics.compute_all(retrieved, relevant, k_values=[1, 3, 5])
    check(abs(metrics["MRR"] - 1.0) < 0.001, f"MRR 应 = 1.0，实际: {metrics['MRR']:.4f}")
    # 前3名全部命中 → P@3 = 3/3 = 1.0
    check(abs(metrics["P@3"] - 1.0) < 0.001, f"P@3 应 = 1.0，实际: {metrics['P@3']:.4f}")
    # 前3名全部命中 → R@3 = 3/3 = 1.0
    check(abs(metrics["R@3"] - 1.0) < 0.001, f"R@3 应 = 1.0，实际: {metrics['R@3']:.4f}")
    check(abs(metrics["R@5"] - 1.0) < 0.001, f"R@5 应 = 1.0 (全部命中), 实际: {metrics['R@5']:.4f}")

    # 2b: MRR — 第3位才命中
    print("\n2b: MRR — 第3位命中")
    mrr = EvalMetrics.compute_mrr([7, 9, 3, 1], {3})
    check(abs(mrr - 1 / 3) < 0.001, f"MRR 应 = 0.333，实际: {mrr:.4f}")

    # 2c: 完全未命中
    print("\n2c: 完全未命中")
    mrr = EvalMetrics.compute_mrr([7, 9, 2], {3})
    check(mrr == 0.0, f"MRR 应为 0，实际: {mrr:.4f}")

    # 2d: Precision@k — 相关块为空
    print("\n2d: Recall@k — 无相关块")
    rec = EvalMetrics.compute_recall_at_k([1, 2, 3], set(), 3)
    check(rec == 0.0, f"无相关块时 Recall 应为 0，实际: {rec:.4f}")

    # 2e: TestSet 存取
    print("\n2e: TestSet 存取")
    ts = TestSet()
    ts.name = "测试集"
    ts.add_query("q1", "什么是RAG？", [3, 7, 12])
    ts.add_query("q2", "Python协程怎么用？", [1, 5])
    check(len(ts) == 2, f"应有 2 条查询，实际: {len(ts)}")

    ts_path = os.path.join(tmpdir, "testset.json")
    ts.save(ts_path)
    ts2 = TestSet(ts_path)
    check(len(ts2) == 2, f"加载后应有 2 条，实际: {len(ts2)}")
    check(ts2.queries[0]["question"] == "什么是RAG？", "第一条问题应匹配")

    # 2f: TestSet — 重复 id 覆盖（新条目追加到末尾）
    print("\n2f: TestSet — 同 id 覆盖")
    ts.add_query("q1", "什么是RAG？（修订版）", [3, 7, 12, 15])
    check(len(ts) == 2, f"覆盖后仍为 2 条，实际: {len(ts)}")
    # 覆盖后 q1 被移除再追加到末尾，所以 q1 现在是 queries[1]
    all_qs = {q["id"]: q for q in ts.queries}
    check(all_qs["q1"]["question"] == "什么是RAG？（修订版）", "q1 问题应被更新")
    check(len(all_qs["q1"]["relevant_chunk_ids"]) == 4, "q1 相关块数应为 4")

    # 2g: TestSet — 删除
    print("\n2g: TestSet — 删除")
    ts.remove_query("q1")
    check(len(ts) == 1, f"删除后应为 1 条，实际: {len(ts)}")

    # ============================================================
    # 测试 3: 重排序
    # ============================================================
    print("\n" + "=" * 60)
    print("模块 3: 重排序")
    print("=" * 60)

    # 3a: 关键词重排序 — 基本功能
    print("\n3a: 关键词重排序 — 语义相关 vs 关键词相关")
    results = [
        ("JavaScript 是一种前端编程语言，用于构建交互式网页应用", 0.75),
        ("Python 是一种广泛使用的编程语言，特别适合数据分析和 AI 开发", 0.73),
        ("今天天气真好，适合出门散步", 0.70),
    ]
    reranker = KeywordReranker(alpha=0.5)
    reranked = reranker.rerank("Python数据分析", results)

    # Python 相关的应该排到第一位
    top_text = reranked[0][0]
    check(
        "Python" in top_text,
        f"Python 相关块应排第一，实际第一: {top_text[:50]}",
    )

    # 3b: 关键词重排序 — alpha=1 保持原序
    print("\n3b: alpha=1 应保持原序")
    reranker_full_vec = KeywordReranker(alpha=1.0)
    reranked = reranker_full_vec.rerank("Python数据分析", results)
    for i, (text, _) in enumerate(reranked):
        check(
            text == results[i][0],
            f"位置 {i} 应保持原序",
        )

    # 3c: 空结果
    print("\n3c: 空结果")
    reranked = reranker.rerank("test", [])
    check(len(reranked) == 0, "空结果应返回空列表")

    # 3d: tokenize 函数
    print("\n3d: tokenize 分词")
    tokens = tokenize("Python数据分析 和 机器学习")
    check("python" in tokens, "应包含 python")
    check("数据分析" in tokens, "应包含 数据分析")
    check("和" not in tokens, "停用词 '和' 应被过滤")
    check("机器" in tokens or "机器学习" in tokens, "应包含机器学习相关 token")

    # 3e: 短语奖励
    print("\n3e: 短语精确匹配奖励")
    query = "Python异步编程"
    text = "Python 异步编程是一种重要的并发编程范式"
    bonus = KeywordReranker._phrase_bonus(query, text)
    check(bonus > 0, f"短语命中应有奖励，实际: {bonus:.3f}")

    # 3f: rerank_results 统一接口
    print("\n3f: rerank_results 统一接口")
    reranked = rerank_results("Python数据分析", results, method="keyword", alpha=0.5)
    check(len(reranked) == 3, f"应返回 3 条结果，实际: {len(reranked)}")
    reranked_none = rerank_results("Python数据分析", results, method="none")
    check(reranked_none == results, "method='none' 应返回原序")

    # ============================================================
    # 测试 4: 对话历史
    # ============================================================
    print("\n" + "=" * 60)
    print("模块 4: 对话历史")
    print("=" * 60)

    # 4a: 基本存取
    print("\n4a: 基本存取")
    hist = ConversationHistory(max_turns=20)
    check(len(hist) == 0, f"初始应为 0，实际: {len(hist)}")

    hist.add_turn("什么是RAG？", "RAG是检索增强生成技术...", [("chunk1", 0.9)])
    check(len(hist) == 1, f"添加后应为 1，实际: {len(hist)}")

    hist.add_turn("它有什么优点？", "RAG可以引用外部知识...", [("chunk2", 0.85)])
    check(len(hist) == 2, f"再次添加后应为 2，实际: {len(hist)}")

    # 4b: get_context_for_prompt
    print("\n4b: get_context_for_prompt")
    ctx = hist.get_context_for_prompt(max_context_turns=5)
    check("[对话历史]" in ctx, "应包含 [对话历史] 标记")
    check("什么是RAG" in ctx, "应包含第一轮问题")
    check("它有什么优点" in ctx, "应包含第二轮问题")
    check("[当前对话]" in ctx, "应包含 [当前对话] 标记")

    # 4c: format_history_for_display
    print("\n4c: format_history_for_display")
    display = hist.format_history_for_display()
    check("[1] 用户:" in display, "应包含第1轮")
    check("[2] 用户:" in display, "应包含第2轮")

    # 4d: clear
    print("\n4d: clear")
    hist.clear()
    check(len(hist) == 0, f"清空后应为 0，实际: {len(hist)}")
    check(hist.get_context_for_prompt() == "", "清空后上下文应为空字符串")

    # 4e: 容量限制
    print("\n4e: 容量限制 (max_turns=3)")
    hist_small = ConversationHistory(max_turns=3)
    for i in range(5):
        hist_small.add_turn(f"Q{i}", f"A{i}")
    check(len(hist_small) == 3, f"超出容量应截断为 3，实际: {len(hist_small)}")
    # 最早的应该被丢弃
    ctx = hist_small.get_context_for_prompt(max_context_turns=10)
    check("Q0" not in ctx, "最早的 Q0 应被丢弃")
    check("Q4" in ctx, "最新的 Q4 应保留")

    # 4f: build_prompt_with_history
    print("\n4f: build_prompt_with_history")
    prompt = build_prompt_with_history(
        "Python协程怎么用？",
        ["[资料 1]\nPython协程使用async/await语法..."],
        history_context="[对话历史]\n用户: 什么是RAG？\n助手: RAG是...\n\n[当前对话]",
    )
    check("[对话历史]" in prompt, "prompt 应包含历史")
    check("[资料 1]" in prompt, "prompt 应包含参考资料")
    check("Python协程怎么用" in prompt, "prompt 应包含当前问题")

    # 4g: 无历史时的 prompt
    print("\n4g: 无历史时的 prompt")
    prompt_no_hist = build_prompt_with_history(
        "什么是RAG？", ["[资料 1]\nRAG全称是检索增强生成..."]
    )
    check("[对话历史]" not in prompt_no_hist, "无历史时不应包含 [对话历史]")
    check("参考资料" in prompt_no_hist, "仍应包含参考资料")

    # ============================================================
    # 清理 + 总结
    # ============================================================
    shutil.rmtree(tmpdir)
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed > 0:
        print("\n部分测试失败，请检查上述 [FAIL] 标记。")
        raise SystemExit(1)
    else:
        print("\n全部测试通过！")
