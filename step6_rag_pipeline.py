"""
RAG 管线串联——把前 5 步的零件拼成一台能用的机器。

用法:
  # 建库（需要 LM Studio 加载 BGE-M3 嵌入模型）
  python step6_rag_pipeline.py build data/ knowledge_base.json

  # 问答（先 BGE-M3 检索，再 Qwen 生成回答）
  python step6_rag_pipeline.py ask "Python 协程怎么用？"

LM Studio 同一时间只能加载一个模型，所以问答阶段会分两步：
  1. BGE-M3 嵌入问题 + 检索 → 显示找到的资料块
  2. 提示用户切换到 Qwen → 调用 LLM 生成最终回答
"""

import sys
import os
from openai import OpenAI

from step3_chunker import chunk_text
from step4_vectorstore import VectorStore
from step5_loader import load_documents


# LM Studio 本地 API 地址
BASE_URL = "http://127.0.0.1:1234/v1"

# 嵌入模型（建库 + 检索用）
EMBEDDING_MODEL = "text-embedding-bge-m3"

# 对话模型（生成回答用）
CHAT_MODEL = "qwen/qwen3.5-9b"


def _get_client():
    """创建 API 客户端。每次调用都新建——因为中间用户可能切换了模型。"""
    return OpenAI(base_url=BASE_URL, api_key="not-needed")


def _embed(text: str) -> list[float]:
    """调用 BGE-M3 嵌入接口，返回 768 维向量。"""
    client = _get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _source_prefix(doc_name: str) -> str:
    """给每个 chunk 加上来源标记，检索时就能溯源。"""
    return f"[来源: {doc_name}]"


# ================================================================
# 命令 1: build —— 建知识库
# ================================================================

def build_knowledge_base(
    docs_dir: str,
    output_path: str = "knowledge_base.json",
    chunk_size: int = 512,
    overlap: int = 128,
) -> VectorStore:
    """扫描文档目录，分块、向量化，存入向量库，持久化为 JSON。

    需要 LM Studio 加载 BGE-M3 嵌入模型。
    """
    print("=" * 60)
    print("RAG 管线 — 构建知识库")
    print("=" * 60)
    print()
    print(f"[提示] 请确保 LM Studio 已加载嵌入模型: {EMBEDDING_MODEL}")
    print(f"        文档目录: {docs_dir}")
    print(f"        输出文件: {output_path}")
    print(f"        分块参数: chunk_size={chunk_size}, overlap={overlap}")
    print()

    # 第 1 步：加载文档
    print("─" * 40)
    print("第 1 步: 加载文档 (step5)")
    docs = load_documents(docs_dir)
    if not docs:
        print("[错误] 没有找到可用的文档，请检查目录路径")
        return VectorStore()
    print(f"  已加载 {len(docs)} 篇文档:")
    for d in docs:
        print(f"    [{d['type']}] {d['name']} ({len(d['content'])} 字符)")

    # 第 2 步：分块
    print()
    print("─" * 40)
    print("第 2 步: 文本分块 (step3)")
    store = VectorStore()
    total_chunks = 0

    for doc in docs:
        if not doc["content"].strip():
            print(f"  [跳过] {doc['name']} (空文档)")
            continue

        chunks = chunk_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
        print(f"  {doc['name']}: {len(chunks)} 个块")

        # 第 3 步：向量化 + 存入
        for chunk in chunks:
            text_with_source = f"{_source_prefix(doc['name'])}\n{chunk}"
            try:
                vec = _embed(chunk)
                store.add(text_with_source, vec)
                total_chunks += 1
            except Exception as e:
                print(f"    [错误] 向量化失败: {e}")
                print(f"    [提示] 请确认 LM Studio 已加载 {EMBEDDING_MODEL}")
                return VectorStore()

    print(f"  共 {total_chunks} 个块已向量化")

    # 第 3 步：持久化
    print()
    print("─" * 40)
    print("第 3 步: 持久化 (step4)")
    store.save(output_path)
    print(f"  知识库已保存到: {output_path}")
    print(f"  库大小: {len(store)} 个块")

    print()
    print("=" * 60)
    print("构建完成!")
    print("=" * 60)
    return store


# ================================================================
# 命令 2: ask —— 问答
# ================================================================

def ask(question: str, kb_path: str = "knowledge_base.json", top_k: int = 5) -> str:
    """加载知识库，检索相关文档块，调用 LLM 生成回答。

    流程分两步（因为 LM Studio 只能同时加载一个模型）：
      第 1 步: 用 BGE-M3 嵌入问题 → 检索 → 构建提示词
      第 2 步: 用 Qwen 基于提示词生成回答
    """
    print("=" * 60)
    print("RAG 管线 — 问答")
    print("=" * 60)
    print()

    # --- 第 1 步: 检索 ---
    print("─" * 40)
    print(f"第 1 步: 检索 (需要 {EMBEDDING_MODEL})")
    print()

    # 加载知识库
    if not os.path.exists(kb_path):
        print(f"[错误] 知识库文件不存在: {kb_path}")
        print(f"[提示] 请先运行: python step6_rag_pipeline.py build <文档目录> {kb_path}")
        return ""

    store = VectorStore.load(kb_path)
    print(f"  已加载知识库: {kb_path} ({len(store)} 个块)")

    # 嵌入问题
    try:
        question_vec = _embed(question)
    except Exception as e:
        print(f"  [错误] 嵌入失败: {e}")
        print(f"  [提示] 请确认 LM Studio 已加载 {EMBEDDING_MODEL}")
        return ""

    # 检索
    results = store.search(question_vec, top_k=top_k)
    print(f"  检索到 {len(results)} 个相关块:")
    print()
    for i, (text, sim) in enumerate(results, 1):
        # 截取第一行作为预览
        preview = text.split("\n")[0][:80]
        print(f"  [{i}] (相似度 {sim:.4f}) {preview}...")

    # --- 构建提示词 ---
    print()
    print("─" * 40)
    print("第 2 步: 构建提示词")

    references = []
    for i, (text, _) in enumerate(results, 1):
        references.append(f"[资料 {i}]\n{text}")

    prompt = _build_prompt(question, references)

    print(f"  提示词长度: {len(prompt)} 字符")
    print()

    # --- 第 2 步: 生成回答 ---
    print("─" * 40)
    print(f"第 3 步: 生成回答 (需要 {CHAT_MODEL})")
    print()

    try:
        answer = _call_llm(prompt)
        print("回答:")
        print(answer)
        return answer
    except Exception as e:
        # Qwen 没加载 → 打印提示词，用户手动问
        print(f"  [提示] LLM 调用失败: {e}")
        print(f"  [提示] 请在 LM Studio 中切换到 {CHAT_MODEL}，然后把下面的提示词发给模型。")
        print()
        print("=" * 60)
        print("提示词 (复制以下内容发送给 LLM):")
        print("=" * 60)
        print()
        print(prompt)
        return ""


def _build_prompt(question: str, references: list[str]) -> str:
    """把问题 + 检索到的资料 + 指令 拼接成完整的提示词。"""
    ref_text = "\n\n".join(references)

    return f"""你是一个知识助手，请根据以下参考资料回答用户的问题。

要求:
- 如果参考资料中有答案，请基于资料回答，并在引用处标注 [编号]
- 如果参考资料中没有相关信息，请如实说"根据现有资料，无法回答这个问题"，不要编造
- 用中文回答，语言简洁清晰

参考资料:
---
{ref_text}
---

用户问题: {question}

请回答:"""


def _call_llm(prompt: str) -> str:
    """调用 Qwen 对话模型，返回生成的回答。"""
    client = _get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    msg = response.choices[0].message
    answer = msg.content or getattr(msg, "reasoning_content", None)
    return (answer or str(msg)).strip()


# ================================================================
# 命令行入口
# ================================================================

def _print_usage():
    print("用法:")
    print("  建库: python step6_rag_pipeline.py build <文档目录> [输出文件]")
    print("  问答: python step6_rag_pipeline.py ask <问题> [知识库文件]")
    print()
    print("示例:")
    print('  python step6_rag_pipeline.py build data/ knowledge_base.json')
    print('  python step6_rag_pipeline.py ask "Python 协程怎么用？" knowledge_base.json')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "build":
        docs_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
        output_path = sys.argv[3] if len(sys.argv) > 3 else "knowledge_base.json"
        build_knowledge_base(docs_dir, output_path)

    elif command == "ask":
        if len(sys.argv) < 3:
            print("[错误] 请输入问题")
            print('示例: python step6_rag_pipeline.py ask "Python 协程怎么用？"')
            sys.exit(1)
        question = sys.argv[2]
        kb_path = sys.argv[3] if len(sys.argv) > 3 else "knowledge_base.json"
        ask(question, kb_path)

    else:
        print(f"[错误] 未知命令: {command}")
        _print_usage()
        sys.exit(1)
