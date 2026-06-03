"""
交互式 RAG 命令行界面——可以持续提问、查看检索结果、构建知识库。

用法:
  python step7_cli.py [知识库路径]

启动后进入交互式对话环境，支持以下命令:
  /build <目录>    构建知识库（需要 LM Studio 加载 BGE-M3）
  /ask <问题>      RAG 问答（也可直接输入问题，无需 /ask 前缀）
  /sources         查看最近一次检索到的参考资料全文
  /kb <路径>       切换知识库文件
  /topk <数量>     设置检索返回的块数（默认 5）
  /status          查看当前状态（知识库路径、检索数量等）
  /help            显示帮助信息
  /exit, /quit     退出程序

LM Studio 模型切换提示:
  - 建库阶段: 需要加载 BGE-M3 嵌入模型
  - 问答阶段: 先用 BGE-M3 检索，再切换到 Qwen3.5-9B 生成回答
  - 程序会在需要切换时给出明确提示
"""

import sys
import os
import shlex

from openai import OpenAI

from step3_chunker import chunk_text
from step4_vectorstore import VectorStore
from step5_loader import load_documents
from step8_optimizations import (
    rerank_results,
    ConversationHistory,
    build_prompt_with_history,
)
from typing import Optional

# ================================================================
# 配置
# ================================================================

BASE_URL = "http://127.0.0.1:1234/v1"
EMBEDDING_MODEL = "text-embedding-bge-m3"
CHAT_MODEL = "qwen/qwen3.5-9b"

# ================================================================
# 模型检测辅助函数
# ================================================================

_last_checked_model: Optional[str] = None
_last_check_time: float = 0.0


def _get_current_model(force: bool = False) -> Optional[str]:
    """查询 LM Studio 当前加载的模型名。

    通过 /v1/models 端点获取，带缓存（5秒内不重复查询）。
    返回 None 表示 LM Studio 未启动或无法访问。
    """
    global _last_checked_model, _last_check_time
    import time
    now = time.time()
    if not force and (now - _last_check_time) < 5:
        return _last_checked_model

    try:
        client = _get_client()
        models = client.models.list()
        if models.data:
            _last_checked_model = models.data[0].id
        else:
            _last_checked_model = None
    except Exception:
        _last_checked_model = None

    _last_check_time = now
    return _last_checked_model


def _is_embedding_model(model_id: Optional[str]) -> bool:
    """判断当前模型是否为嵌入模型。"""
    if model_id is None:
        return False
    low = model_id.lower()
    return any(kw in low for kw in ("embed", "bge", "bce", "stella", "text2vec", "multilingual-e5"))


def _is_chat_model(model_id: Optional[str]) -> bool:
    """判断当前模型是否为对话模型。"""
    if model_id is None:
        return False
    low = model_id.lower()
    return not _is_embedding_model(model_id)

WELCOME = r"""
╔══════════════════════════════════════════════════════════╗
║           本地 RAG 知识库 — 交互式命令行                    ║
║                                                          ║
║  基于 LM Studio 本地模型，从零手动实现的 RAG 系统            ║
║  输入 /help 查看可用命令，直接输入问题即可开始问答            ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
可用命令:
  /build <目录>     构建知识库——扫描目录中的文档，分块、向量化、存储
                    需要: LM Studio 加载 BGE-M3 嵌入模型
                    示例: /build data/

  /ask <问题>       RAG 问答——检索相关文档块，交给 LLM 生成回答
                    需要: 先 BGE-M3 检索，再 Qwen3.5-9B 生成
                    快捷: 直接输入问题即可，无需 /ask 前缀

  /sources          查看最近一次检索到的参考资料（完整内容）

  /kb <路径>        切换知识库文件
                    示例: /kb my_knowledge_base.json

  /topk <数量>      设置检索返回的文档块数量（默认 5）
                    示例: /topk 10

  /rerank <模式>    设置重排序模式（step8）
                    keyword - 关键词增强重排序（默认，无需 API）
                    llm     - LLM 重排序（需加载对话模型）
                    off     - 关闭重排序

  /history [N]      查看最近 N 轮对话历史（默认 10）
  /clear            清除对话历史

  /eval <测试集>    运行检索质量评估（step8）
                    示例: /eval data/test_questions.json

  /status           查看当前状态

  /models           查看 LM Studio 当前加载的模型

  /help             显示此帮助信息

  /exit, /quit      退出程序

使用提示:
  - LM Studio 同一时间只能加载一个模型到 Server
  - 程序会自动检测当前模型，不对时会暂停等你切换
  - 检索阶段需要嵌入模型 (BGE-M3)，生成阶段需要对话模型 (Qwen3.5-9B)
  - 如果 LLM 调用失败，程序会把提示词打印出来供你手动发送
  - 对话历史会自动维护，支持多轮追问；/clear 可重置
  - 重排序默认开启 (keyword)，检索时自动提升关键词匹配度高的结果
"""


# ================================================================
# 核心功能（与 step6 共用逻辑，但增加了交互式引导）
# ================================================================

def _get_client():
    """创建 API 客户端。"""
    return OpenAI(base_url=BASE_URL, api_key="not-needed")


def _embed(text: str) -> list[float]:
    """调用 BGE-M3 嵌入接口。"""
    client = _get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _call_llm(prompt: str) -> str:
    """调用 Qwen 对话模型。"""
    client = _get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4000,
    )
    msg = response.choices[0].message
    answer = msg.content or getattr(msg, "reasoning_content", None)
    return (answer or str(msg)).strip()


def _build_prompt(question: str, references: list[str]) -> str:
    """拼接 RAG 提示词。"""
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


def _source_prefix(doc_name: str) -> str:
    """给 chunk 加上来源标记。"""
    return f"[来源: {doc_name}]"


# ================================================================
# 交互式 CLI 主类
# ================================================================

class RAGCli:
    """管理交互式会话的状态和行为。"""

    def __init__(self, kb_path: str = "knowledge_base.json", top_k: int = 5):
        self.kb_path = kb_path
        self.top_k = top_k
        self._last_question: str = ""
        self._last_results: list[tuple[str, float]] = []  # 最近一次检索结果
        self.rerank_method = "keyword"  # keyword | llm | off
        self.history = ConversationHistory(max_turns=20)

    # ── 模型切换辅助 ──

    def _ensure_model(self, target: str):
        """确保 LM Studio 加载了指定模型，不对则暂停等用户切换。

        target 取值:
          "embedding"  → 需要嵌入模型（BGE-M3）
          "chat"       → 需要对话模型（Qwen3.5-9B）

        如果是正确的模型，直接过；否则打印提示并暂停等待用户切换后按 Enter。
        """
        current = _get_current_model(force=True)

        if target == "embedding":
            expected_name = EMBEDDING_MODEL
            is_ok = _is_embedding_model(current)
            usage = "做文本向量化（把你的问题转成向量，去知识库里搜索相关文档块）"
        else:
            expected_name = CHAT_MODEL
            is_ok = _is_chat_model(current)
            usage = "做回答生成（读取搜索到的文档块，生成最终答案）"

        if current is None:
            self._print_hint(f"LM Studio 未响应，请确认 Server 已启动 (端口 1234)")
            input(f"  启动后按 Enter 继续...")
            return

        if is_ok:
            print(f"  [模型] 当前加载: {current} ✓")
            return

        # 模型不对 —— 暂停等待切换
        self._print_model_hint(f"当前加载: {current}")
        self._print_model_hint(f"需要切换为: {expected_name}")
        self._print_model_hint(f"用途: {usage}")
        print()
        print(f"  请在 LM Studio 中:")
        print(f"    1. 停止当前模型（点击 Stop 按钮）")
        print(f"    2. 加载 {expected_name}")
        print(f"    3. 确认 Server 已启动")
        input(f"  完成后按 Enter 继续...")
        # 切换后刷新缓存
        _get_current_model(force=True)

    # ── 命令: build ──

    def cmd_build(self, docs_dir: str, chunk_size: int = 512, overlap: int = 128):
        """构建知识库。"""
        if not os.path.isdir(docs_dir):
            self._print_error(f"目录不存在: {docs_dir}")
            return

        self._print_header("构建知识库")
        print(f"  文档目录: {docs_dir}")
        print(f"  输出文件: {self.kb_path}")
        print(f"  分块参数: chunk_size={chunk_size}, overlap={overlap}")
        print()

        # 加载文档
        print("  [1/3] 加载文档...")
        docs = load_documents(docs_dir)
        if not docs:
            self._print_error("没有找到可用的文档，请检查目录路径")
            return
        print(f"  已加载 {len(docs)} 篇文档:")
        for d in docs:
            print(f"    [{d['type']}] {d['name']} ({len(d['content'])} 字符)")

        # 分块 + 向量化 + 存入
        print()
        print("  [2/3] 分块 + 向量化...")
        self._ensure_model("embedding")

        store = VectorStore()
        total_chunks = 0

        for doc in docs:
            if not doc["content"].strip():
                print(f"  [跳过] {doc['name']} (空文档)")
                continue

            chunks = chunk_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
            print(f"  {doc['name']}: {len(chunks)} 个块")

            for chunk in chunks:
                text_with_source = f"{_source_prefix(doc['name'])}\n{chunk}"
                try:
                    vec = _embed(chunk)
                    store.add(text_with_source, vec)
                    total_chunks += 1
                except Exception as e:
                    self._print_error(f"向量化失败: {e}")
                    self._print_hint(
                        f"请确认 LM Studio 已加载 {EMBEDDING_MODEL}，且 API 端口 {BASE_URL} 可访问"
                    )
                    return

        print(f"  共 {total_chunks} 个块已向量化")

        # 持久化
        print()
        print("  [3/3] 持久化...")
        store.save(self.kb_path)
        print(f"  知识库已保存到: {self.kb_path}")
        print(f"  库大小: {len(store)} 个块")
        self._print_header("构建完成!")

    # ── 命令: ask ──

    def cmd_ask(self, question: str):
        """RAG 问答。"""
        self._last_question = question
        self._last_results = []

        self._print_header("RAG 问答")
        print(f"  问题: {question}")
        print()

        # --- 第 1 步: 检索 ---
        print("  [1/3] 检索相关文档块...")
        self._ensure_model("embedding")

        # 加载知识库
        if not os.path.exists(self.kb_path):
            self._print_error(f"知识库文件不存在: {self.kb_path}")
            self._print_hint(f"请先构建知识库: /build <文档目录>")
            return

        store = VectorStore.load(self.kb_path)
        print(f"  已加载知识库: {self.kb_path} ({len(store)} 个块)")

        # 嵌入问题 + 检索（如果启用重排序，先多取一些候选）
        search_k = self.top_k * 3 if self.rerank_method != "off" else self.top_k
        try:
            question_vec = _embed(question)
        except Exception as e:
            self._print_error(f"嵌入失败: {e}")
            self._print_hint(
                f"请确认 LM Studio 已加载 {EMBEDDING_MODEL}，且 API 端口 {BASE_URL} 可访问"
            )
            return

        results = store.search(question_vec, top_k=search_k)
        raw_count = len(results)

        # 重排序
        if self.rerank_method != "off" and len(results) > 1:
            print(f"  向量检索: {raw_count} 个候选块")
            print(f"  重排序:   {self.rerank_method} 模式...")
            results = rerank_results(question, results, method=self.rerank_method)
            # 取 top_k 个
            results = results[: self.top_k]

        self._last_results = results

        print(f"  检索到 {len(results)} 个相关块:")
        print()
        for i, (text, sim) in enumerate(results, 1):
            # 截取前 100 字符作为预览
            clean = text.replace("\n", " ")[:100]
            print(f"  [{i}] (相似度 {sim:.4f}) {clean}...")

        if not results:
            print("  没有找到相关内容，无法进行问答。")
            return

        # --- 第 2 步: 构建提示词 ---
        print()
        print("  [2/3] 构建提示词...")
        references = []
        for i, (text, _) in enumerate(results, 1):
            references.append(f"[资料 {i}]\n{text}")

        # 获取对话历史上下文
        history_context = self.history.get_context_for_prompt(max_context_turns=5)
        prompt = build_prompt_with_history(question, references, history_context)
        print(f"  提示词长度: {len(prompt)} 字符")
        if history_context:
            print(f"  对话历史: {len(self.history)} 轮")

        # --- 第 3 步: 生成回答 ---
        print()
        print("  [3/3] 生成回答...")
        self._ensure_model("chat")
        print()

        try:
            answer = _call_llm(prompt)
            self._print_header("回答")
            print(answer)
            # 记录本轮对话
            self.history.add_turn(question, answer, self._last_results)
        except Exception as e:
            self._print_hint(f"LLM 调用失败: {e}")
            self._print_hint(
                f"请在 LM Studio 中切换到 {CHAT_MODEL}，"
                f"然后将下面的提示词发给模型。"
            )
            print()
            print("=" * 60)
            print("提示词 (复制以下内容发送给 LLM):")
            print("=" * 60)
            print()
            print(prompt)

    # ── 命令: sources ──

    def cmd_sources(self):
        """查看最近一次检索到的参考资料全文。"""
        if not self._last_results:
            self._print_hint("还没有进行过检索。请先提问: /ask <问题>")
            return

        self._print_header("最近一次检索的参考资料")
        print(f"  问题: {self._last_question}")
        print(f"  共 {len(self._last_results)} 条结果:")
        print()

        for i, (text, sim) in enumerate(self._last_results, 1):
            print(f"─── 资料 [{i}] (相似度 {sim:.4f}) ───")
            print(text)
            print()

    # ── 命令: status ──

    def cmd_status(self):
        """查看当前状态。"""
        self._print_header("运行状态")
        current_model = _get_current_model()
        print(f"  当前模型:   {current_model or '(无法获取)'}")
        print(f"  知识库路径: {self.kb_path}")
        print(f"  检索数量:   top_k = {self.top_k}")
        print(f"  重排序:     {self.rerank_method}")
        print(f"  对话轮数:   {len(self.history)}")

        if os.path.exists(self.kb_path):
            try:
                store = VectorStore.load(self.kb_path)
                print(f"  知识库大小: {len(store)} 个文档块")
            except Exception:
                print(f"  知识库大小: (读取失败)")
        else:
            print(f"  知识库状态: 文件不存在，请先 /build")

        if self._last_question:
            print(f"  最近提问:   {self._last_question[:60]}...")
        else:
            print(f"  最近提问:   (无)")

    # ── 响应解析 ──

    def handle_input(self, user_input: str) -> bool:
        """解析并执行用户输入。返回 False 表示退出。"""
        text = user_input.strip()
        if not text:
            return True

        # 命令模式: 以 / 开头
        if text.startswith("/"):
            parts = shlex.split(text)
            if not parts:
                return True
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in ("/exit", "/quit", "/q"):
                return self._cmd_exit(args)

            elif cmd == "/models":
                current = _get_current_model(force=True)
                if current is None:
                    self._print_hint("无法获取当前模型，LM Studio 可能未启动")
                else:
                    tag = "嵌入模型" if _is_embedding_model(current) else "对话模型"
                    print(f"  当前加载: {current} ({tag})")

            elif cmd == "/help":
                print(HELP_TEXT)

            elif cmd == "/build":
                if not args:
                    self._print_hint("用法: /build <文档目录>")
                    self._print_hint("示例: /build data/")
                    return True
                self.cmd_build(args[0])

            elif cmd == "/ask":
                if not args:
                    self._print_hint("用法: /ask <问题>")
                    self._print_hint("也可以直接输入问题，无需 /ask 前缀")
                    return True
                self.cmd_ask(" ".join(args))

            elif cmd == "/sources":
                self.cmd_sources()

            elif cmd == "/kb":
                if not args:
                    self._print_hint(f"当前知识库: {self.kb_path}")
                    self._print_hint("用法: /kb <路径>  切换知识库文件")
                    return True
                new_path = args[0]
                if not os.path.exists(new_path):
                    self._print_hint(f"文件不存在: {new_path}（路径已设置，但可能需要先 /build）")
                self.kb_path = new_path
                self._last_results = []
                print(f"  知识库已切换为: {self.kb_path}")

            elif cmd == "/topk":
                if not args:
                    self._print_hint(f"当前 top_k = {self.top_k}")
                    self._print_hint("用法: /topk <数量>")
                    return True
                try:
                    n = int(args[0])
                    if n < 1:
                        raise ValueError
                    self.top_k = n
                    print(f"  top_k 已设置为: {self.top_k}")
                except ValueError:
                    self._print_error("请输入一个正整数")

            elif cmd == "/status":
                self.cmd_status()

            elif cmd == "/history":
                n = 10
                if args:
                    try:
                        n = int(args[0])
                    except ValueError:
                        self._print_hint("用法: /history [数量]")
                        return True
                print(self.history.format_history_for_display(n))

            elif cmd == "/clear":
                self.history.clear()
                self._last_results = []
                self._last_question = ""
                print("  对话历史已清除")

            elif cmd == "/rerank":
                if not args:
                    self._print_hint(f"当前重排序: {self.rerank_method}")
                    self._print_hint("用法: /rerank keyword|llm|off")
                    return True
                mode = args[0].lower()
                if mode in ("keyword", "llm", "off"):
                    self.rerank_method = mode
                    print(f"  重排序模式已切换为: {mode}")
                else:
                    self._print_error(f"无效模式: {mode}")
                    self._print_hint("可用: keyword, llm, off")

            elif cmd == "/eval":
                from step8_optimizations import EvalRunner

                if not args:
                    self._print_hint("用法: /eval <测试集路径>")
                    self._print_hint("示例: /eval data/test_questions.json")
                    return True
                testset_path = args[0]
                if not os.path.exists(testset_path):
                    self._print_error(f"测试集文件不存在: {testset_path}")
                    return True
                if not os.path.exists(self.kb_path):
                    self._print_error(f"知识库不存在: {self.kb_path}，请先 /build")
                    return True

                self._print_model_hint(f"评估需要嵌入模型: {EMBEDDING_MODEL}")
                try:
                    runner = EvalRunner(
                        self.kb_path, testset_path, _embed, top_k=self.top_k
                    )
                    runner.run()
                    print(runner.generate_report())
                except Exception as e:
                    self._print_error(f"评估失败: {e}")

            else:
                self._print_error(f"未知命令: {cmd}")
                self._print_hint("输入 /help 查看可用命令")

        else:
            # 非命令输入 → 当成问题，直接走 RAG 问答
            self.cmd_ask(text)

        return True

    def run(self):
        """启动交互式主循环。"""
        print(WELCOME)
        self.cmd_status()
        print()
        print("输入 /help 查看帮助，直接输入问题即可开始提问。")
        print()

        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见!")
                break

            if not self.handle_input(user_input):
                break

    # ── 格式化输出辅助函数 ──

    @staticmethod
    def _print_header(title: str):
        """打印格式化标题。"""
        print()
        print("=" * 60)
        print(f"  {title}")
        print("=" * 60)
        print()

    @staticmethod
    def _print_error(msg: str):
        """打印错误消息。"""
        print(f"  [错误] {msg}")

    @staticmethod
    def _print_hint(msg: str):
        """打印提示消息。"""
        print(f"  [提示] {msg}")

    @staticmethod
    def _print_model_hint(msg: str):
        """打印模型切换提示。"""
        print(f"  [模型] {msg}")

    @staticmethod
    def _cmd_exit(args: list[str]) -> bool:
        """处理退出命令。"""
        print("再见!")
        return False


# ================================================================
# 入口
# ================================================================

if __name__ == "__main__":
    kb_path = sys.argv[1] if len(sys.argv) > 1 else "knowledge_base.json"
    cli = RAGCli(kb_path=kb_path)
    cli.run()
