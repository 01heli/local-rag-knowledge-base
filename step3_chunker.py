import re


def count_tokens_approx(text: str) -> int:
    """粗略估算文本的 token 数量。

    中文：1 个汉字 ≈ 1 个 token
    英文：1 个单词 ≈ 1 个 token
    标点符号忽略不计，作为一个大致估算已足够用于分块决策。
    """
    # 统计中文字符（CJK 统一表意文字）
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    # 统计英文单词（连续字母序列）
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    # 数字序列也算 token
    numbers = len(re.findall(r'\d+', text))

    return chinese_chars + english_words + numbers


def find_split_point(text: str, target: int) -> int:
    """在 target 位置附近找到最自然的语义断点。

    从 target 位置往前搜索，找到第一个自然的句子边界。
    优先级：句号/问号/感叹号 > 换行 > 逗号/分号 > 空格

    返回：实际切分位置（切在断点字符之后），如果找不到自然断点则返回 target 本身。
    """
    # 搜索范围：从 target 往前最多找 target 的一半距离
    search_start = max(0, target - target // 2)

    # 在搜索窗口内截取文本
    window = text[search_start:target]

    # 按优先级找断点——从高到低
    for pattern in [
        r'[。！？]\s*',        # 优先级 1：中文句号、感叹号、问号
        r'\n\s*\n',           # 优先级 2：空行（段落边界）
        r'\n',                # 优先级 3：单换行
        r'[，、；：]\s*',      # 优先级 4：中文逗号、顿号、分号、冒号
        r'\.\s+',             # 优先级 5：英文句号 + 空格
    ]:
        matches = list(re.finditer(pattern, window))
        if matches:
            # 取最后一个匹配（离 target 最近的断点）
            last_match = matches[-1]
            return search_start + last_match.end()

    # 找不到任何自然断点 → 在 target 处硬切
    return target


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> list[str]:
    """将长文本切分成有重叠的片段。

    参数：
        text: 待切分的文本
        chunk_size: 每个块的最大 token 数（近似），默认 512
        overlap: 相邻块之间的重叠 token 数（近似），默认 128

    返回：
        文本块列表。如果输入文本为空，返回空列表。
        如果文本长度不足 chunk_size，返回只包含一个元素的列表。
    """
    if not text.strip():
        return []

    total_tokens = count_tokens_approx(text)

    # 文本太短，不需要切分
    if total_tokens <= chunk_size:
        return [text.strip()]

    chunks = []
    pos = 0  # 当前起始位置（字符索引，非 token 索引）

    while pos < len(text):
        # 计算当前块的结束位置：从 pos 开始往后数 chunk_size 个 token
        end = _advance_by_tokens(text, pos, chunk_size)

        # 到头了
        if end >= len(text):
            remaining = text[pos:].strip()
            if remaining:
                chunks.append(remaining)
            break

        # 在 end 附近找自然断点
        split = find_split_point(text, end)

        # 如果 split 没有前进（断点 <= pos），说明遇到了超长句子，退化为硬切
        if split <= pos:
            split = end

        # 如果切出来的块太小（不足 overlap），往后推到下一个断点
        if split < len(text):
            chunk_tokens = count_tokens_approx(text[pos:split])
            if chunk_tokens < overlap and chunk_tokens > 0:
                # 从当前 split 继续往后找一个更大的断点
                extended_end = _advance_by_tokens(text, split, overlap - chunk_tokens + 10)
                extended_split = find_split_point(text, min(extended_end, len(text)))
                if extended_split > split:
                    split = extended_split

        chunk = text[pos:split].strip()
        if chunk:
            chunks.append(chunk)

        # 下一个块的起始位置 = 当前切分点 - 重叠区域
        next_pos = _rewind_by_tokens(text, split, overlap)
        # 确保位置有前进，否则死循环
        if next_pos <= pos:
            next_pos = split
        pos = next_pos

    return chunks


def _advance_by_tokens(text: str, start: int, token_count: int) -> int:
    """从 start 位置向后推进约 token_count 个 token，返回字符索引。

    这是一个辅助函数，用来把"token 数"近似映射到"字符索引"。
    """
    pos = start
    counted = 0

    while pos < len(text) and counted < token_count:
        char = text[pos]

        # 中文字符 → 1 token
        if '一' <= char <= '鿿':
            counted += 1
        # 英文字母 → 按单词计，跳过连续字母
        elif char.isalpha() and char.isascii():
            word_start = pos
            while pos < len(text) and text[pos].isalpha() and text[pos].isascii():
                pos += 1
            counted += 1
            continue  # 已经移动了 pos，跳过最后的 pos += 1
        # 数字序列 → 1 token
        elif char.isdigit():
            while pos < len(text) and text[pos].isdigit():
                pos += 1
            counted += 1
            continue
        # 空白和标点不计入 token
        else:
            pass

        pos += 1

    return pos


def _rewind_by_tokens(text: str, end: int, token_count: int) -> int:
    """从 end 位置向前回退约 token_count 个 token，返回字符索引。

    与 _advance_by_tokens 方向相反，用来计算重叠区域的起点。
    """
    pos = end
    counted = 0

    while pos > 0 and counted < token_count:
        pos -= 1
        char = text[pos]

        if '一' <= char <= '鿿':
            counted += 1
        elif char.isalpha() and char.isascii():
            # 回退到单词开头
            while pos > 0 and text[pos - 1].isalpha() and text[pos - 1].isascii():
                pos -= 1
            counted += 1
        elif char.isdigit():
            while pos > 0 and text[pos - 1].isdigit():
                pos -= 1
            counted += 1

    return pos


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    sample = (
        "RAG（Retrieval-Augmented Generation，检索增强生成）是一种让大语言模型"
        "在回答问题时能够参考外部知识的技术。它的核心思想很简单：大模型的知识是"
        "训练时固定的，无法知道训练截止日期之后的事情，也无法访问私有文档。"
        "RAG 通过在大模型生成答案之前，先从外部知识库中检索相关文档片段，"
        '然后将这些片段作为"参考资料"一起送给大模型，从而让大模型能够基于'
        "这些资料来生成更准确的回答。\n\n"
        "RAG 的工作流程可以分为两个阶段。第一个阶段是知识库构建，也叫离线阶段。"
        "在这个阶段，我们需要把准备好的文档（Markdown、PDF、TXT 等）加载进来，"
        "切成小块，然后用嵌入模型把每个小块转成向量，存入向量库。"
        "第二个阶段是问答阶段，也叫在线阶段。用户提问后，系统先把问题转成向量，"
        "在向量库中找到最相似的几个文档块，然后把问题和这些文档块拼接成提示词，"
        "送给大模型生成最终的回答。"
    )

    print("=" * 60)
    print(f"原文长度: {len(sample)} 字符, 估算 token: {count_tokens_approx(sample)}")
    print("=" * 60)

    chunks = chunk_text(sample, chunk_size=80, overlap=20)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} (估算 {count_tokens_approx(chunk)} tokens, 实际 {len(chunk)} 字符) ---")
        print(chunk)

    print(f"\n{'=' * 60}")
    print(f"共 {len(chunks)} 个 chunk")

    # 测试边界情况
    print(f"\n空文本: {chunk_text('')}")
    print(f"短文本: {len(chunk_text('一句话搞定。', chunk_size=512))} 个 chunk")
