import os
from pypdf import PdfReader


def strip_frontmatter(text: str) -> str:
    """去掉 Markdown 文件开头的 YAML frontmatter（--- ... ---）。

    只有文件开头的 "---" 才算 frontmatter——正文里的 "---"（水平分割线）
    不动。如果找不到配对的两个 "---"，原样返回。
    """
    if not text:
        return ""

    lines = text.split("\n")

    # 第一行必须是 "---"（允许末尾有空格）
    if not lines[0].strip() == "---":
        return text

    # 从第二行开始找配对的 "---"
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            # 找到配对的 ---，返回之后的内容
            remaining = lines[i + 1:]
            return "\n".join(remaining).strip()

    # 只有一个 ---，格式不完整，原样返回
    return text


def load_single(filepath: str) -> dict | None:
    """加载单个文件，根据后缀名分发到不同的读取逻辑。

    返回：
        {"path": ..., "name": ..., "type": "txt"|"md"|"pdf", "content": ...}
        如果格式不支持或读取失败，返回 None。
    """
    if not os.path.isfile(filepath):
        print(f"  [跳过] 文件不存在: {filepath}")
        return None

    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    try:
        if ext == ".txt":
            content = _read_text_file(filepath)
            return {
                "path": filepath,
                "name": os.path.basename(filepath),
                "type": "txt",
                "content": content,
            }

        elif ext == ".md":
            raw = _read_text_file(filepath)
            content = strip_frontmatter(raw)
            return {
                "path": filepath,
                "name": os.path.basename(filepath),
                "type": "md",
                "content": content,
            }

        elif ext == ".pdf":
            content = _read_pdf_file(filepath)
            return {
                "path": filepath,
                "name": os.path.basename(filepath),
                "type": "pdf",
                "content": content,
            }

        else:
            print(f"  [跳过] 不支持的格式 ({ext}): {filepath}")
            return None

    except Exception as e:
        print(f"  [跳过] 读取失败: {filepath} ({e})")
        return None


def load_documents(directory: str) -> list[dict]:
    """扫描目录，加载所有支持的文档。

    只扫当前目录，不递归子目录。返回文档列表，每个元素是一个 dict。
    单个文件失败不会影响其他文件——打印警告，跳过，继续。
    """
    if not os.path.isdir(directory):
        print(f"[错误] 目录不存在: {directory}")
        return []

    supported = {".txt", ".md", ".pdf"}
    results = []

    entries = sorted(os.listdir(directory))

    for entry in entries:
        filepath = os.path.join(directory, entry)
        if not os.path.isfile(filepath):
            continue

        _, ext = os.path.splitext(entry)
        ext = ext.lower()

        if ext not in supported:
            print(f"  [跳过] 不支持的格式 ({ext}): {entry}")
            continue

        doc = load_single(filepath)
        if doc is not None:
            results.append(doc)

    return results


# ============================================================
# 内部辅助函数
# ============================================================

def _read_text_file(filepath: str) -> str:
    """读取文本文件，utf-8 优先，失败回退 gbk。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="gbk") as f:
            return f.read()


def _read_pdf_file(filepath: str) -> str:
    """从 PDF 逐页提取文字，页之间用两个换行分隔。"""
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


# ============================================================
# 测试（纯本地，不需要 LM Studio）
# ============================================================
if __name__ == "__main__":
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="step5_test_")
    print(f"测试临时目录: {tmpdir}\n")

    def _make(path, content):
        """在临时目录下创建文件。"""
        full = os.path.join(tmpdir, path)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

    # --- 测试 1: 加载 .txt 文件 ---
    print("=" * 60)
    print("测试 1: 加载 .txt 文件")
    _make("readme.txt", "这是一段纯文本内容。\n第二行。")
    docs = load_documents(tmpdir)
    assert len(docs) == 1, f"应该加载 1 个文档，实际 {len(docs)}"
    assert docs[0]["type"] == "txt"
    assert docs[0]["name"] == "readme.txt"
    assert "纯文本内容" in docs[0]["content"]
    print(f"  通过 → {docs[0]['name']} ({docs[0]['type']}), {len(docs[0]['content'])} 字符")

    # 清理
    os.remove(os.path.join(tmpdir, "readme.txt"))

    # --- 测试 2: 加载 .md 文件（有 frontmatter） ---
    print("\n测试 2: 加载 .md 文件（有 frontmatter）")
    _make("note.md", """---
title: 测试文档
date: 2026-01-01
---

# 正文标题

这是正文内容。

## 第二节

正文中用 --- 做分割线是可以的。""")
    docs = load_documents(tmpdir)
    assert len(docs) == 1
    assert docs[0]["type"] == "md"
    content = docs[0]["content"]
    assert "title:" not in content, "frontmatter 应该被剥离"
    assert "# 正文标题" in content, "正文标题应保留"
    assert "正文中用 --- 做分割线是可以的" in content, "正文中的 --- 应保留"
    print(f"  通过 → frontmatter 已剥离，正文中的 --- 保留")

    os.remove(os.path.join(tmpdir, "note.md"))

    # --- 测试 3: .md 文件（无 frontmatter） ---
    print("\n测试 3: .md 文件（无 frontmatter）")
    _make("plain.md", "# 直接标题\n\n没有 frontmatter 的文档。")
    docs = load_documents(tmpdir)
    assert len(docs) == 1
    content = docs[0]["content"]
    assert "# 直接标题" in content, "无 frontmatter 时应原样保留"
    print(f"  通过 → 无 frontmatter 的 md 原样返回")

    os.remove(os.path.join(tmpdir, "plain.md"))

    # --- 测试 4: strip_frontmatter 边界情况 ---
    print("\n测试 4: strip_frontmatter 边界情况")
    # 空文本
    assert strip_frontmatter("") == ""
    # 不以 --- 开头
    assert strip_frontmatter("hello\nworld") == "hello\nworld"
    # 只有一个 ---
    assert strip_frontmatter("---\n只有开头，没有结尾") == "---\n只有开头，没有结尾"
    # 空 frontmatter
    result = strip_frontmatter("---\n---\n正文")
    assert result == "正文", f"空 frontmatter 应返回正文，实际: {result!r}"
    print(f"  通过 → 空文本、无 ---、单个 ---、空 frontmatter 全部正确")

    # --- 测试 5: 混合目录 ---
    print("\n测试 5: 混合目录（txt + md + 不支持格式）")
    _make("a.txt", "txt content")
    _make("b.md", "# md content")
    _make("c.docx", "word content")  # 不支持
    _make("d.pdf", "")  # 空 PDF 无法测试，但文件存在会触发 PdfReader
    # 用一个真实的临时 PDF 不太好构造，这里只验证不支持格式被跳过
    docs = load_documents(tmpdir)
    types = [d["type"] for d in docs]
    assert "txt" in types, "应包含 txt"
    assert "md" in types, "应包含 md"
    # docx 不应出现
    names = [d["name"] for d in docs]
    assert "c.docx" not in names, "docx 应被跳过"
    print(f"  通过 → 加载了 {len(docs)} 个文档 ({sorted(types)}), docx 被跳过")

    # --- 测试 6: 空目录 ---
    print("\n测试 6: 空目录")
    # 把临时目录清空
    for f in os.listdir(tmpdir):
        fp = os.path.join(tmpdir, f)
        if os.path.isfile(fp):
            os.remove(fp)
    docs = load_documents(tmpdir)
    assert docs == [], f"空目录应返回 []，实际: {docs}"
    print(f"  通过 → 空目录返回 []")

    # --- 测试 7: 文件不存在 ---
    print("\n测试 7: 文件不存在")
    result = load_single(os.path.join(tmpdir, "not_exist.txt"))
    assert result is None, f"不存在的文件应返回 None"
    print(f"  通过 → 不存在的文件返回 None")

    # 清理
    shutil.rmtree(tmpdir)

    print("\n" + "=" * 60)
    print("7 个测试全部通过")
