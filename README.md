# 本地 RAG 知识库

从零手动实现 RAG（检索增强生成）知识库系统，基于 LM Studio 本地部署的 LLM 和嵌入模型，逐步深入理解 RAG 原理。

## 技术栈

| 组件 | 选型 |
|------|------|
| 对话模型 | Qwen3.5-9B（LM Studio 本地部署） |
| 嵌入模型 | BGE-M3（中文语义向量） |
| API 客户端 | OpenAI Python SDK（兼容 LM Studio） |
| 本地服务 | LM Studio（`http://127.0.0.1:1234/v1`） |

## 项目结构

```
local-rag-knowledge-base/
├── step1_test.py          # 第1步：验证对话接口
├── step2_embedding.py     # 第2步：嵌入接口 + 余弦相似度
├── step3_chunker.py       # 第3步：文本分块器
├── step4_vectorstore.py   # 第4步：向量库
├── step5_loader.py        # 第5步：文档加载器
├── step6_rag_pipeline.py  # 第6步：RAG 管线串联
├── step7_cli.py           # 第7步：交互命令行界面
├── step8_optimizations.py # 第8步：优化与扩展
├── data/                  # 知识库文档 + 向量数据（不入库）
└── .gitignore
```

## 整体路线图

```
用户提问 ──→ [检索] ──→ 找到相关文档片段 ──→ [增强] ──→ 拼接提示词 ──→ [生成] ──→ LLM 回答
               ↑                                    ↑                        ↑
         第3~5步构建                          第6步串联                  第1步对接
```

### 8 个步骤

| 步骤 | 文件 | 目标 | 状态 |
|------|------|------|------|
| 1 | `step1_test.py` | 验证 LM Studio 对话接口，调用 Qwen3.5-9B 生成回答 | [OK] 已完成 |
| 2 | `step2_embedding.py` | 调用 BGE-M3 嵌入接口，纯 Python 实现余弦相似度，验证中文语义区分能力 | [OK] 已完成 |
| 3 | `step3_chunker.py` | 实现文本分块器——将长文档切成固定大小的片段，保持语义边界 | [OK] 已完成 |
| 4 | `step4_vectorstore.py` | 实现向量库——存储文档块 + 对应向量，支持相似度检索 | [OK] 已完成 |
| 5 | `step5_loader.py` | 实现文档加载器——从文件系统读取 Markdown/TXT/PDF 文档 | [OK] 已完成 |
| 6 | `step6_rag_pipeline.py` | 串联完整管线：加载文档 → 分块 → 向量化 → 检索 → 生成回答 | [OK] 已完成 |
| 7 | `step7_cli.py` | 搭建交互式命令行界面，支持提问和查看检索结果 | [OK] 已完成 |
| 8 | `step8_optimizations.py` | 4 个优化模块——多文档类型、检索质量评估、关键词重排序、多轮对话 | [OK] 已完成 |

### 数据流

```
文档文件               知识库构建（离线）                问答（在线）
───────               ──────────────                  ────────

 Markdown  ──→ [5.文档加载器] ──→ 纯文本               用户提问
  HTML/CSV        │                     │                │
  /JSON ──→ [8.扩展加载器]            ▼                  ▼
                              [3.文本分块器]        [4.向量库检索]
                                    │                     │
                                    ▼                     ▼
                              文档块列表         [8.重排序] → Top-K
                                    │                     │
                                    ▼                     │
                              [2.嵌入接口]                │
                                    │                     │
                                    ▼                     ▼
                              [4.向量库存储]  ──→  [6.RAG管线]
                                                      │
                                              [8.对话历史]
                                                      │
                                                      ▼
                                                 [1.LLM对话]
                                                      │
                                                      ▼
                                                   最终回答
```

## 环境准备

```bash
# 1. 安装依赖
pip install openai

# 2. 启动 LM Studio，加载两个模型：
#    - qwen/qwen3.5-9b（对话）
#    - text-embedding-bge-m3（嵌入）
#    注意：LM Studio 同一时间只能加载一个模型，分步操作时需手动切换

# 3. 验证 API 可用
python step1_test.py
```

## 关键经验

- **Qwen3.5-9B 是推理模型**：对话结果可能存放在 `reasoning_content` 字段而非 `content`，需设置 `max_tokens=4000` 避免截断
- **英文嵌入模型对中文无效**：最初使用 `nomic-embed-text-v1.5`，中文文本的相似度全部挤在 0.73~0.78，无法区分语义相关性。替换为 BGE-M3 后分数立即拉开（跨度 0.22）
- **LM Studio 单模型限制**：同一时间只能加载一个模型到 Server，RAG 管线串联时需要考虑双实例或快速切换方案
