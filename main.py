from openai import OpenAI

client = OpenAI(
    base_url = "http://127.0.0.1:1234/v1",
    api_key = "not-needed"
)

response = client.chat.completions.create(
    model="qwen/qwen3.5-9b",
    messages=[
        {"role": "system", "content": "你是鹤唳的私人工作助手，回答问题时请使用中文。"},
        {"role": "user", "content": "用简单的语言介绍一下什么是RAG"}
    ],
    temperature=0.7,
    max_tokens=4000
)

msg = response.choices[0].message
answer = msg.content or getattr(msg, "reasoning_content", None)
print(answer if answer else str(msg))
