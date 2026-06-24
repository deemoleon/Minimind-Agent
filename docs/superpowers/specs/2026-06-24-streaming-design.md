# 流式输出功能设计

> 为 MiniMind Agent 全链路添加 SSE 流式输出支持。

**状态：✅ 已实现** — 2026-06-24

## 需求

- Mock 模式下模拟逐 token 输出（50ms 延迟）
- serve.py 实现 OpenAI SSE 协议
- agent.py 流式接收并 yield
- CLI 逐字符打印，WebUI Gradio 逐字显示
- tool_calls 不流式，一次性返回
- 向后兼容：`stream=False` 时行为不变

## 技术方案

- 协议：SSE（Server-Sent Events），OpenAI 标准
- Mock 模型：`stream_generate()` 方法，逐字符 yield + time.sleep(0.05)
- serve.py：`StreamingResponse` + `text/event-stream`
- agent.py：新增 `chat_stream()` 生成器方法
- main.py：CLI 用 `print(end="", flush=True)`，WebUI 用 Gradio generator

## 修改文件

| 文件 | 改动 |
|------|------|
| mock_model.py | 新增 `stream_generate()` 方法 |
| serve.py | 新增 SSE 响应分支 + `stream_chat()` 生成器 |
| agent.py | 新增 `chat_stream()` + `_call_model_stream()` |
| main.py | CLI/WebUI 改用流式输出 |

## SSE 格式

```
data: {"id":"chatcmpl-xxx","model":"minimind","choices":[{"index":0,"delta":{"role":"assistant","content":"你"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","model":"minimind","choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}

data: [DONE]

```

## 内部 chunk 格式（agent ↔ main.py）

- `{"type": "chunk", "data": "你"}` — 单个 token
- `{"type": "tool_calls", "data": {...}}` — 工具调用（非流式）
- `{"type": "done", "data": ""}` — 生成结束
