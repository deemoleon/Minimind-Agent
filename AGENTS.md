# AGENTS.md — AI 编程助手上下文

## 项目概述

MiniMind Function Calling Agent 系统。基于 MiniMind3（Qwen3 架构）轻量级大模型，提供 OpenAI 兼容的 Function Calling API 和 ReAct Agent。

## 环境约束

- OS: Windows 10/11
- GPU: AMD RX 9070 GRE（ROCm 7.2.1，通过 HIP 暴露为 CUDA）
- Python: 3.12
- 包管理: pip

## 架构原则

1. **Clean Service Separation**：serve.py 和 agent.py 通过 HTTP 通信，各自独立
2. **Mock 优先**：Mock 模式验证全流程，切换真实模型只改 config.yaml
3. **OpenAI 兼容**：所有对外接口遵循 OpenAI Chat Completions 规范
4. **职责分离**：模型输出原生对象 → serve.py 做格式转换 → agent.py 只接触标准格式

## 关键组件

| 文件 | 职责 | 依赖 |
|------|------|------|
| serve.py | FastAPI 服务，OpenAI 兼容接口，加载模型 | mock_model.py, model_adapter.py, config.yaml |
| agent.py | ReAct 循环，工具调度，对话管理 | memory.py, tools.py, httpx→serve |
| tools.py | 工具注册、JSON Schema 生成、MCP Client | 无 |
| memory.py | ChromaDB 记忆存储与检索 | chromadb |
| mock_model.py | Mock 推理，模拟 Function Calling | 无 |
| model_adapter.py | 真实模型适配器（Qwen3ForCausalLM + AutoConfig） | transformers, torch |
| main.py | CLI/WebUI 双入口 | agent.py, gradio |
| start_serve.bat | 非阻塞启动 serve.py（解决 PowerShell Start-Process 卡住问题） | 无 |

## 关键约定

### API 格式

- 请求/响应严格遵循 OpenAI Chat Completions 格式
- tool_calls 中 `function.arguments` 是 JSON 字符串（非 dict 对象）
- 每个请求生成唯一 `id`：`chatcmpl-{timestamp}-{random}`

### 模型推理接口

```python
def generate(messages: list, tools: list = None) -> str | dict:
    """
    无 tools → 返回 str
    有 tools → 返回 {"name": "函数名", "arguments": {参数dict}}
    """

def stream_generate(messages: list, tools: list = None):
    """
    流式生成，yield chunk:
    - {"type": "chunk", "data": "token"}     # 单个 token
    - {"type": "tool_calls", "data": {...}}   # 工具调用（非流式）
    - {"type": "done", "data": ""}           # 生成结束
    """
```

### 工具注入

工具列表注入到 system prompt 末尾，格式：
```
当前可用工具：
1. tool_name - 描述
   参数: param1 (type, 必填/可选), param2 (type, 必填/可选)
```

### 对话模板

MiniMind 原生格式：
```
<|system|>
系统提示词
</s>
<|user|>
用户输入
</s>
<|assistant|>
模型回复
</s>
```

### 配置管理

所有配置集中在 config.yaml，不硬编码到代码。包括：
- model.mock（Mock 开关）
- model.model_path（模型目录，含 config.json + tokenizer）
- memory.backend（chromadb/milvus）
- agent.max_rounds（最大 ReAct 循环次数）

### 流式输出（SSE）

serve.py 根据 `stream` 参数返回不同格式：
- `stream=False`：返回完整 JSON 响应
- `stream=True`：返回 `StreamingResponse`，逐 token 推送 SSE 数据

内部 chunk 格式（agent ↔ main.py）：
- `{"type": "chunk", "data": "token"}` — 单个 token
- `{"type": "tool_calls", "data": {...}}` — 工具调用（非流式）
- `{"type": "done", "data": ""}` — 生成结束

### 错误处理

- 模型未加载 → 503
- 请求格式错误 → 422
- 推理超时 → 504
- 工具调用失败 → 重试 retry_count 次

## 当前状态

- Mock 模式全流程验证通过（test_api.py 8/8）
- 真实模型验证通过（test_api.py --real 10/11）
- SSE 流式输出：Mock 模型 → serve.py SSE → agent 流式接收 → CLI/WebUI 逐字显示
- model_adapter.py 使用官方 Qwen3ForCausalLM（来自 HuggingFace transformers）
- 模型文件位于 models/minimind-fc/（config.json 从 jingyaogong/minimind-3 拉取）
- ROCm 7.2.1 已安装，AMD RX 9070 GRE GPU 识别正常

## 后续开发注意事项

1. 修改 serve.py 时保持 OpenAI 接口兼容性
2. 新增工具只需在 tools.py 加 @tool 装饰器
3. 切换 Milvus 时实现 memory.py 的 MilvusBackend 类
4. 所有 .bat 文件保持 chcp 65001 开头
5. 启动 serve.py 用 start_serve.bat（PowerShell Start-Process 会卡住）
