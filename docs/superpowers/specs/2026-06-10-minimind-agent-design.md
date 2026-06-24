# MiniMind Agent 系统设计文档

## 1. 概述

为 MiniMind 模型开发 Windows + AMD GPU 环境下的部署和 Agent 系统。当前开发阶段使用 Mock 模式（随机权重/极小模型）验证全流程，后续替换模型权重即可上线。

### 1.1 环境约束

- OS: Windows 10/11
- GPU: AMD（DirectML 后端）
- Python: 3.10+
- 包管理: pip

### 1.2 核心约束

- 当前没有现成模型权重，开发阶段必须可独立验证
- 必须支持 Mock 模式：用随机权重或极小模型验证全流程
- 模型路径通过 config.yaml 配置，后续替换权重文件即可上线
- 必须提供 OpenAI 兼容 API（/v1/chat/completions）

## 2. 架构设计

### 2.1 架构方案：Clean Service Separation

```
┌─────────────────────────────────────────────────┐
│                  main.py                         │
│  --cli → agent CLI 模式                         │
│  --webui → Gradio WebUI                         │
└──────────┬──────────────────┬───────────────────┘
           │                  │
           ▼                  ▼
┌──────────────────┐  ┌──────────────────────────┐
│   agent.py       │  │   serve.py (FastAPI)      │
│  - ReAct 循环    │──│  - POST /v1/chat/completions│
│  - 工具调度      │  │  - GET /health            │
│  - 记忆管理      │  │  - GET /v1/models         │
│  - 子 Agent 委派 │  │  - 推理后端抽象层          │
└───────┬──────────┘  └──────────┬───────────────┘
        │                        │
        ▼                        ▼
┌───────────────┐  ┌──────────────────────────┐
│ tools.py      │  │ mock_model.py            │
│ - 装饰器注册  │  │ - MockModel.generate()   │
│ - JSON Schema │  │ - 微型 GPT (2层/128dim)  │
│ - MCP Client  │  │ - Function Calling JSON  │
└───────────────┘  └──────────────────────────┘
        │
        ▼
┌───────────────┐
│ memory.py     │
│ - ChromaDB    │
│ - Milvus预留  │
└───────────────┘
```

### 2.2 数据流

1. 用户输入 → agent.py ReAct 循环
2. agent.py 先从 memory.py 检索相关历史 → 拼入 messages
3. agent.py 构造 OpenAI messages → POST 到 localhost:8000/v1/chat/completions
4. serve.py 接收请求 → 转换为 MiniMind 格式 → 调用推理后端
5. 推理后端返回文本 → serve.py 解析是否含 tool_calls → 返回 OpenAI 格式
6. agent.py 收到响应 → 如有 tool_calls 则执行工具 → 将结果加入对话上下文 → 同时写入 memory.py（对话历史持久化）→ 回到步骤 2
7. 无 tool_calls 时返回最终答案

## 3. 组件设计

### 3.1 config.yaml — 全局配置

```yaml
model:
  mock: true                    # Mock 模式开关
  model_path: ""                # 真实模型权重路径（后续填写）
  backend: "mock"               # mock | directml | onnx | cpu
  max_tokens: 2048
  temperature: 0.7

server:
  host: "0.0.0.0"
  port: 8000

memory:
  backend: "chromadb"           # chromadb | milvus
  persist_dir: "./data/memory"
  collection_name: "minimind_agent"
  embedding_model: "all-MiniLM-L6-v2"

agent:
  max_rounds: 10                # ReAct 最大循环次数
  retry_count: 3                # 工具调用失败重试次数
  system_prompt: "你是 MiniMind Agent..."

tools:
  mcp_servers: []               # 外部 MCP Server 地址
  enabled: ["get_time", "read_file", "run_shell", "web_search"]
```

### 3.2 mock_model.py — Mock 模型

#### 接口设计

```python
class MockModel:
    def __init__(self, config: dict):
        """加载配置，初始化微型 GPT 结构"""
        pass

    def generate(self, messages: list, tools: list = None) -> str | dict:
        """
        核心推理接口

        返回值：
        - 无 tools：返回 str（纯文本）
        - 有 tools：返回 dict，如 {"name": "get_time", "arguments": {}}
          arguments 是 Python dict 对象，不是 JSON 字符串
        """
        pass
```

#### 内部结构

- 2 层 Transformer，128 hidden dim，CPU 可运行
- 无 tool 定义时：返回预设的中文回复模板
- 有 tool 定义时：根据用户输入关键词匹配工具，返回合法 tool_calls dict

#### Mock 行为

- 无 tools → 返回预设中文回复
- 有 tools → 根据关键词匹配，返回 `{"name": "xxx", "arguments": {}}`
- 不依赖 GPU

### 3.3 serve.py — OpenAI 兼容推理服务

#### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回模型加载状态 |
| GET | `/v1/models` | 返回模型列表（含 id、owned_by） |
| POST | `/v1/chat/completions` | 核心推理端点 |
| POST | `/v1/completions` | 可选，纯文本补全 |

#### POST /v1/chat/completions 核心逻辑

**输入处理**：
- OpenAI messages[] → MiniMind 对话模板
- system 消息 → 系统提示词
- user 消息 → 用户输入
- assistant 消息 → 历史回复
- tool 消息 → 工具调用结果（转为 `<|user|>` 格式）

**工具定义转换**：
- OpenAI tools[] JSON Schema → 追加到 system prompt 末尾
- 使用 MiniMind 原生 `<|system|>` 模板，不搞特殊标记
- 格式示例：
  ```
  <|system|>
  你是一个函数调用助手...
  当前可用工具：
  1. get_time - 获取当前时间
     参数: timezone (string, 可选)
  2. read_file - 读取文件
     参数: path (string, 必填)
  </s>
  ```

**tool 消息处理**：
- `role: "tool"` 转为 `<|user|>` 格式：`工具 {name} 的执行结果：{content}`

**推理调用**：
- model.generate(messages, tools) → 原始输出

**输出转换**：
- 原始输出 → OpenAI ChatCompletion 格式
- 纯文本 → `{"choices": [{"message": {"content": "..."}}]}`
- tool_calls → `{"choices": [{"message": {"tool_calls": [...]}}]}`
- arguments dict → `json.dumps()` → JSON 字符串

**流式支持**（已实现）：
- Mock 模式下 `stream_generate()` 逐字符 yield，50ms 延迟模拟
- serve.py 根据 `stream` 参数返回 `StreamingResponse`（SSE）
- agent.py 新增 `chat_stream()` 生成器，CLI 逐字符打印，WebUI 逐字显示
- tool_calls 不流式，一次性返回
- 向后兼容：`stream=False` 时行为不变

#### 错误处理

- 模型未加载 → 503 Service Unavailable
- 请求格式错误 → 422 Validation Error
- 推理超时 → 504 Gateway Timeout

### 3.4 tools.py — 工具注册系统

#### 核心机制

```python
@tool(description="获取当前时间", parameters={"timezone": {"type": "string"}})
def get_time(timezone: str = "Asia/Shanghai") -> str:
    return datetime.now().isoformat()
```

- 装饰器自动注册到 `tool_registry: Dict[str, ToolDef]`
- 自动生成 JSON Schema

#### 预置工具

| 工具 | 功能 | 参数 | Mock 行为 |
|------|------|------|-----------|
| `get_time` | 获取当前时间 | timezone: string (可选) | 返回固定时间戳 |
| `read_file` | 读取文件内容 | path: string (必填) | 返回 "Mock file content: xxx" |
| `run_shell` | 执行 Shell 命令 | command: string (必填) | 返回 "Mock command output" |
| `web_search` | 网页搜索 | query: string (必填), num_results: int (可选, 默认5) | 返回模拟搜索结果 |

#### Mock 返回格式

Mock 模式下所有工具返回包含 `_mock: true` 标记：

```json
{"_mock": true, "result": "模拟数据...", "datetime": "2024-01-01T12:00:00"}
```

#### MCP 混合模式

- 内置工具：装饰器注册
- MCP Client：连接外部 MCP Server
  - `list_tools()` → 合并到工具列表
  - `call_tool(name, args)` → 调用远程工具
- `requires_confirm` 机制：危险工具（如 run_shell）需用户确认

### 3.5 agent.py — ReAct Agent

#### ReAct 循环

```python
while round < max_rounds:
    1. memory.search(query=current_input) → 相关历史
    2. 构造 messages = [system + history + current]
    3. POST /v1/chat/completions → response
    4. if response.tool_calls:
         for each tool_call:
           a. 检查 requires_confirm:
              - CLI 模式：input("确认执行? (y/n)")
              - WebUI 模式：通过 Gradio 弹窗确认
              - 用户拒绝 → skip 该工具，继续下一个 tool_call
              - 确认后 → 执行工具，写入 memory，回到步骤 1
           b. tools.execute(name, args) → result
           c. messages.append(tool_result)
           d. memory.store(tool_call + result)
         continue
    else:
         return response.content
```

#### 记忆管理

```python
class MemoryManager:
    def __init__(self, config):
        """初始化 ChromaDB 连接"""
        pass

    def store(self, conversation: dict):
        """存储对话片段"""
        pass

    def search(self, query: str, top_k: int = 5) -> list:
        """语义检索相关历史"""
        pass

    def get_history(self, session_id: str) -> list:
        """获取会话历史"""
        pass
```

- ChromaDB 优先：嵌入式，无需外部服务
- Milvus 预留接口：同 API，不同实现

#### 子 Agent 委派

- Mock 阶段**接口预留**
- 有 `create_sub_agent()` 方法但只用主 Agent
- 子 Agent：独立 system_prompt + 工具子集 + 独立 ReAct 循环

#### 错误重试

- 工具调用失败 → 重试 `retry_count` 次
- 模型返回格式错误 → 自动修正重试
- 超过 max_rounds → 返回当前结果 + 警告

### 3.6 main.py — 入口

```python
# 双模式入口
python main.py --cli      # CLI 交互模式
python main.py --webui    # Gradio WebUI 模式

# 启动前检查
1. 加载 config.yaml
2. 检查 /health 端点是否可达
3. 不可达 → 提示启动 serve.py
4. 可达 → 启动对应模式
```

#### Gradio WebUI

```
┌──────────────────────────────────────┐
│  MiniMind Agent                      │
├──────────────────────────────────────┤
│  [对话历史区域]                       │
│  用户: xxx                           │
│  Agent: xxx                          │
│  [工具调用详情可展开]                  │
├──────────────────────────────────────┤
│  [输入框]                    [发送]   │
├──────────────────────────────────────┤
│  状态: 已连接 | 模型: mock | 记忆: chromadb │
└──────────────────────────────────────┘
```

- 状态栏从 `/health` 接口动态读取模型信息

### 3.7 test_api.py — 测试脚本

| # | 测试 | 验证 |
|---|------|------|
| 1 | `GET /health` | 状态码 200，返回 `{"status": "ok"}` |
| 2 | `GET /v1/models` | 返回模型列表，含 `id: "minimind"` |
| 3 | `POST /v1/chat/completions` (纯文本) | 返回 `choices[0].message.content` |
| 4 | `POST /v1/chat/completions` (含 tools) | 返回 `tool_calls` 格式 |
| 5 | `POST /v1/chat/completions` (tool 结果) | 模型基于工具结果返回最终文本，`finish_reason="stop"` |
| 6 | 错误请求 | 返回 422 |
| 7 | 模型未加载 | 返回 503 |

用 `openai` 库连接 `localhost:8000`，每个测试独立，最后汇总通过率。

### 3.8 启动脚本

| 脚本 | 功能 |
|------|------|
| `install.bat` | pip install 依赖，chcp 65001 支持中文 |
| `start_server.bat` | 启动 serve.py |
| `start_agent.bat` | 启动 main.py --webui |
| `start_all.bat` | 先启 serve，健康检查轮询等待就绪，再启 agent |

`start_all.bat` 使用健康检查轮询：
```batch
:wait
timeout /t 1 >nul
curl -s http://localhost:8000/health >nul
if %errorlevel% neq 0 goto wait
```

## 4. 技术栈总结

| 层级 | 方案 |
|------|------|
| 推理后端 | Mock（当前）→ DirectML → ONNX Runtime → CPU fallback |
| API 框架 | FastAPI + uvicorn |
| API 格式 | OpenAI Chat Completions 兼容（含 tool_calls） |
| Agent 框架 | 自建 ReAct 循环 |
| 记忆存储 | ChromaDB（优先）→ Milvus（预留） |
| 工具协议 | MCP 混合模式（内置 + 外部 MCP Server） |
| 前端界面 | Gradio WebUI |
| 配置管理 | config.yaml |

## 5. 验收标准

1. test_api.py 全部通过（7/7）
2. Gradio 界面可进行多轮工具调用对话
3. 后续替换模型权重，改一行配置（config.yaml 中 model.mock: false + model.model_path）即可切换
