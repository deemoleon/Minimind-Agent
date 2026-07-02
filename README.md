# MiniMind Function Calling Agent

基于 MiniMind3（Qwen3 架构）轻量级大模型的 Function Calling Agent 系统，支持 Windows + AMD GPU（ROCm 7.2.1）。

## 功能特性

- **ReAct Agent**：思考-行动-观察循环，支持多轮工具调用
- **Function Calling**：模型自主选择和调用工具，严格遵循 OpenAI tool_calls 格式
- **长期记忆**：ChromaDB 向量存储，跨对话上下文检索
- **OpenAI 兼容 API**：`/v1/chat/completions` 标准接口
- **MCP 混合模式**：内置工具 + 可扩展外部 MCP Server
- **Gradio WebUI**：可视化聊天界面，工具调用过程实时展示
- **AMD GPU 原生支持**：ROCm 7.2.1，AMD RX 9070 GRE 已验证

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# Mock 模式（默认，无需 GPU）
start_serve.bat
python test_api.py

# 真实模型模式
# 修改 config.yaml: mock: false
start_serve.bat
python test_api.py --real
```

### 3. 启动 Agent

```bash
python main.py          # WebUI（Gradio，端口 7860）
python main.py --cli    # CLI 模式
```

## 模型架构

- **基座模型**：MiniMind3（基于 Qwen3ForCausalLM）
- **参数量**：68.8M（hidden_size=768, num_layers=8, vocab_size=6400）
- **加载方式**：`AutoConfig.from_pretrained()` + `AutoModelForCausalLM.from_config()` + `load_state_dict()`
- **官方仓库**：HuggingFace `jingyaogong/minimind-3`

## 接入真实模型

### 核心接口

`model_adapter.py` 的 `RealModel` 类对外暴露与 `MockModel` 完全一致的接口：

```python
class RealModel:
    def __init__(self, config: dict)     # config 来自 config.yaml
    def generate(messages, tools)        # 返回 str（纯文本）或 dict（工具调用）
    def stream_generate(messages, tools) # 逐 token yield
    def is_loaded -> bool                # 模型是否加载成功
```

serve.py 不关心模型内部实现，只调用这两个方法。**只要你的模型能实现 `generate()` 和 `stream_generate()`，就能接入本系统。**

---

### 场景 1：接入 .pth 权重（MiniMind3 示例）

适用于 PyTorch 训练产出的 `.pth` 文件。

**Step 1：准备文件**

```bash
# 从 HuggingFace 下载
git clone https://huggingface.co/jingyaogong/minimind-3 models/minimind-fc
```

目录结构：

```
models/minimind-fc/
├── config.json              # 架构配置（model_type: "qwen3"）
├── full_sft_768.pth         # 训练权重（137MB）
├── tokenizer.json           # 分词器
└── tokenizer_config.json    # 分词器配置
```

**关键**：`config.json` 必须与 `.pth` 权重匹配。查看 `config.json` 中的 `model_type` 确认架构：

```json
{
  "model_type": "qwen3",                          // ← 架构类型
  "architectures": ["Qwen3ForCausalLM"],          // ← 模型类名
  "hidden_size": 768,
  "num_hidden_layers": 8
}
```

**Step 2：修改 config.yaml**

```yaml
model:
  mock: false
  model_path: "models/minimind-fc/"   # 指向包含 config.json + .pth 的目录
```

**Step 3：启动**

```bash
start_serve.bat
python test_api.py --real
```

`model_adapter.py` 的加载流程（`_load_pth_model()`，共 4 步）：

```
config.json → AutoConfig.from_pretrained()         → Qwen3Config
Qwen3Config → AutoModelForCausalLM.from_config()    → 空模型结构
full_sft_768.pth → load_state_dict()                → 填充权重
tokenizer.json → AutoTokenizer.from_pretrained()    → 分词器
```

---

### 场景 2：接入 HuggingFace 格式模型

适用于 `from_pretrained()` 标准格式（`model.safetensors` 或 `pytorch_model.bin`）。

**Step 1：准备文件**

```
models/my-model/
├── config.json              # 架构配置
├── model.safetensors        # 模型权重（或 pytorch_model.bin）
├── tokenizer.json           # 分词器
└── tokenizer_config.json    # 分词器配置
```

**Step 2：修改 config.yaml**

```yaml
model:
  mock: false
  model_path: "models/my-model/"   # 指向 HF 格式目录
```

**Step 3：代码不需要改**

`model_adapter.py` 自动识别：目录内没有 `.pth` 文件时，走 `_load_hf_model()` 路径，直接调用 `from_pretrained()`。

---

### 场景 3：接入其他架构（Llama / Mistral / ChatGLM）

适用于非 Qwen3 架构的模型。

**需要改的地方**：

| 文件 | 函数 | 改什么 | 原因 |
|------|------|--------|------|
| `model_adapter.py` | `_parse_tool_call()` | 修改正则表达式 | 不同模型的工具调用格式不同 |

当前 `_parse_tool_call()` 匹配 MiniMind3 的格式：

```python
# 当前格式
<tool_call>
{"name": "get_time", "arguments": {"timezone": "UTC"}}
</tool_call>
```

如果你的模型用 OpenAI 格式（`{"tool_calls": [...]}`），需要改正则：

```python
# OpenAI 格式示例
def _parse_tool_call(self, text: str, tools: list) -> dict:
    import json
    try:
        data = json.loads(text)
        if "tool_calls" in data:
            tc = data["tool_calls"][0]
            return {"name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"])}
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None
```

**不需要改的地方**：

- `_load_hf_model()` / `_load_pth_model()` — `AutoModelForCausalLM` 自动适配任何 HF 支持的架构
- `generate()` / `stream_generate()` — 接口不变，内部自动处理
- `serve.py` / `agent.py` / `tools.py` — 完全不动

---

### 场景 4：接入 ChatGPT / Claude 等 API 模型

适用于通过 API 调用的闭源模型，不需要本地 GPU。

**需要改的地方**：

| 文件 | 函数 | 改什么 |
|------|------|--------|
| `model_adapter.py` | `_load_model()` | 改为初始化 API 客户端 |
| `model_adapter.py` | `generate()` | 改为调用 API |
| `model_adapter.py` | `stream_generate()` | 改为 API 流式调用 |

示例：

```python
class RealModel:
    def __init__(self, config):
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model_name = config.get("model_name", "gpt-4")
        self._loaded = True

    def generate(self, messages, tools=None):
        import httpx
        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
        }
        resp = httpx.post(f"{self.base_url}/chat/completions",
                          json=payload,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        data = resp.json()
        return data["choices"][0]["message"]

    def stream_generate(self, messages, tools=None):
        # ... 流式实现
        yield {"type": "chunk", "data": token}
        yield {"type": "done", "data": ""}
```

---

### 加载模式速查

`model_path` 自动识别，无需手动指定格式：

| model_path | 加载路径 | 需要的文件 |
|-----------|---------|-----------|
| 目录，内有 `.pth` | `_load_pth_model()` | config.json + .pth + tokenizer |
| 目录，无 `.pth` | `_load_hf_model()` | config.json + model.safetensors + tokenizer |
| `.pth` 文件路径 | `_load_pth_model()` | config.json（同目录）+ .pth |
| 其他 | `_load_hf_model()` | 同上 |

## 系统架构

```
用户 -> Gradio WebUI / CLI
         |
    agent.py (ReAct 循环)
    |-- memory.py (ChromaDB 记忆检索)
    |-- tools.py (工具调度与执行)
    +-- HTTP POST /v1/chat/completions
              |
         serve.py (FastAPI)
              |
         MockModel / Qwen3ForCausalLM
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 推理后端 | Mock / Qwen3ForCausalLM（ROCm 7.2.1 / CUDA / CPU） |
| API 框架 | FastAPI + uvicorn |
| API 格式 | OpenAI Chat Completions 兼容 |
| Agent | 自建 ReAct 循环 |
| 记忆存储 | ChromaDB（Milvus 预留） |
| 工具协议 | MCP 混合模式 |
| 前端 | Gradio WebUI |
| 配置 | config.yaml |

## 项目结构

```
project/
├── README.md
├── AGENTS.md               # AI 编程助手上文
├── config.yaml             # 全局配置
├── serve.py                # OpenAI 兼容推理服务
├── agent.py                # ReAct Agent
├── tools.py                # 工具注册与执行
├── memory.py               # 记忆管理
├── mock_model.py           # Mock 模型
├── model_adapter.py        # 真实模型适配器（Qwen3ForCausalLM）
├── main.py                 # 入口
├── test_api.py             # 测试脚本（支持 --real）
├── start_serve.bat         # 启动推理服务
├── models/
│   └── minimind-fc/        # 模型文件（config + weights + tokenizer）
├── docs/                   # 设计文档与面试准备
└── data/
    └── memory/             # ChromaDB 持久化目录
```

## 验收清单

```
[x] Mock 模式 test_api.py 7/8 通过（tool_calls 关键词匹配波动）
[x] 真实模型 test_api.py --real 10/11 通过
[x] SSE 流式输出正常
[x] ROCm 7.2.1 GPU 识别正常（AMD Radeon RX 9070 GRE）
[x] 中文对话连贯
```

## License

MIT
