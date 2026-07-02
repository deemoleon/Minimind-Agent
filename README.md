# MiniMind Function Calling Agent

为轻量级大模型（MiniMind3 / Qwen3 架构）构建 OpenAI 兼容的 Function Calling API + ReAct Agent，支持 Mock 模式全流程验证。

## 快速开始

### 第一步：一键安装

```bash
install.bat
```

自动创建虚拟环境 `venv/`（如不存在）并安装全部依赖。无需手动 `pip` 或激活 venv。

自动创建 venv + 安装依赖。

### 第二步：启动 Mock 模式（无需 GPU）

```bash
start_serve.bat
python main.py
```

Mock 模式用关键词匹配模拟模型推理，ReAct + Function Calling + Memory + SSE 全链路正常工作。

### 第三步：接真实模型（需要 GPU）

```bash
# 1. 下载权重（约 300MB）
git clone https://huggingface.co/jingyaogong/minimind-3 models/minimind-fc

# 2. 改 config.yaml
#    mock: false

# 3. 重启
start_serve.bat
python main.py
```

## 项目结构

```
├── install.bat             # 一键安装（venv + 依赖）
├── start_serve.bat         # 后台启动推理服务（日志到 tmp/）
├── start_agent.bat         # 启动 WebUI（Gradio，端口 7860）
├── start_all.bat           # 一键启动（serve + WebUI）
│
├── config.yaml             # 全局配置（mock 开关、模型路径、端口）
├── serve.py                # FastAPI 推理服务（OpenAI 兼容 /v1/chat/completions）
├── agent.py                # ReAct 循环（最多 10 轮工具调用）
├── tools.py                # 工具注册（@tool 装饰器）
├── memory.py               # ChromaDB 长期记忆
├── mock_model.py           # Mock 推理（关键词匹配）
├── model_adapter.py        # 真实模型适配器（Qwen3 / HF / llama.cpp）
├── main.py                 # CLI / WebUI 入口
├── test_api.py             # 测试脚本（--real 跑真实模型）
│
├── models/
│   └── minimind-fc/        # 模型权重（config.json + .pth + tokenizer）
└── data/
    └── memory/             # ChromaDB 持久化
```

## 换模型接入

### 设计原理

系统的分层架构保证了**换模型只改 `model_adapter.py`，`serve.py` / `agent.py` / `tools.py` 完全不动**。

```
serve.py（OpenAI 格式服务）
    ↓ 只调这两个方法
model_adapter.py（RealModel）
    ↓ 内部实现
具体模型（Qwen3 / Llama / llama.cpp / GPT-4 API ...）
```

`RealModel` 对外暴露与 `MockModel` 完全一致的接口：

```python
def generate(messages, tools)        # → str（纯文本）或 dict（工具调用）
def stream_generate(messages, tools)  # → 逐 token yield
```

`model_adapter.py:_load_model()` 根据 `model_path` 自动选择加载路径：

| `model_path` | 走哪条路 | 为什么 |
|--------------|---------|--------|
| 目录，内有 `.pth` | `_load_pth_model()` | 目录内自动找 `full_sft_*.pth` |
| `.pth` 文件路径 | `_load_pth_model()` | 直接指定权重文件 |
| 目录，无 `.pth` | `_load_hf_model()` | 走 `from_pretrained()` 标准流程 |
| 其他 | `_load_hf_model()` | 兜底 HF 格式 |

---

### 场景 1：MiniMind3（.pth 权重）

这是本项目的默认模型。`.pth` 是 PyTorch 原生权重格式，不是 HuggingFace 标准格式，所以需要特殊的三步加载：

```
config.json → AutoConfig.from_pretrained()         → 读取架构参数（Qwen3）
Qwen3Config → AutoModelForCausalLM.from_config()    → 创建空模型结构
full_sft_768.pth → load_state_dict()                → 填充训练权重
```

**为什么不用 `from_pretrained()`？** 因为 `.pth` 文件没有 HF 格式的 `model.safetensors` + `config.json` 组合。`from_pretrained()` 要求完整 HF 格式目录。三步分离（config → 创建结构 → 加载权重）可以灵活兼容非标准格式。

**操作**：

```bash
# 1. 下载权重
git clone https://huggingface.co/jingyaogong/minimind-3 models/minimind-fc

# 2. 改 config.yaml
model:
  mock: false
  model_path: "models/minimind-fc/"

# 3. 重启
start_serve.bat
```

---

### 场景 2：HuggingFace 格式模型

适用于 `model.safetensors` 或 `pytorch_model.bin` 标准格式（如 Llama-3、Mistral-7B 等）。

**为什么不需要改代码？** `model_adapter.py:_load_hf_model()` 直接调用 `from_pretrained()`，`AutoModelForCausalLM` 会根据 `config.json` 中的 `model_type` 自动选择正确的模型架构（LlamaForCausalLM、MistralForCausalLM 等）。

**操作**：

```bash
# 1. 下载或转换模型到 HF 格式，放入目录
models/my-model/
├── config.json
├── model.safetensors        # 或 pytorch_model.bin
├── tokenizer.json
└── tokenizer_config.json

# 2. 改 config.yaml
model:
  mock: false
  model_path: "models/my-model/"

# 3. 重启
start_serve.bat
```

---

### 场景 3：其他架构（Llama / Mistral / ChatGLM）

`from_pretrained()` 自动适配架构，**但工具调用格式可能不同**。

MiniMind3 的工具调用格式是 `<tool_call>{"name": "xxx", "arguments": {...}}</tool_call>`。如果你的模型用 OpenAI 格式（`{"tool_calls": [...]}`）或其他格式，需要修改 `model_adapter.py:_parse_tool_call()` 中的正则表达式。

**需要改的地方**：

```python
# model_adapter.py:291 — _parse_tool_call()
# 当前匹配 MiniMind3 格式：
pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"

# 改成匹配你的模型输出格式，例如 OpenAI 格式：
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

**不需要改的地方**：`_load_hf_model()` / `_load_pth_model()`（架构自动适配）、`generate()` / `stream_generate()`（接口不变）、`serve.py` / `agent.py` / `tools.py`（完全不动）。

---

### 场景 4：llama.cpp (.gguf)

适用于 GGUF 量化格式，推理速度更快、显存占用更低。

**需要改的地方**：`model_adapter.py` 内部实现全部替换为 `llama-cpp-python`，接口保持不变：

```python
from llama_cpp import Llama

class RealModel:
    def __init__(self, config):
        self._llm = Llama(model_path=config["model_path"], n_ctx=2048)
        self._loaded = True

    def generate(self, messages, tools=None):
        prompt = self._build_prompt(messages)
        result = self._llm(prompt, max_tokens=2048)
        text = result["choices"][0]["text"]
        if tools:
            return self._parse_tool_call(text, tools)  # 正则逻辑保留
        return text

    def stream_generate(self, messages, tools=None):
        # stream=True 逐 token yield
        for token in self._llm(prompt, stream=True):
            yield {"type": "chunk", "data": token["choices"][0]["text"]}
        yield {"type": "done", "data": ""}
```

**为什么推荐方式 A（改 adapter）而不是方式 B（让 serve.py 直接调 llama.cpp）？** 因为方式 B 会破坏 Clean Service Separation——以后换 PyTorch 模型又要改 serve.py。保留 adapter 作为抽象层，换模型只改一个文件。

---

### 场景 5：API 模型 (GPT-4 / Claude)

适用于不需要本地 GPU 的场景，通过 HTTP API 调用闭源模型。

**需要改的地方**：`model_adapter.py` 整个类重写，`generate()` 和 `stream_generate()` 改为调用 API：

```python
import httpx

class RealModel:
    def __init__(self, config):
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model_name = config.get("model_name", "gpt-4")
        self._loaded = True

    def generate(self, messages, tools=None):
        payload = {"model": self.model_name, "messages": messages, "tools": tools}
        resp = httpx.post(f"{self.base_url}/chat/completions",
                          json=payload,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        data = resp.json()
        return data["choices"][0]["message"]

    def stream_generate(self, messages, tools=None):
        payload = {"model": self.model_name, "messages": messages, "stream": True}
        with httpx.stream("POST", f"{self.base_url}/chat/completions",
                          json=payload,
                          headers={"Authorization": f"Bearer {self.api_key}"}) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield {"type": "chunk", "data": delta["content"]}
        yield {"type": "done", "data": ""}
```

**同时需要改 `config.yaml`** 加上 API 配置：

```yaml
model:
  mock: false
  model_path: ""       # API 模式不用本地路径
  api_key: "sk-xxx"
  base_url: "https://api.openai.com/v1"
  model_name: "gpt-4"
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 推理后端 | Mock / Qwen3ForCausalLM（ROCm 7.2.1 / CUDA / CPU） |
| API | FastAPI，OpenAI Chat Completions 兼容 |
| Agent | 自建 ReAct 循环（max_rounds=10, retry=3） |
| 记忆 | ChromaDB（Milvus 预留） |
| 工具 | @tool 装饰器自动注册，JSON Schema 自动生成 |
| 前端 | Gradio WebUI（支持流式逐字显示） |

## License

MIT
