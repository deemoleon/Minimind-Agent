# MiniMind Function Calling Agent

为轻量级大模型（MiniMind3 / Qwen3 架构）构建 OpenAI 兼容的 Function Calling API + ReAct Agent，支持 Mock 模式全流程验证。

## 快速开始

### 第一步：安装

```bash
install.bat
```

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

`model_adapter.py` 的 `RealModel` 暴露与 `MockModel` 完全一致的接口：

```python
def generate(messages, tools)        # → str（纯文本）或 dict（工具调用）
def stream_generate(messages, tools)  # → 逐 token yield
```

serve.py 不关心模型内部。**只要实现这两个方法，就能接入。**

### 接入 MiniMind3（.pth 权重）

```bash
git clone https://huggingface.co/jingyaogong/minimind-3 models/minimind-fc
```

```
models/minimind-fc/
├── config.json              # model_type: "qwen3"
├── full_sft_768.pth         # 训练权重
├── tokenizer.json
└── tokenizer_config.json
```

```yaml
# config.yaml
model:
  mock: false
  model_path: "models/minimind-fc/"
```

`model_path` 指向目录时自动找 `.pth` 文件，优先 `full_sft_*.pth`。

### 其他场景速查

| 场景 | 需要改的文件 | 备注 |
|------|-------------|------|
| HuggingFace safetensors | 无 | 目录内无 `.pth` 时自动走 `from_pretrained()` |
| Llama / Mistral / ChatGLM | `model_adapter.py:_parse_tool_call()` | 只改正则，适配你的工具调用格式 |
| llama.cpp (.gguf) | `model_adapter.py` 内部实现 | 保留接口，内部换 llama-cpp-python |
| API 模型 (GPT-4 / Claude) | `model_adapter.py` 整个类 | `generate()` 改为调 HTTP API |

所有场景下 `serve.py` / `agent.py` / `tools.py` 完全不动。

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
