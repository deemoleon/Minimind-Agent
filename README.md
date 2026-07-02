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

### 1. 获取模型文件

从 HuggingFace 下载 MiniMind3 权重：

```bash
# 方法 1: git clone（推荐，约 300MB）
git clone https://huggingface.co/jingyaogong/minimind-3 models/minimind-fc

# 方法 2: 手动下载
# 需要以下 4 个文件：
#   config.json              ← 模型架构配置（Qwen3）
#   full_sft_768.pth         ← 训练权重（137MB）
#   tokenizer.json           ← 分词器
#   tokenizer_config.json    ← 分词器配置
```

### 2. 放置模型文件

将文件放入 `models/minimind-fc/` 目录：

```
models/minimind-fc/
├── config.json              # 必须：Qwen3 架构配置
├── full_sft_768.pth         # 必须：训练权重
├── tokenizer.json           # 必须：分词器
└── tokenizer_config.json    # 必须：分词器配置
```

**注意**：`config.json` 必须与 `.pth` 权重匹配。MiniMind3 使用 Qwen3 架构，`config.json` 中的 `model_type` 应为 `"qwen3"`。

### 3. 修改配置

编辑 `config.yaml`：

```yaml
model:
  mock: false                              # 关闭 Mock
  model_path: "models/minimind-fc/"        # 指向模型目录
```

`model_adapter.py` 支持三种加载模式，`model_path` 自动识别：

| model_path 指向 | 加载方式 | 需要的文件 |
|----------------|---------|-----------|
| 目录（如 `models/minimind-fc/`） | 自动找 `*.pth`，优先 `full_sft_*.pth` | config.json + .pth + tokenizer |
| `.pth` 文件 | 直接加载 PyTorch 权重 | config.json（同目录）+ .pth |
| HF 格式目录 | `from_pretrained()` 加载 | config.json + model.safetensors + tokenizer |

### 4. 启动验证

```bash
# 启动服务
start_serve.bat

# 运行真实模型测试（3 个额外测试）
python test_api.py --real

# 预期输出：10/11 通过
```

### 5. 代码适配说明

如果你使用**非 MiniMind3 的模型**（如 Llama、Mistral），需要修改 `model_adapter.py`：

| 场景 | 需要改什么 |
|------|-----------|
| 模型架构不同（非 Qwen3） | `_load_pth_model()` 中 `AutoModelForCausalLM.from_config()` 会自动适配，只要 `config.json` 正确 |
| 分词器不同 | `_load_pth_model()` 已用 `AutoTokenizer`，自动适配 |
| 工具调用格式不同 | `_parse_tool_call()` 中的正则表达式需要调整（当前匹配 `<tool_call>...</tool_call>`） |
| 使用 HF safetensors 格式 | 直接用 `_load_hf_model()`，`model_path` 指向包含 `model.safetensors` 的目录 |
| 需要 GPU 显存优化 | 修改 `_load_pth_model()` 中的 `dtype`（当前 FP16，可改 INT8/INT4） |

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
