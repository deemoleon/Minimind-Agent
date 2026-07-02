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

模型文件位于 `models/minimind-fc/`：

```
models/minimind-fc/
├── config.json              # 模型配置（从 HuggingFace 拉取）
├── full_sft_768.pth         # 训练权重
├── tokenizer.json           # 分词器
└── tokenizer_config.json    # 分词器配置
```

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
[x] Mock 模式 test_api.py 8/8 通过
[x] 真实模型 test_api.py --real 10/11 通过
[x] SSE 流式输出正常
[x] ROCm 7.2.1 GPU 识别正常（AMD Radeon RX 9070 GRE）
[x] 中文对话连贯
```

## License

MIT
