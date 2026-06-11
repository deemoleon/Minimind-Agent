# MiniMind Function Calling Agent

基于 MiniMind 轻量级大模型的 Function Calling Agent 系统，支持 Windows + AMD GPU 环境。

## 项目状态

当前为 **Mock 模式**，使用模拟模型验证全流程架构。训练完成后替换权重即可上线。

## 功能特性

- 🤖 **ReAct Agent**：思考-行动-观察循环，支持多轮工具调用
- 🔧 **Function Calling**：模型自主选择和调用工具，严格遵循 OpenAI tool_calls 格式
- 🧠 **长期记忆**：ChromaDB 向量存储，跨对话上下文检索
- 🌐 **OpenAI 兼容 API**：`/v1/chat/completions` 标准接口，LangChain/AutoGen 等框架零侵入接入
- 🛠️ **MCP 混合模式**：内置工具 + 可扩展外部 MCP Server
- 🖥️ **Gradio WebUI**：可视化聊天界面，工具调用过程实时展示
- ⚡ **AMD GPU 适配**：DirectML 后端，消费级显卡可运行

## 快速开始

### 1. 安装依赖

```bash
install.bat
```

### 2. 启动服务

```bash
# 终端 1：启动模型推理服务
start_server.bat

# 终端 2：运行测试
python test_api.py
```

### 3. 启动 Agent

```bash
# WebUI 模式（默认）
start_agent.bat

# CLI 模式
python main.py --cli
```

浏览器自动打开 `http://localhost:7860`。

### 4. 接入真实模型

训练完成后：

1. 将模型权重放入 `models/minimind-fc/`
2. 修改 `config.yaml`：

```yaml
model:
  mock: false
  model_path: "./models/minimind-fc"
  backend: "directml"
```

3. 重启服务

## 系统架构

```
用户 → Gradio WebUI / CLI
         ↓
    agent.py (ReAct 循环)
    ├── memory.py (ChromaDB 记忆检索)
    ├── tools.py (工具调度与执行)
    └── HTTP POST /v1/chat/completions
              ↓
         serve.py (FastAPI)
              ↓
         MockModel / MiniMind 模型
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 推理后端 | Mock → DirectML → ONNX Runtime → CPU fallback |
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
├── README.md               # 本文件
├── AGENTS.md               # AI 编程助手上文
├── config.yaml             # 全局配置
├── serve.py                # OpenAI 兼容推理服务
├── agent.py                # ReAct Agent
├── tools.py                # 工具注册与执行
├── memory.py               # 记忆管理
├── mock_model.py           # Mock 模型
├── main.py                 # 入口
├── test_api.py             # 测试脚本
├── install.bat             # 环境安装
├── start_server.bat        # 启动推理服务
├── start_agent.bat         # 启动 Agent
├── start_all.bat           # 一键启动
├── models/
│   └── README.md           # 模型部署说明
└── data/
    └── memory/             # ChromaDB 持久化目录
```

## 验收清单

```
□ install.bat 安装无报错
□ start_server.bat 正常启动
□ test_api.py 输出 7/7
□ start_agent.bat 打开 Gradio 页面
□ 纯文本对话正常
□ 工具调用正常
□ 状态栏显示正确
```

## License

MIT
