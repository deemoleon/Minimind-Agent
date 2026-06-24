# Changelog

## [0.2.0] - 2026-06-24

### Added
- SSE 流式输出全链路：mock_model.stream_generate() → serve.py StreamingResponse → agent.chat_stream() → main.py 流式显示
- test_api.py 第 8 个测试：test_stream_text（验证 SSE 格式和 [DONE] 标记）
- docs/analysis-summary.md 七大功能深度分析（含 SSE 流式 Section 8）
- docs/superpowers/specs/2026-06-24-streaming-design.md 流式功能设计文档

### Changed
- serve.py：stream=True 时返回 StreamingResponse（SSE 协议）
- agent.py：新增 chat_stream() 和 _call_model_stream()，保留 chat() 同步方法
- main.py：CLI 逐字符打印，WebUI Gradio generator 逐字显示

## [0.1.0] - 2026-06-11

### Added
- config.yaml 全局配置（model/server/memory/agent/tools）
- mock_model.py Mock 模型（关键词匹配，CPU 可运行）
- tools.py 工具注册系统 + 4 个预置工具（get_time, read_file, run_shell, web_search）
- serve.py OpenAI 兼容 FastAPI 推理服务（/health, /v1/models, /v1/chat/completions）
- memory.py ChromaDB 记忆管理（语义检索 + 会话历史）
- agent.py ReAct Agent（多轮工具调用、记忆读写、错误重试）
- main.py CLI / WebUI 双入口
- test_api.py API 测试脚本（7/7 通过）
- install.bat / start_server.bat / start_agent.bat / start_all.bat 启动脚本
- README.md / AGENTS.md / models/README.md 项目文档

### Notes
- 当前为 Mock 模式，不依赖 GPU
- 所有 .bat 文件使用纯 ASCII 字符，避免 GBK 编码问题
- Python 路径硬编码为 C:\Users\64987\anaconda3\python.exe（用户环境）
- 等待接入训练完成的 MiniMind 模型权重
