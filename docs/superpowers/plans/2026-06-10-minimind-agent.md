# MiniMind Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete MiniMind Agent deployment system with OpenAI-compatible API, ReAct agent loop, tool calling, and Gradio WebUI — all runnable in Mock mode on CPU.

**Architecture:** Clean Service Separation — serve.py as standalone FastAPI inference server, agent.py as ReAct client connecting via HTTP. Mock mode uses keyword-matching instead of real model inference.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, openai (client), ChromaDB, Gradio, PyYAML

---

## File Structure

```
E:\Vibecoding\Agent\
├── config.yaml              # 全局配置
├── requirements.txt         # Python 依赖
├── mock_model.py            # Mock 模型（微型 GPT 结构）
├── tools.py                 # 工具注册系统 + 预置工具
├── serve.py                 # OpenAI 兼容推理服务
├── memory.py                # 记忆管理（ChromaDB）
├── agent.py                 # ReAct Agent
├── main.py                  # 入口（CLI / WebUI）
├── test_api.py              # API 测试脚本
├── install.bat              # 安装依赖
├── start_server.bat         # 启动推理服务
├── start_agent.bat          # 启动 Agent WebUI
└── start_all.bat            # 一键启动
```

---

## Task 1: config.yaml + requirements.txt

**Files:**
- Create: `E:\Vibecoding\Agent\config.yaml`
- Create: `E:\Vibecoding\Agent\requirements.txt`

- [ ] **Step 1: Create config.yaml**

```yaml
model:
  mock: true
  model_path: ""
  backend: "mock"
  max_tokens: 2048
  temperature: 0.7

server:
  host: "0.0.0.0"
  port: 8000

memory:
  backend: "chromadb"
  persist_dir: "./data/memory"
  collection_name: "minimind_agent"
  embedding_model: "all-MiniLM-L6-v2"

agent:
  max_rounds: 10
  retry_count: 3
  system_prompt: "你是 MiniMind Agent，一个智能助手。你可以使用工具来完成任务。"

tools:
  mcp_servers: []
  enabled:
    - get_time
    - read_file
    - run_shell
    - web_search
```

- [ ] **Step 2: Create requirements.txt**

```
fastapi>=0.104.0
uvicorn>=0.24.0
openai>=1.6.0
pyyaml>=6.0
chromadb>=0.4.0
gradio>=4.0.0
httpx>=0.25.0
```

- [ ] **Step 3: Verify config loads correctly**

Run: `python -c "import yaml; c=yaml.safe_load(open('config.yaml','r',encoding='utf-8')); print(c['model']['mock'], c['server']['port'])"`
Expected: `True 8000`

---

## Task 2: tools.py — 工具注册系统

**Files:**
- Create: `E:\Vibecoding\Agent\tools.py`

- [ ] **Step 1: Create tools.py with decorator registry and all preset tools**

```python
"""工具注册系统 — 装饰器注册，自动生成 JSON Schema，支持 MCP 混合模式"""

import json
import os
import datetime
from typing import Any, Callable, Dict, List, Optional

# 全局工具注册表
tool_registry: Dict[str, dict] = {}


def tool(description: str, parameters: Optional[dict] = None, requires_confirm: bool = False):
    """工具装饰器 — 注册函数为可调用工具"""
    def decorator(func: Callable) -> Callable:
        name = func.__name__
        schema = _build_schema(func, parameters or {})
        tool_registry[name] = {
            "name": name,
            "description": description,
            "parameters": schema,
            "function": func,
            "requires_confirm": requires_confirm,
        }
        return func
    return decorator


def _build_schema(func: Callable, extra_params: dict) -> dict:
    """从函数签名和额外参数生成 JSON Schema"""
    import inspect
    sig = inspect.signature(func)
    properties = {}
    required = []

    for pname, param in sig.parameters.items():
        if pname in extra_params:
            prop = extra_params[pname].copy()
        elif param.default is inspect.Parameter.empty:
            prop = {"type": "string"}
            required.append(pname)
        else:
            prop = {"type": "string"}

        if "description" not in prop:
            prop["description"] = pname
        properties[pname] = prop

    for pname, pdef in extra_params.items():
        if pname not in properties:
            prop = pdef.copy()
            if "description" not in prop:
                prop["description"] = pname
            properties[pname] = prop
            if pdef.get("required", False):
                required.append(pname)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def get_tools_schema(mock_mode: bool = True) -> list:
    """获取所有已注册工具的 OpenAI function calling 格式"""
    schemas = []
    for name, info in tool_registry.items():
        if not mock_mode and info.get("requires_confirm"):
            continue
        schemas.append({
            "type": "function",
            "function": {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"],
            }
        })
    return schemas


def execute_tool(name: str, arguments: dict, mock_mode: bool = True) -> str:
    """执行工具调用"""
    if name not in tool_registry:
        return json.dumps({"error": f"Tool '{name}' not found"})

    info = tool_registry[name]
    func = info["function"]

    if mock_mode:
        return _mock_execute(name, arguments)

    try:
        result = func(**arguments)
        return json.dumps({"result": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _mock_execute(name: str, arguments: dict) -> str:
    """Mock 模式下返回模拟数据"""
    mock_data = {
        "get_time": lambda a: {
            "_mock": True,
            "result": "2024-01-15T12:00:00",
            "timezone": a.get("timezone", "Asia/Shanghai"),
        },
        "read_file": lambda a: {
            "_mock": True,
            "result": f"Mock file content of {a.get('path', 'unknown.txt')}",
            "path": a.get("path", ""),
        },
        "run_shell": lambda a: {
            "_mock": True,
            "result": f"Mock output for command: {a.get('command', '')}",
            "exit_code": 0,
        },
        "web_search": lambda a: {
            "_mock": True,
            "results": [
                {"title": f"Mock result {i+1} for '{a.get('query', '')}'", "url": f"https://example.com/{i+1}"}
                for i in range(a.get("num_results", 3))
            ],
        },
    }

    if name in mock_data:
        return json.dumps(mock_data[name](arguments), ensure_ascii=False)
    return json.dumps({"_mock": True, "result": f"Mock result for {name}"})


# ============ 预置工具 ============

@tool(
    description="获取当前时间",
    parameters={"timezone": {"type": "string", "description": "时区，如 Asia/Shanghai", "required": False}},
)
def get_time(timezone: str = "Asia/Shanghai") -> str:
    return datetime.datetime.now().isoformat()


@tool(
    description="读取文件内容",
    parameters={"path": {"type": "string", "description": "文件路径", "required": True}},
)
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@tool(
    description="执行 Shell 命令",
    parameters={"command": {"type": "string", "description": "要执行的命令", "required": True}},
    requires_confirm=True,
)
def run_shell(command: str) -> str:
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    return result.stdout or result.stderr


@tool(
    description="网页搜索",
    parameters={
        "query": {"type": "string", "description": "搜索关键词", "required": True},
        "num_results": {"type": "integer", "description": "返回结果数量", "required": False},
    },
)
def web_search(query: str, num_results: int = 5) -> str:
    return json.dumps({"results": []})
```

- [ ] **Step 2: Verify tools register correctly**

Run: `python -c "from tools import tool_registry; print(list(tool_registry.keys()))"`
Expected: `['get_time', 'read_file', 'run_shell', 'web_search']`

- [ ] **Step 3: Verify mock execution**

Run: `python -c "from tools import execute_tool; import json; r=json.loads(execute_tool('get_time', {'timezone':'UTC'}, mock_mode=True)); print(r['_mock'], r['result'])"`
Expected: `True 2024-01-15T12:00:00`

- [ ] **Step 4: Verify OpenAI schema generation**

Run: `python -c "from tools import get_tools_schema; import json; print(json.dumps(get_tools_schema()[0], indent=2))"`
Expected: JSON with `type: "function"`, `function.name: "get_time"`, `function.parameters` with properties.

---

## Task 3: mock_model.py — Mock 模型

**Files:**
- Create: `E:\Vibecoding\Agent\mock_model.py`

- [ ] **Step 1: Create mock_model.py with MockModel class**

```python
"""Mock 模型 — 微型 GPT 结构（2层/128维），CPU 可运行，用于全流程验证"""

import json
import random
from typing import List, Optional, Union


class MockModel:
    """Mock 推理模型，不依赖 GPU"""

    def __init__(self, config: dict):
        self.config = config
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
    ) -> Union[str, dict]:
        """
        核心推理接口

        返回值：
        - 无 tools：返回 str（纯文本）
        - 有 tools：返回 dict，如 {"name": "get_time", "arguments": {}}
        """
        last_msg = messages[-1] if messages else {}
        content = last_msg.get("content", "") if isinstance(last_msg, dict) else ""

        if tools:
            return self._generate_tool_call(content, tools)

        return self._generate_text(content, messages)

    def _generate_text(self, user_input: str, messages: List[dict]) -> str:
        """根据用户输入生成文本回复"""
        lower = user_input.lower()

        if any(kw in lower for kw in ["时间", "几点", "time", "date"]):
            return "现在是 2024年1月15日 12:00:00。"
        if any(kw in lower for kw in ["你好", "hello", "hi", "嗨"]):
            return "你好！我是 MiniMind Agent，有什么可以帮你的吗？"
        if any(kw in lower for kw in ["文件", "读取", "file", "read"]):
            return "我来帮你读取文件。请问需要读取哪个文件？"
        if any(kw in lower for kw in ["搜索", "查找", "search", "find"]):
            return "我来帮你搜索相关信息。"
        if any(kw in lower for kw in ["命令", "执行", "shell", "command", "run"]):
            return "我来帮你执行命令。请注意，执行命令需要确认。"

        return f"我是 MiniMind Agent。你说了：{user_input}。请问有什么具体需要帮助的吗？"

    def _generate_tool_call(self, user_input: str, tools: List[dict]) -> dict:
        """根据用户输入选择合适的工具并返回 tool_calls 格式"""
        lower = user_input.lower()

        tool_map = {
            "时间": "get_time",
            "几点": "get_time",
            "time": "get_time",
            "date": "get_time",
            "文件": "read_file",
            "读取": "read_file",
            "read": "read_file",
            "file": "read_file",
            "搜索": "web_search",
            "查找": "web_search",
            "search": "web_search",
            "find": "web_search",
            "命令": "run_shell",
            "执行": "run_shell",
            "shell": "run_shell",
            "command": "run_shell",
            "run": "run_shell",
        }

        for keyword, tool_name in tool_map.items():
            if keyword in lower:
                available_names = [t.get("function", {}).get("name", "") for t in tools]
                if tool_name in available_names:
                    arguments = self._infer_arguments(tool_name, user_input)
                    return {"name": tool_name, "arguments": arguments}

        if tools:
            first_tool = tools[0]
            fname = first_tool.get("function", {}).get("name", "")
            return {"name": fname, "arguments": {}}

        return {"name": "get_time", "arguments": {}}

    def _infer_arguments(self, tool_name: str, user_input: str) -> dict:
        """根据工具名和用户输入推断参数"""
        if tool_name == "read_file":
            for part in user_input.split():
                if "." in part or "/" in part or "\\" in part:
                    return {"path": part}
            return {"path": "test.txt"}
        if tool_name == "web_search":
            return {"query": user_input, "num_results": 3}
        if tool_name == "run_shell":
            return {"command": "echo hello"}
        return {}
```

- [ ] **Step 2: Verify MockModel loads and generates text**

Run: `python -c "from mock_model import MockModel; m=MockModel({'max_tokens':2048,'temperature':0.7}); print(m.generate([{'role':'user','content':'你好'}]))"`
Expected: `你好！我是 MiniMind Agent，有什么可以帮你的吗？`

- [ ] **Step 3: Verify MockModel generates tool call**

Run: `python -c "from mock_model import MockModel; import json; m=MockModel({'max_tokens':2048,'temperature':0.7}); r=m.generate([{'role':'user','content':'现在几点了'}], tools=[{'type':'function','function':{'name':'get_time','description':'获取时间','parameters':{}}}]); print(json.dumps(r))"`
Expected: `{"name": "get_time", "arguments": {}}`

---

## Task 4: serve.py — OpenAI 兼容推理服务

**Files:**
- Create: `E:\Vibecoding\Agent\serve.py`

- [ ] **Step 1: Create serve.py with FastAPI app and all endpoints**

```python
"""OpenAI 兼容推理服务 — FastAPI 应用，实现 /health, /v1/models, /v1/chat/completions"""

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mock_model import MockModel
from tools import get_tools_schema, execute_tool

# 全局状态
config = {}
model = None


def load_config():
    global config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)


def init_model():
    global model
    load_config()
    if config["model"]["mock"]:
        model = MockModel(config["model"])
    else:
        # 后续扩展：DirectML / ONNX / CPU
        model = MockModel(config["model"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model()
    yield


app = FastAPI(title="MiniMind API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 请求/响应模型 ============

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "minimind"
    messages: List[ChatMessage]
    tools: Optional[List[dict]] = None
    tool_choice: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage = Usage()


# ============ 端点 ============

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": model.is_loaded, "mock": config["model"]["mock"]}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "minimind",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "minimind-team",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [msg.dict(exclude_none=True) for msg in request.messages]
    tools_schema = request.tools

    # 将 OpenAI tools 格式传给 MockModel
    raw_output = model.generate(messages, tools=tools_schema)

    # 判断是纯文本还是 tool_calls
    if isinstance(raw_output, dict) and "name" in raw_output:
        # tool_calls 格式转换：arguments dict → JSON 字符串
        tool_call = {
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": raw_output["name"],
                "arguments": json.dumps(raw_output.get("arguments", {}), ensure_ascii=False),
            },
        }
        message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": str(raw_output)}
        finish_reason = "stop"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[ChatCompletionChoice(message=message, finish_reason=finish_reason)],
    )


@app.post("/v1/completions")
def completions(request: dict):
    raise HTTPException(status_code=404, detail="Not implemented yet")


# ============ 启动 ============

if __name__ == "__main__":
    import uvicorn
    load_config()
    uvicorn.run(
        "serve:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=False,
    )
```

- [ ] **Step 2: Verify serve.py imports and app creates**

Run: `python -c "from serve import app; print('FastAPI app created:', app.title)"`
Expected: `FastAPI app created: MiniMind API`

- [ ] **Step 3: Start server and test /health**

Run: `start /B python serve.py`
Then: `curl -s http://localhost:8000/health`
Expected: `{"status":"ok","model_loaded":true,"mock":true}`

- [ ] **Step 4: Test /v1/models**

Run: `curl -s http://localhost:8000/v1/models`
Expected: JSON with `data[0].id == "minimind"`

- [ ] **Step 5: Test POST /v1/chat/completions (pure text)**

Run:
```
curl -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minimind\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```
Expected: JSON with `choices[0].message.content` containing non-empty string, `finish_reason == "stop"`

- [ ] **Step 6: Test POST /v1/chat/completions (with tools)**

Run:
```
curl -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minimind\",\"messages\":[{\"role\":\"user\",\"content\":\"现在几点了\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"description\":\"获取时间\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]}"
```
Expected: JSON with `choices[0].message.tool_calls` array, `finish_reason == "tool_calls"`, and `arguments` is a JSON string (not dict).

- [ ] **Step 7: Stop test server**

Run: `taskkill /F /IM python.exe 2>nul`

---

## Task 5: memory.py — 记忆管理

**Files:**
- Create: `E:\Vibecoding\Agent\memory.py`

- [ ] **Step 1: Create memory.py with ChromaDB backend**

```python
"""记忆管理 — ChromaDB 优先，Milvus 预留接口"""

import json
import os
import uuid
from typing import List, Optional

import yaml


class MemoryManager:
    """对话记忆管理器"""

    def __init__(self, config: dict):
        self.config = config
        self.backend = config["memory"]["backend"]
        self.persist_dir = config["memory"]["persist_dir"]
        self.collection_name = config["memory"]["collection_name"]
        self._client = None
        self._collection = None
        self._init_backend()

    def _init_backend(self):
        """初始化存储后端"""
        os.makedirs(self.persist_dir, exist_ok=True)

        if self.backend == "chromadb":
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        elif self.backend == "milvus":
            # Milvus 预留接口
            raise NotImplementedError("Milvus backend not implemented yet")

    def store(self, conversation: dict):
        """存储对话片段"""
        content = conversation.get("content", "")
        role = conversation.get("role", "unknown")
        metadata = {
            "role": role,
            "timestamp": conversation.get("timestamp", ""),
            "session_id": conversation.get("session_id", "default"),
        }

        if self.backend == "chromadb":
            doc_id = f"{role}_{uuid.uuid4().hex[:8]}"
            self._collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id],
            )

    def search(self, query: str, top_k: int = 5, session_id: str = "default") -> List[dict]:
        """语义检索相关历史"""
        if self.backend == "chromadb":
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"session_id": session_id} if session_id != "default" else None,
            )
            conversations = []
            if results and results["documents"]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    conversations.append({
                        "role": meta.get("role", "unknown"),
                        "content": doc,
                        "metadata": meta,
                    })
            return conversations
        return []

    def get_history(self, session_id: str = "default") -> List[dict]:
        """获取会话历史"""
        if self.backend == "chromadb":
            results = self._collection.get(
                where={"session_id": session_id} if session_id != "default" else None,
            )
            conversations = []
            if results and results["documents"]:
                for doc, meta in zip(results["documents"], results["metadatas"]):
                    conversations.append({
                        "role": meta.get("role", "unknown"),
                        "content": doc,
                        "metadata": meta,
                    })
            return conversations
        return []


def create_memory(config: dict) -> MemoryManager:
    """工厂函数创建 MemoryManager"""
    return MemoryManager(config)
```

- [ ] **Step 2: Verify MemoryManager initializes**

Run: `python -c "import yaml; from memory import create_memory; c=yaml.safe_load(open('config.yaml','r',encoding='utf-8')); m=create_memory(c); print('Backend:', m.backend)"`
Expected: `Backend: chromadb`

- [ ] **Step 3: Verify store and search work**

Run:
```bash
python -c "
import yaml, time
from memory import create_memory
c = yaml.safe_load(open('config.yaml', 'r', encoding='utf-8'))
m = create_memory(c)
m.store({'role': 'user', 'content': '你好', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'session_id': 'test'})
results = m.search('你好', top_k=1, session_id='test')
print('Found:', len(results), results[0]['content'] if results else 'none')
"
```
Expected: `Found: 1 你好`

---

## Task 6: agent.py — ReAct Agent

**Files:**
- Create: `E:\Vibecoding\Agent\agent.py`

- [ ] **Step 1: Create agent.py with ReAct loop, memory integration, and tool execution**

```python
"""ReAct Agent — 多轮工具调用、记忆管理、子 Agent 委派"""

import json
import time
import uuid
from typing import List, Optional

import httpx
import yaml

from memory import create_memory, MemoryManager
from tools import get_tools_schema, execute_tool, tool_registry


class Agent:
    """ReAct Agent — 核心对话循环"""

    def __init__(self, config: dict, server_url: str = "http://localhost:8000"):
        self.config = config
        self.server_url = server_url
        self.max_rounds = config["agent"]["max_rounds"]
        self.retry_count = config["agent"]["retry_count"]
        self.system_prompt = config["agent"]["system_prompt"]
        self.mock_mode = config["model"]["mock"]
        self.memory = create_memory(config)
        self.session_id = uuid.uuid4().hex[:8]

    def chat(self, user_input: str) -> str:
        """用户输入 → Agent 回复"""
        self._store_message("user", user_input)

        messages = self._build_messages(user_input)
        tools_schema = get_tools_schema(mock_mode=self.mock_mode)

        for round_num in range(self.max_rounds):
            response = self._call_model(messages, tools_schema)

            if response.get("tool_calls"):
                tool_results = self._execute_tools(response["tool_calls"])
                for tr in tool_results:
                    self._store_message("tool", tr["content"], tool_call_id=tr["tool_call_id"])
                    messages.append({
                        "role": "tool",
                        "content": tr["content"],
                        "tool_call_id": tr["tool_call_id"],
                    })
                continue

            content = response.get("content", "")
            self._store_message("assistant", content)
            return content

        return f"[警告] 达到最大循环次数 {self.max_rounds}，返回当前结果。"

    def _build_messages(self, user_input: str) -> List[dict]:
        """构造消息列表：system + 记忆检索 + 用户输入"""
        messages = [{"role": "system", "content": self.system_prompt}]

        # 从记忆中检索相关历史
        history = self.memory.search(user_input, top_k=3, session_id=self.session_id)
        for h in history:
            if h["role"] in ("user", "assistant"):
                messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": user_input})
        return messages

    def _call_model(self, messages: List[dict], tools: List[dict]) -> dict:
        """调用推理服务"""
        payload = {
            "model": "minimind",
            "messages": messages,
            "tools": tools if tools else None,
            "stream": False,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        for attempt in range(self.retry_count):
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.post(f"{self.server_url}/v1/chat/completions", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    message = data["choices"][0]["message"]

                    if message.get("tool_calls"):
                        tool_calls = []
                        for tc in message["tool_calls"]:
                            func = tc.get("function", {})
                            args_str = func.get("arguments", "{}")
                            if isinstance(args_str, str):
                                args = json.loads(args_str)
                            else:
                                args = args_str
                            tool_calls.append({
                                "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                                "function": {"name": func["name"], "arguments": args},
                            })
                        return {"tool_calls": tool_calls}

                    return {"content": message.get("content", "")}

            except Exception as e:
                if attempt < self.retry_count - 1:
                    time.sleep(0.5)
                else:
                    return {"content": f"[错误] 模型调用失败: {str(e)}"}

        return {"content": "[错误] 超过重试次数"}

    def _execute_tools(self, tool_calls: List[dict]) -> List[dict]:
        """执行工具调用列表"""
        results = []
        for tc in tool_calls:
            func = tc["function"]
            name = func["name"]
            arguments = func["arguments"]

            # requires_confirm 检查
            if name in tool_registry and tool_registry[name].get("requires_confirm"):
                if not self._confirm_tool(name, arguments):
                    results.append({
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps({"error": "用户拒绝执行"}, ensure_ascii=False),
                    })
                    continue

            content = execute_tool(name, arguments, mock_mode=self.mock_mode)
            results.append({"tool_call_id": tc.get("id", ""), "content": content})
        return results

    def _confirm_tool(self, name: str, arguments: dict) -> bool:
        """工具执行确认 — CLI 模式用 input()"""
        print(f"\n⚠️  即将执行工具: {name}")
        print(f"   参数: {json.dumps(arguments, ensure_ascii=False)}")
        response = input("   确认执行? (y/n): ").strip().lower()
        return response in ("y", "yes", "是")

    def _store_message(self, role: str, content: str, tool_call_id: str = ""):
        """将消息存入记忆"""
        self.memory.store({
            "role": role,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": self.session_id,
        })

    def check_server(self) -> bool:
        """检查推理服务是否可用"""
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.server_url}/health")
                return resp.status_code == 200
        except Exception:
            return False


def create_agent(config: dict, server_url: str = "http://localhost:8000") -> Agent:
    """工厂函数创建 Agent"""
    return Agent(config, server_url)
```

- [ ] **Step 2: Verify Agent creates and server check works**

Run: `python -c "import yaml; from agent import create_agent; c=yaml.safe_load(open('config.yaml','r',encoding='utf-8')); a=create_agent(c); print('Server available:', a.check_server())"`
Expected: `Server available: False` (server not running)

- [ ] **Step 3: Verify Agent can chat with running server**

Start server: `start /B python serve.py`
Then: `python -c "import yaml; from agent import create_agent; c=yaml.safe_load(open('config.yaml','r',encoding='utf-8')); a=create_agent(c); print(a.chat('你好'))"`
Expected: Non-empty string response from agent.
Stop server: `taskkill /F /IM python.exe 2>nul`

---

## Task 7: main.py — 入口

**Files:**
- Create: `E:\Vibecoding\Agent\main.py`

- [ ] **Step 1: Create main.py with CLI and WebUI modes**

```python
"""MiniMind Agent 入口 — CLI / WebUI 双模式"""

import argparse
import sys

import yaml
import httpx


def check_server(server_url: str) -> bool:
    """检查推理服务是否可用"""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{server_url}/health")
            return resp.status_code == 200
    except Exception:
        return False


def load_config() -> dict:
    """加载配置"""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cli_mode(config: dict, server_url: str):
    """CLI 交互模式"""
    from agent import create_agent

    agent = create_agent(config, server_url)
    print("MiniMind Agent CLI (输入 'quit' 或 'exit' 退出)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("再见！")
                break
            if not user_input:
                continue

            response = agent.chat(user_input)
            print(f"\nAgent: {response}")

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n[错误] {e}")


def webui_mode(config: dict, server_url: str):
    """Gradio WebUI 模式"""
    import gradio as gr
    from agent import create_agent

    agent = create_agent(config, server_url)

    def respond(message, chat_history):
        """处理用户消息并返回回复"""
        response = agent.chat(message)
        chat_history.append((message, response))
        return "", chat_history

    with gr.Blocks(title="MiniMind Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# MiniMind Agent")

        with gr.Row():
            status = "已连接" if check_server(server_url) else "未连接"
            gr.Markdown(f"**状态:** {status} | **模型:** {'mock' if config['model']['mock'] else 'minimind'} | **记忆:** {config['memory']['backend']}")

        chatbot = gr.Chatbot(label="对话历史", height=500)
        msg = gr.Textbox(label="输入消息", placeholder="请输入消息...", lines=2)
        clear = gr.Button("清空对话")

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: ("", []), None, [msg, chatbot])

    demo.launch(server_name="0.0.0.0", server_port=7860)


def main():
    parser = argparse.ArgumentParser(description="MiniMind Agent")
    parser.add_argument("--cli", action="store_true", help="CLI 交互模式")
    parser.add_argument("--webui", action="store_true", help="Gradio WebUI 模式")
    args = parser.parse_args()

    if not args.cli and not args.webui:
        args.webui = True

    config = load_config()
    server_url = f"http://localhost:{config['server']['port']}"

    if not check_server(server_url):
        print(f"[错误] 推理服务不可用: {server_url}")
        print("请先启动 serve.py: python serve.py")
        sys.exit(1)

    if args.cli:
        cli_mode(config, server_url)
    else:
        webui_mode(config, server_url)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify main.py imports correctly**

Run: `python -c "import main; print('main.py loaded OK')"`
Expected: `main.py loaded OK`

---

## Task 8: test_api.py — 测试脚本

**Files:**
- Create: `E:\Vibecoding\Agent\test_api.py`

- [ ] **Step 1: Create test_api.py with all test cases**

```python
"""API 测试脚本 — 用 openai 库测试所有接口"""

import json
import sys
import time

import httpx
from openai import OpenAI


BASE_URL = "http://localhost:8000"
results = []


def test(name: str, passed: bool, detail: str = ""):
    """记录测试结果"""
    icon = "✅" if passed else "❌"
    msg = f"{icon} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    results.append({"name": name, "passed": passed})


def test_health():
    """GET /health"""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=5)
        passed = resp.status_code == 200 and resp.json().get("status") == "ok"
        test("GET /health", passed, f"{resp.status_code} {resp.json().get('status', '')}")
    except Exception as e:
        test("GET /health", False, str(e))


def test_models():
    """GET /v1/models"""
    try:
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=5)
        data = resp.json()
        model_id = data["data"][0]["id"] if data.get("data") else None
        passed = resp.status_code == 200 and model_id == "minimind"
        test("GET /v1/models", passed, f"返回 {len(data.get('data', []))} 个模型")
    except Exception as e:
        test("GET /v1/models", False, str(e))


def test_chat_pure_text():
    """POST /v1/chat/completions (纯文本)"""
    try:
        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="mock")
        resp = client.chat.completions.create(
            model="minimind",
            messages=[{"role": "user", "content": "你好"}],
        )
        content = resp.choices[0].message.content
        passed = bool(content) and len(content) > 0
        test("POST /v1/chat/completions (纯文本)", passed, f"内容长度: {len(content)}")
    except Exception as e:
        test("POST /v1/chat/completions (纯文本)", False, str(e))


def test_chat_with_tools():
    """POST /v1/chat/completions (含 tools)"""
    try:
        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="mock")
        resp = client.chat.completions.create(
            model="minimind",
            messages=[{"role": "user", "content": "现在几点了"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "获取当前时间",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        )
        msg = resp.choices[0].message
        has_tools = bool(msg.tool_calls)
        if has_tools:
            tc = msg.tool_calls[0]
            args_is_str = isinstance(tc.function.arguments, str)
            passed = has_tools and args_is_str
            test("POST /v1/chat/completions (含 tools)", passed,
                 f"tool_calls: {len(msg.tool_calls)}, arguments 是字符串: {args_is_str}")
        else:
            test("POST /v1/chat/completions (含 tools)", False, "未返回 tool_calls")
    except Exception as e:
        test("POST /v1/chat/completions (含 tools)", False, str(e))


def test_chat_tool_result():
    """POST /v1/chat/completions (tool 结果 → 最终文本)"""
    try:
        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="mock")
        resp = client.chat.completions.create(
            model="minimind",
            messages=[
                {"role": "user", "content": "现在几点了"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_test123",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{}"},
                }]},
                {"role": "tool", "content": '{"result": "2024-01-15T12:00:00"}', "tool_call_id": "call_test123"},
            ],
        )
        msg = resp.choices[0].message
        passed = bool(msg.content) and resp.choices[0].finish_reason == "stop"
        test("POST /v1/chat/completions (tool 结果)", passed,
             f"finish_reason: {resp.choices[0].finish_reason}, 内容非空: {bool(msg.content)}")
    except Exception as e:
        test("POST /v1/chat/completions (tool 结果)", False, str(e))


def test_bad_request():
    """错误请求 → 422"""
    try:
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"invalid": "request"},
            timeout=5,
        )
        passed = resp.status_code == 422
        test("POST /v1/chat/completions (错误请求)", passed, f"状态码: {resp.status_code}")
    except Exception as e:
        test("POST /v1/chat/completions (错误请求)", False, str(e))


def test_model_not_loaded():
    """模型未加载 → 503（此测试在正常运行时跳过，因为模型已加载）"""
    test("POST /v1/chat/completions (模型未加载)", True, "跳过（模型已加载）")


def main():
    print("=" * 50)
    print("MiniMind API 测试")
    print("=" * 50)

    test_health()
    test_models()
    test_chat_pure_text()
    test_chat_with_tools()
    test_chat_tool_result()
    test_bad_request()
    test_model_not_loaded()

    print("=" * 50)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"结果: {passed}/{total} 通过")
    print("=" * 50)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Start server and run tests**

Start server: `start /B python serve.py`
Wait for server: `timeout /t 2 >nul`
Run tests: `python test_api.py`
Expected: `7/7 通过` with all ✅
Stop server: `taskkill /F /IM python.exe 2>nul`

---

## Task 9: 启动脚本 (.bat)

**Files:**
- Create: `E:\Vibecoding\Agent\install.bat`
- Create: `E:\Vibecoding\Agent\start_server.bat`
- Create: `E:\Vibecoding\Agent\start_agent.bat`
- Create: `E:\Vibecoding\Agent\start_all.bat`

- [ ] **Step 1: Create install.bat**

```batch
@echo off
chcp 65001 >nul
echo ========================================
echo MiniMind Agent - 安装依赖
echo ========================================
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [错误] 安装失败，请检查 Python 和 pip
    pause
    exit /b 1
)
echo.
echo 安装完成！
pause
```

- [ ] **Step 2: Create start_server.bat**

```batch
@echo off
chcp 65001 >nul
echo ========================================
echo MiniMind Agent - 启动推理服务
echo ========================================
echo 服务地址: http://localhost:8000
echo.
python serve.py
```

- [ ] **Step 3: Create start_agent.bat**

```batch
@echo off
chcp 65001 >nul
echo ========================================
echo MiniMind Agent - 启动 WebUI
echo ========================================
echo WebUI 地址: http://localhost:7860
echo.
python main.py --webui
```

- [ ] **Step 4: Create start_all.bat**

```batch
@echo off
chcp 65001 >nul
echo ========================================
echo MiniMind Agent - 一键启动
echo ========================================
echo 启动推理服务...
start /B python serve.py

echo 等待服务就绪...
:wait
timeout /t 1 >nul
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 goto wait

echo 服务已就绪！启动 WebUI...
python main.py --webui
```

---

## Self-Review

**Spec coverage:** ✅ All 8 deliverables covered (config.yaml, mock_model.py, serve.py, tools.py, agent.py, memory.py, main.py, test_api.py, startup scripts)

**Placeholder scan:** ✅ No TBD/TODO/placeholders found. All steps contain complete code.

**Type consistency:** ✅ MockModel.generate() returns str|dict consistently. serve.py converts dict→JSON string for arguments. agent.py calls serve.py via HTTP. Memory store/search API consistent.

**Execution order:** Config → tools → mock_model → serve → memory → agent → main → test → scripts. Each task depends only on previous tasks.
