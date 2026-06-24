# MiniMind Agent 七大功能深度分析

> 面试准备级别详解，覆盖实现原理、代码上下文、跨文件调用链、高频面试题。

---

## 1. ReAct Agent — 思考-行动-观察循环

### 1.1 什么是 ReAct

ReAct = **Rea**soning + **Act**ing。2022 年 Google 论文提出，让大模型在**推理**和**行动**之间交替执行，每一步都能看到之前行动的结果。核心思想：模型不直接解决问题，而是决定"下一步该做什么"，由外部系统执行，再把结果反馈给模型。

### 1.2 核心代码（agent.py:28-53）

```python
def chat(self, user_input: str) -> str:
    self._store_message("user", user_input)       # ① 存用户输入到记忆
    messages = self._build_messages(user_input)    # ② 组装消息：system + 历史 + 当前
    tools_schema = get_tools_schema(mock_mode=self.mock_mode)  # ③ 获取工具 Schema

    for round_num in range(self.max_rounds):       # ④ 最多 10 轮循环
        response = self._call_model(messages, tools_schema)  # ⑤ 调模型

        if response.get("tool_calls"):              # ⑥ 模型说"要调工具"
            tool_results = self._execute_tools(response["tool_calls"])  # ⑦ 执行工具
            for tr in tool_results:
                self._store_message("tool", tr["content"], tool_call_id=tr["tool_call_id"])
                messages.append({                    # ⑧ 工具结果放回对话
                    "role": "tool",
                    "content": tr["content"],
                    "tool_call_id": tr["tool_call_id"],
                })
            continue                                 # ⑨ 进入下一轮"再思考"

        content = response.get("content", "")        # ⑩ 无 tool_calls → 任务完成
        self._store_message("assistant", content)
        return content

    return f"[警告] 达到最大循环次数 {self.max_rounds}，返回当前结果。"
```

### 1.3 跨文件调用链

```
agent.chat() 被调用
    ↓ L30
agent._store_message() → memory.py:46  store()
    ↓ L32
agent._build_messages() → memory.py:66  search()     # 语义检索历史
    ↓ L33
tools.get_tools_schema() → tools.py:64  遍历 tool_registry 生成 JSON Schema
    ↓ L36
agent._call_model() → HTTP POST → serve.py:118  chat_completions()
    ↓ L127
serve.py → mock_model.py:21  generate()               # 模型推理
    ↓ L89
agent._execute_tools() → tools.py:81  execute_tool()   # 执行工具函数
    ↓ L41
agent._store_message() → memory.py:46  store()         # 存工具结果
```

### 1.4 多轮循环的典型场景

```
用户: "帮我查北京天气然后发邮件给我朋友"
Round 1: 模型 → tool_calls: [web_search("北京天气")]
         → agent 执行，天气结果放回 messages
Round 2: 模型看到天气结果
         → tool_calls: [send_email("天气数据", "friend@mail.com")]
         → agent 执行，发送结果放回 messages
Round 3: 模型看到发送成功
         → content: "已查询北京天气并发送邮件"
         → finish_reason: "stop" → 返回用户
```

### 1.5 面试点

**Q: 为什么需要循环？模型不能一步到位吗？**
> A: 模型只负责"决定调什么工具"，不负责"执行工具"。一个复杂任务需要多步工具调用才能完成（如先查天气再发邮件），所以需要循环。

**Q: max_rounds 为什么设 10？设太大或太小有什么问题？**
> A: 太小（如 3）可能完不成复杂任务；太大（如 100）可能陷入死循环（模型反复调同一个工具）。10 是经验值，覆盖大多数实际场景。

**Q: 一轮可以有多个 tool_calls 吗？怎么处理？**
> A: 可以。OpenAI API 允许一次返回多个 tool_calls。`_execute_tools`（agent.py:114-133）遍历所有 tool_calls，逐个执行，结果全部放回 messages。

**Q: 工具执行失败了怎么办？**
> A: `execute_tool`（tools.py:81-96）捕获异常返回 `{"error": "..."}` 字符串，作为 tool result 放回 messages。模型下一轮会看到错误信息，自主决定是否重试或换方案。

---

## 2. Function Calling — 格式转换链路

### 2.1 什么是 Function Calling

Function Calling 不是模型"执行"函数，而是模型**输出**应该调用哪个函数、参数是什么。执行由外部系统（agent）完成。这是 OpenAI 定义的标准协议。

### 2.2 完整数据流（6 个文件接力）

**第 1 棒：tools.py 注册工具（导入时自动执行）**

```python
# tools.py:8-9 全局注册表
tool_registry: Dict[str, dict] = {}

# tools.py:12-25 装饰器
def tool(description, parameters=None, requires_confirm=False):
    def decorator(func):
        name = func.__name__
        schema = _build_schema(func, parameters or {})  # 自动生成 JSON Schema
        tool_registry[name] = {
            "name": name,
            "description": description,
            "parameters": schema,      # JSON Schema 格式
            "function": func,          # 函数引用
            "requires_confirm": requires_confirm,
        }
        return func
    return decorator

# tools.py:133-138 实际注册
@tool(description="获取当前时间", parameters={"timezone": {"type": "string"}})
def get_time(timezone="Asia/Shanghai"):
    return datetime.datetime.now().isoformat()
# ↑ 执行时自动注册到 tool_registry["get_time"]
```

**第 2 棒：tools.py 生成 OpenAI 格式 Schema**

```python
# tools.py:64-78
def get_tools_schema(mock_mode=True):
    schemas = []
    for name, info in tool_registry.items():
        schemas.append({
            "type": "function",
            "function": {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"],  # JSON Schema
            }
        })
    return schemas
# 返回: [{"type": "function", "function": {"name": "get_time", ...}}, ...]
```

**第 3 棒：agent.py 把 Schema 放入 HTTP 请求**

```python
# agent.py:33 读取工具列表
tools_schema = get_tools_schema(mock_mode=self.mock_mode)

# agent.py:73-79 构造请求
payload = {
    "model": "minimind",
    "messages": messages,               # 消息列表
    "tools": tools_schema,              # ← 工具定义列表
    "stream": False,
}
# POST 到 serve.py
```

**第 4 棒：serve.py 传给模型并做格式转换**

```python
# serve.py:123-141
messages = [msg.dict(exclude_none=True) for msg in request.messages]
tools_schema = request.tools

raw_output = model.generate(messages, tools=tools_schema)
# raw_output = {"name": "get_time", "arguments": {"timezone": "Asia/Shanghai"}}
#                ↑ arguments 是 Python dict 对象

# 关键转换：dict → JSON 字符串
if isinstance(raw_output, dict) and "name" in raw_output:
    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": raw_output["name"],
            "arguments": json.dumps(raw_output.get("arguments", {}), ensure_ascii=False),
            # ↑ 关键！dict → JSON 字符串，符合 OpenAI 规范
        },
    }
    message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
    finish_reason = "tool_calls"
```

**第 5 棒：agent.py 反序列化 JSON 字符串**

```python
# agent.py:89-101
if message.get("tool_calls"):
    for tc in message["tool_calls"]:
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")    # JSON 字符串
        if isinstance(args_str, str):
            args = json.loads(args_str)             # JSON 字符串 → Python dict
        else:
            args = args_str                         # 兼容已是 dict 的情况
        tool_calls.append({
            "id": tc.get("id", ...),
            "function": {"name": func["name"], "arguments": args},
        })
```

**第 6 棒：tools.py 执行函数**

```python
# tools.py:81-96
def execute_tool(name, arguments, mock_mode=True):
    if name not in tool_registry:
        return json.dumps({"error": f"Tool '{name}' not found"})

    if mock_mode:
        return _mock_execute(name, arguments)   # 返回模拟数据
    else:
        result = func(**arguments)               # 真正执行函数
        return json.dumps({"result": result})
```

### 2.3 _build_schema 自动生成 JSON Schema

```python
# tools.py:28-61
def _build_schema(func, extra_params):
    import inspect
    sig = inspect.signature(func)
    properties = {}
    required = []

    # 从函数签名推断参数
    for pname, param in sig.parameters.items():
        if pname in extra_params:
            prop = extra_params[pname].copy()     # 有显式定义就用显式
        elif param.default is inspect.Parameter.empty:
            prop = {"type": "string"}              # 无默认值 → 必填
            required.append(pname)
        else:
            prop = {"type": "string"}              # 有默认值 → 可选
        properties[pname] = prop

    # 补充 extra_params 中有但函数签名里没有的参数
    for pname, pdef in extra_params.items():
        if pname not in properties:
            properties[pname] = pdef.copy()

    return {"type": "object", "properties": properties, "required": required}
```

### 2.4 Mock 工具执行

```python
# tools.py:99-128
def _mock_execute(name, arguments):
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
                {"title": f"Mock result {i+1}", "url": f"https://example.com/{i+1}"}
                for i in range(a.get("num_results", 3))
            ],
        },
    }
    return json.dumps(mock_data[name](arguments), ensure_ascii=False)
```

### 2.5 面试点

**Q: 为什么 arguments 必须是 JSON 字符串而不是 dict？**
> A: OpenAI 协议规定。原因：1) SSE 流式传输只能传字符串；2) 历史兼容；3) 不同客户端可用不同 JSON 解析器。serve.py:137 用 `json.dumps()` 做转换，agent.py:95 用 `json.loads()` 做反序列化。

**Q: Mock 模式下模型怎么知道该调哪个工具？**
> A: 关键词匹配（mock_model.py:62-80）。`"几点"` 匹配 `get_time`，`"文件"` 匹配 `read_file`，`"搜索"` 匹配 `web_search`。匹配不到时用第一个工具作为默认。

**Q: 为什么要在 tools.py 加 `requires_confirm`？**
> A: 安全机制。`run_shell` 可以执行任意命令，agent.py:122-129 在执行前检查此标记，弹出确认框让用户决定是否执行，防止 AI 自主执行破坏性命令。

---

## 3. 长期记忆 — ChromaDB 向量检索

### 3.1 为什么需要向量数据库

**传统方案（滑动窗口）**：把最近 N 条对话拼到 messages 里。缺点：长对话会丢失早期关键信息，token 限制也放不下所有历史。

**向量检索方案**：每轮对话转为向量（embedding），存到向量数据库。新对话时，用语义相似度检索与当前输入最相关的历史片段，注入 messages。优势：突破 token 限制，跨对话记忆，语义匹配而非简单关键词。

### 3.2 ChromaDB 初始化

```python
# memory.py:23-44
def _init_backend(self):
    os.makedirs(self.persist_dir, exist_ok=True)     # data/memory/ 目录
    if self.backend == "chromadb":
        import chromadb
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,                # "minimind_agent"
            metadata={"hnsw:space": "cosine"},        # 余弦相似度
        )
```

`hnsw:space: cosine` — 用余弦相似度衡量向量距离。对话场景语义相似度用 cosine 最合适（忽略文本长度差异）。其他选项：`l2`（欧氏距离）、`ip`（内积）。

### 3.3 存储（写入链路）

**触发点有 3 个**（agent.py）：

```python
# agent.py:30  存用户输入
self._store_message("user", user_input)

# agent.py:41  存工具执行结果
self._store_message("tool", tr["content"])

# agent.py:50  存模型最终回复
self._store_message("assistant", content)
```

**实际存储**（memory.py:46-64）：

```python
def store(self, conversation):
    content = conversation.get("content", "")
    role = conversation.get("role", "unknown")
    metadata = {
        "role": role,
        "timestamp": conversation.get("timestamp", ""),
        "session_id": conversation.get("session_id", "default"),
    }

    if self.backend == "chromadb":
        doc_id = f"{role}_{uuid.uuid4().hex[:8]}"    # 唯一 ID
        self._collection.add(
            documents=[content],                       # 文本（ChromaDB 自动 embedding）
            metadatas=[metadata],                      # 元数据（可过滤）
            ids=[doc_id],
        )
```

ChromaDB 在 `add()` 时自动调用内置的 `all-MiniLM-L6-v2` 模型将文本转为 384 维向量。

### 3.4 检索（读取链路）

```python
# agent.py:60-64  在 _build_messages 中调用
history = self.memory.search(user_input, top_k=3, session_id=self.session_id)
# ↓ 跳到 memory.py:66-87

def search(self, query, top_k=5, session_id="default"):
    results = self._collection.query(
        query_texts=[query],                           # 当前输入自动 embedding
        n_results=top_k,                               # 返回前 3 条
        where={"session_id": session_id},              # 只查当前会话
    )
    conversations = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        conversations.append({"role": meta["role"], "content": doc})
    return conversations
```

**注入到 messages 的位置**：

```python
# agent.py:57-69
messages = [{"role": "system", "content": self.system_prompt}]

# 记忆检索的历史插在 system 后面
for h in history:
    messages.append({"role": h["role"], "content": h["content"]})

messages.append({"role": "user", "content": user_input})
```

最终 messages 结构：
```
[system: "你是 MiniMind Agent..."]     ← 固定
[user: "昨天我说了什么"]                ← 记忆检索到的
[assistant: "你好！我是..."]            ← 记忆检索到的
[user: "现在几点了"]                    ← 当前输入
```

### 3.5 Session 隔离

```python
# agent.py:26  每个 Agent 实例生成唯一 session_id
self.session_id = uuid.uuid4().hex[:8]   # "a1b2c3d4"

# memory.py:53  写入时带 session_id
metadata = {"session_id": self.session_id}

# memory.py:72  检索时按 session_id 过滤
where={"session_id": session_id}
```

效果：每次重启 Agent 获得新 session，不同 session 的记忆互不干扰。

### 3.6 Fallback：内存后端

```python
# memory.py:38-40  配置 memory.backend: "none"
elif self.backend == "none":
    self._memory = []                           # 纯内存列表，不持久化

# memory.py:83-86  搜索时取最近 N 条
elif self.backend == "none":
    filtered = [m for m in self._memory if m["metadata"]["session_id"] == session_id]
    return [{"role": m["metadata"]["role"], "content": m["content"]} for m in filtered[-top_k:]]
```

当 ChromaDB 初始化失败时（如嵌入模型下载失败），自动降级为内存后端。

### 3.7 面试点

**Q: ChromaDB 和 Milvus 的区别？为什么选 ChromaDB 优先？**
> A: ChromaDB 是嵌入式数据库，不需要独立服务，进程内运行，适合轻量部署。Milvus 是分布式向量数据库，适合大规模生产环境，但需要 Docker 或 K8s 部署。本项目先用 ChromaDB 验证全流程，后续可切换 Milvus。

**Q: 为什么用余弦相似度而不是欧氏距离？**
> A: 余弦相似度衡量的是向量方向（语义方向）的差异，不受文本长度影响。"北京天气" 和 "北京的天气怎么样" 语义相同但长度不同，cosine 能正确匹配，L2 可能因为长度差异而降低相似度。

**Q: 向量检索的 top_k 怎么选？**
> A: top_k=3 是经验值。太小可能漏掉重要历史，太大可能注入无关信息干扰模型。生产环境中可以根据对话类型动态调整（如任务型对话用较小的 top_k，闲聊用较大的）。

---

## 4. OpenAI 兼容 API

### 4.1 为什么要兼容 OpenAI

OpenAI 的 Chat Completions API 已成为事实标准。LangChain、AutoGen、LlamaIndex、CrewAI 等主流框架都硬编码了 `/v1/chat/completions` 路径。只要 API 格式兼容，所有这些框架零侵入接入。

### 4.2 请求/响应格式（Pydantic 模型）

```python
# serve.py:55-91

# 请求格式
class ChatCompletionRequest(BaseModel):
    model: str = "minimind"
    messages: List[ChatMessage]       # [{role, content, tool_calls, tool_call_id}]
    tools: Optional[List[dict]]       # [{type, function: {name, description, parameters}}]
    tool_choice: Optional[str]        # "auto" | "none" | "required"
    stream: Optional[bool] = False    # SSE 流式输出
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

# 响应格式
class ChatCompletionResponse(BaseModel):
    id: str                           # "chatcmpl-{uuid}"
    object: str = "chat.completion"
    created: int                      # Unix 时间戳
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: dict                     # {role, content, tool_calls}
    finish_reason: str = "stop"       # "stop" | "tool_calls"
```

### 4.3 三个端点

```python
# serve.py:96-100  健康检查
@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": model.is_loaded, "mock": config["model"]["mock"]}

# serve.py:103-115  模型列表（LangChain 初始化时会调）
@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "minimind", "owned_by": "minimind-team"}]}

# serve.py:118-151  核心推理端点
@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    ...
```

### 4.4 模型加载与生命周期

```python
# serve.py:28-35  根据配置加载模型
def init_model():
    load_config()                                # 读 config.yaml
    if config["model"]["mock"]:
        model = MockModel(config["model"])
    else:
        # 后续扩展：DirectML / ONNX / CPU
        model = MockModel(config["model"])

# serve.py:38-41  FastAPI 生命周期（启动时加载模型）
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model()
    yield
    # shutdown 清理（当前没有）
```

### 4.5 CORS 配置

```python
# serve.py:45-50  允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # 允许所有来源
    allow_methods=["*"],     # 允许所有 HTTP 方法
    allow_headers=["*"],     # 允许所有请求头
)
```

没有 CORS 配置，Gradio WebUI（7860端口）无法调用 serve.py（8000端口）的 API。

### 4.6 第三方框架接入示例

```python
# OpenAI 客户端
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="mock")

# LangChain
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:8000/v1", api_key="mock", model="minimind")
```

### 4.7 面试点

**Q: OpenAI 兼容 API 的核心要点是什么？**
> A: 1) URL 路径 `/v1/chat/completions` 是硬编码标准；2) `finish_reason` 字段区分对话结束和工具调用；3) `id` 每次生成唯一值用于追踪；4) Pydantic 校验请求格式，非法请求返回 422。

**Q: stream 参数预留了但没实现，有什么难点？**
> A: SSE（Server-Sent Events）需要逐字符流式输出。Mock 模式下可以一次性返回，真实模型需要用 `async generator` 逐 token yield，前端也要用 SSE 解析器。Gradio 已支持 `gr.ChatInterface(streaming=True)`。
> 
> **已实现**：Mock 模式已支持 SSE 流式输出。`mock_model.py` 新增 `stream_generate()` 方法逐字符 yield；`serve.py` 根据 `stream` 参数返回 `StreamingResponse`；`agent.py` 新增 `chat_stream()` 生成器；CLI 逐字符打印，WebUI 用 Gradio generator 实现逐字显示。

**Q: 为什么不用 WebSocket 而用 HTTP？**
> A: Chat Completions API 标准就是 HTTP，框架生态都按这个来。WebSocket 适合实时双向通信（如语音助手），但 Function Calling 场景是请求-响应模式，HTTP 足够。

---

## 5. MCP 混合模式

### 5.1 什么是 MCP

MCP = Model Context Protocol，Anthropic 提出的开放协议，定义了 AI 模型发现和调用外部工具的标准接口。核心思想：工具不是硬编码在项目里，而是通过标准协议动态发现。

### 5.2 本项目中的两层设计

**第 1 层：内置工具（已实现）**

```python
# tools.py:8-9  全局注册表
tool_registry: Dict[str, dict] = {}

# tools.py:12-25  装饰器自动注册
def tool(description, parameters=None, requires_confirm=False):
    def decorator(func):
        tool_registry[func.__name__] = {...}
        return func
    return decorator

# tools.py:131-169  4 个预置工具
@tool(description="获取当前时间")
def get_time(timezone="Asia/Shanghai"): ...

@tool(description="读取文件内容", parameters={"path": {...}})
def read_file(path): ...

@tool(description="执行 Shell 命令", requires_confirm=True)
def run_shell(command): ...

@tool(description="网页搜索", parameters={"query": {...}, "num_results": {...}})
def web_search(query, num_results=5): ...
```

**第 2 层：外部 MCP Server（框架预留）**

```yaml
# config.yaml
tools:
  mcp_servers: []   # 用户填 MCP Server 地址
```

```python
# tools.py:64-78  当前只返回内置工具
def get_tools_schema(mock_mode=True):
    schemas = []
    for name, info in tool_registry.items():
        schemas.append(build_openai_schema(info))
    # TODO: 这里加 MCP Client 逻辑
    # for server in mcp_servers:
    #     schemas.extend(mcp_client.list_tools(server))
    return schemas
```

### 5.3 JSON Schema 自动生成

`_build_schema`（tools.py:28-61）从函数签名自动推断：

```python
@tool(
    description="网页搜索",
    parameters={
        "query": {"type": "string", "description": "搜索关键词", "required": True},
        "num_results": {"type": "integer", "description": "返回数量", "required": False},
    },
)
def web_search(query: str, num_results: int = 5) -> str:
    ...

# 自动生成的 JSON Schema:
# {
#     "type": "object",
#     "properties": {
#         "query": {"type": "string", "description": "搜索关键词"},
#         "num_results": {"type": "integer", "description": "返回数量"}
#     },
#     "required": ["query"]     # query 无默认值 → 必填
# }
```

### 5.4 requires_confirm 安全机制

```python
# tools.py:150-158  标记危险工具
@tool(description="执行 Shell 命令", requires_confirm=True)
def run_shell(command: str) -> str:
    result = subprocess.run(command, shell=True, ...)
    return result.stdout or result.stderr

# agent.py:122-129  执行前检查
if name in tool_registry and tool_registry[name].get("requires_confirm"):
    if not self._confirm_tool(name, arguments):
        results.append({"content": json.dumps({"error": "用户拒绝执行"})})
        continue
```

### 5.5 面试点

**Q: 为什么需要 MCP？装饰器注册不够吗？**
> A: 装饰器注册是静态工具（代码写死的），MCP 适合动态工具（外部服务提供的）。比如公司内部的 CRM API、数据库查询工具——这些不应该硬编码到 Agent 项目里，而是由独立服务提供，通过 MCP 协议动态发现。

**Q: 如何实现 MCP Client？**
> A: 用 `httpx` 连接 MCP Server 的 JSON-RPC 接口，调 `tools/list` 获取工具列表，调 `tools/call` 执行工具。将返回的 JSON Schema 转为 OpenAI 格式，合并到 `get_tools_schema()` 返回值中。

**Q: requires_confirm 在 WebUI 模式下怎么实现？**
> A: 当前只有 CLI 的 `input()` 确认。WebUI 模式下可以用 Gradio 的 `gr.Textbox` 弹出确认框，或者用 WebSocket 实现异步确认。这是一个可改进点。

---

## 6. Gradio WebUI

### 6.1 Gradio 的核心模型

Gradio 用**组件-事件-函数**三元组驱动：

```
组件（Textbox、Chatbot） ←→ 事件（submit、click） ←→ 函数（respond）
```

### 6.2 完整实现（main.py:53-80）

```python
def webui_mode(config, server_url):
    agent = create_agent(config, server_url)

    def respond(message, chat_history):
        response = agent.chat(message)       # 调用纯文本 ReAct 循环
        chat_history.append((message, response))
        return "", chat_history              # 清空输入框，更新对话

    with gr.Blocks(title="MiniMind Agent") as demo:
        gr.Markdown("# MiniMind Agent")

        with gr.Row():
            status = "已连接" if check_server(server_url) else "未连接"
            gr.Markdown(f"**状态:** {status} | **模型:** mock | **记忆:** chromadb")

        chatbot = gr.Chatbot(label="对话历史", height=500)  # 显示对话
        msg = gr.Textbox(label="输入消息", placeholder="请输入消息...", lines=2)
        clear = gr.Button("清空对话")

        msg.submit(respond, [msg, chatbot], [msg, chatbot])  # 回车触发
        clear.click(lambda: ("", []), None, [msg, chatbot])   # 清空

    demo.launch(server_name="0.0.0.0", server_port=7860)
```

### 6.3 Gradio 组件对应关系

| 组件 | 变量名 | 作用 |
|------|--------|------|
| `gr.Chatbot` | `chatbot` | 显示对话历史，[(user, bot), ...] 格式 |
| `gr.Textbox` | `msg` | 用户输入框 |
| `gr.Button` | `clear` | 清空对话按钮 |
| `gr.Markdown` | 状态栏 | 显示连接状态、模型类型、记忆后端 |

### 6.4 事件绑定

```python
# 用户按回车或点发送 → 调用 respond()
msg.submit(respond, [msg, chatbot], [msg, chatbot])
#                     输入参数      输出参数

# 用户点清空 → 重置 chatbot 和 msg
clear.click(lambda: ("", []), None, [msg, chatbot])
```

### 6.5 状态栏动态读取

```python
# main.py:70-71
status = "已连接" if check_server(server_url) else "未连接"
# check_server() 调 GET /health → 200 为已连接
```

### 6.6 当前局限

`agent.chat()` 返回最终结果，**中间的工具调用步骤没有展示到 UI**。改进方案：
- 让 `chat()` 返回一个 generator，逐步 yield 工具调用过程
- Gradio 用 `gr.Chatbot` 的 `append` 模式逐步更新界面
- 工具调用结果用不同颜色/格式区分显示

### 6.7 面试点

**Q: Gradio 和 Streamlit 选哪个？**
> A: Gradio 更适合聊天机器人（内置 Chatbot 组件、事件机制成熟、支持流式输出）。Streamlit 更适合数据分析面板（状态管理更简单、组件更丰富）。本项目选 Gradio 因为有原生 Chatbot 组件和 `msg.submit` 事件。

**Q: Gradio 如何处理并发请求？**
> A: Gradio 默认用 `threading` 处理每个请求。多个用户同时对话时，每个请求创建独立的 agent 实例（`create_agent` 在 `respond` 函数内调用），session_id 不同，记忆隔离。

---

## 7. FastAPI 后端开发

### 7.1 为什么选 FastAPI

FastAPI 是现代 Python Web 框架，基于 Starlette + Pydantic，天然支持异步、自动文档生成、类型校验。相比 Flask/Django，更适合构建高性能 API 服务。

| 特性 | FastAPI | Flask | Django |
|------|---------|-------|--------|
| 异步支持 | 原生 async/await | 需要 flask-async | 部分支持 |
| 自动文档 | Swagger/ReDoc | 需要插件 | 无 |
| 类型校验 | Pydantic 自动 | 手动 | Form/Model |
| 性能 | 接近 Node.js | 中等 | 较慢 |
| 学习曲线 | 低 | 最低 | 高 |

### 7.2 本项目中的 FastAPI 应用

**核心文件：serve.py**

```python
# serve.py:14-23  FastAPI 应用初始化
app = FastAPI(
    title="MiniMind API",
    description="OpenAI-compatible API server",
    version="1.0.0",
)

# serve.py:25-32  CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve.py:34-41  生命周期管理（启动时加载模型）
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model()
    yield
    # shutdown 清理（当前没有）
```

### 7.3 三个核心端点

**健康检查端点**

```python
# serve.py:43-50
@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_loaded": model.is_loaded,
        "mock": config["model"]["mock"]
    }
```

用途：客户端验证服务是否就绪，Gradio 状态栏显示连接状态。

**模型列表端点**

```python
# serve.py:52-58
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "minimind", "owned_by": "minimind-team"}]
    }
```

用途：LangChain、OpenAI SDK 初始化时会调用此端点验证模型可用性。

**核心推理端点**

```python
# serve.py:60-101
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [msg.dict(exclude_none=True) for msg in request.messages]
    tools_schema = request.tools

    raw_output = model.generate(messages, tools=tools_schema)

    # 格式转换：dict → OpenAI 格式
    if isinstance(raw_output, dict) and "name" in raw_output:
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
        message = {"role": "assistant", "content": raw_output}
        finish_reason = "stop"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model="minimind",
        choices=[ChatCompletionChoice(message=message, finish_reason=finish_reason)],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
```

### 7.4 Pydantic 数据模型

```python
# serve.py:63-89
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "minimind"
    messages: List[ChatMessage]
    tools: Optional[List[dict]] = None
    tool_choice: Optional[str] = "auto"
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
```

### 7.5 启动与部署

```python
# serve.py:103-104  uvicorn 启动
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
```

```bash
# 启动命令
python serve.py
# 或
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

### 7.6 自动文档

FastAPI 自动生成交互式 API 文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI Schema: `http://localhost:8000/openapi.json`

开发者可以直接在浏览器中测试所有端点，无需 Postman/curl。

### 7.7 错误处理

```python
# serve.py:62-64  模型未加载
if model is None:
    raise HTTPException(status_code=503, detail="Model not loaded")

# serve.py:82-84  请求格式错误（Pydantic 自动校验）
# 返回 422 Unprocessable Entity + 详细错误信息
```

### 7.8 面试点

**Q: FastAPI 的优势是什么？**
> A: 1) 原生 async/await 支持，适合高并发场景；2) Pydantic 自动校验请求/响应格式；3) 自动生成 OpenAPI 文档；4) 类型提示 + 依赖注入系统。

**Q: 为什么用 uvicorn 而不是 gunicorn？**
> A: uvicorn 是 ASGI 服务器，支持 async/await；gunicorn 是 WSGI 服务器，只支持同步。FastAPI 基于 ASGI，必须用 uvicorn 或 hypercorn。

**Q: 如何实现 SSE 流式输出？**
> A: 用 `StreamingResponse` + `async generator`：
> ```python
> @app.post("/v1/chat/completions")
> async def chat_completions(request: ChatCompletionRequest):
>     async def generate():
>         for chunk in model.stream_generate(messages, tools):
>             yield f"data: {json.dumps(chunk)}\n\n"
>     return StreamingResponse(generate(), media_type="text/event-stream")
> ```
> 客户端用 EventSource 或 SSE 解析器接收。

**Q: CORS 配置为什么用 `allow_origins=["*"]`？生产环境安全吗？**
> A: 开发阶段用 `*` 方便调试（Gradio 7860 调 serve.py 8000）。生产环境应限制为具体域名，如 `["https://yourapp.com"]`，防止 CSRF 攻击。

---

## 8. SSE 流式输出 — 逐 Token 推送

### 8.1 什么是 SSE

SSE = Server-Sent Events，HTTP 长连接协议，服务端单向推送数据给客户端。与 WebSocket 的区别：WebSocket 是双向通信，SSE 是单向推送。OpenAI Chat Completions API 的流式输出标准就是 SSE。

### 8.2 为什么需要流式

**非流式（同步）**：模型生成完整回复后一次性返回。用户等待时间 = 生成全部 token 的时间。

**流式（SSE）**：模型每生成一个 token 就推送给客户端。用户感知延迟 = 第一个 token 的时间（TTFT），后续 token 逐字出现，体验类似打字机。

| 模式 | 首字延迟 | 总时间 | 用户体验 |
|------|----------|--------|----------|
| 同步 | 高（等全部生成完） | 相同 | 差（长时间空白） |
| SSE 流式 | 低（第一个 token 立即返回） | 相同 | 好（逐字显示） |

### 8.3 内部 chunk 协议（agent ↔ main.py）

agent.py 和 main.py 之间用统一的 chunk 格式通信：

```python
# 3 种 chunk 类型
{"type": "chunk", "data": "你"}      # 单个 token
{"type": "tool_calls", "data": {...}}  # 工具调用（非流式，一次性返回）
{"type": "done", "data": ""}          # 生成结束
```

为什么用这个格式而不是直接传 SSE 原文？解耦——agent 不需要知道 SSE 协议细节，只关心 chunk 类型；main.py 决定如何展示（CLI 打印 / WebUI 更新）。

### 8.4 Mock 模型流式生成（mock_model.py）

```python
# mock_model.py 新增 stream_generate()
def stream_generate(self, messages, tools=None):
    import time

    if tools:
        # tool_calls 一次性返回，不流式
        result = self.generate(messages, tools=tools)
        yield {"type": "tool_calls", "data": result}
        return

    text = self._generate_text(messages[-1].get("content", ""), messages)
    for char in text:
        yield {"type": "chunk", "data": char}
        time.sleep(0.05)  # 50ms 延迟，模拟真实 token 生成速度

    yield {"type": "done", "data": ""}
```

`time.sleep(0.05)` 的意义：真实模型生成一个 token 约 20-100ms，50ms 是合理的模拟值。没有这个延迟，所有 token 瞬间返回，无法验证前端流式显示效果。

### 8.5 serve.py SSE 响应

```python
# serve.py 新增 StreamingResponse
from fastapi.responses import StreamingResponse

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 流式分支
    if request.stream:
        return StreamingResponse(
            stream_chat(request),
            media_type="text/event-stream",
        )

    # ... 原有非流式逻辑
```

**SSE 生成器**：

```python
async def stream_chat(request: ChatCompletionRequest):
    messages = [msg.dict(exclude_none=True) for msg in request.messages]
    tools_schema = request.tools
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    for chunk in model.stream_generate(messages, tools=tools_schema):
        if chunk["type"] == "chunk":
            sse_chunk = {
                "id": chat_id,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk["data"]},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(sse_chunk, ensure_ascii=False)}\n\n"

        elif chunk["type"] == "tool_calls":
            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": chunk["data"]["name"],
                    "arguments": json.dumps(chunk["data"].get("arguments", {}), ensure_ascii=False),
                },
            }
            sse_chunk = {
                "id": chat_id,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [tool_call]},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(sse_chunk, ensure_ascii=False)}\n\n"

        elif chunk["type"] == "done":
            yield "data: [DONE]\n\n"
            return
```

**SSE 响应格式**（每行一个 chunk）：

```
data: {"id":"chatcmpl-xxx","model":"minimind","choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","model":"minimind","choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}

data: [DONE]

```

`data: [DONE]\n\n` 是 OpenAI SSE 协议的结束标记，客户端收到后关闭连接。

### 8.6 agent.py 流式接收

```python
# agent.py 新增 chat_stream() 生成器
def chat_stream(self, user_input: str):
    """流式对话，yield 每个 token"""
    self._store_message("user", user_input)
    messages = self._build_messages(user_input)
    tools_schema = get_tools_schema(mock_mode=self.mock_mode)

    for round_num in range(self.max_rounds):
        tool_calls_acc = None
        content_acc = ""

        for chunk in self._call_model_stream(messages, tools_schema):
            if chunk["type"] == "chunk":
                content_acc += chunk["data"]
                yield {"type": "chunk", "data": chunk["data"]}
            elif chunk["type"] == "tool_calls":
                tool_calls_acc = chunk["data"]
            elif chunk["type"] == "done":
                break

        if tool_calls_acc:
            # 工具调用：执行 → 存记忆 → 追加到 messages → 继续循环
            tool_results = self._execute_tools(tool_calls_acc)
            for tr in tool_results:
                self._store_message("tool", tr["content"], tool_call_id=tr["tool_call_id"])
                messages.append({"role": "tool", "content": tr["content"], "tool_call_id": tr["tool_call_id"]})
            continue

        self._store_message("assistant", content_acc)
        yield {"type": "done", "data": ""}
        return

    yield {"type": "done", "data": ""}
```

**httpx 流式请求**：

```python
def _call_model_stream(self, messages, tools):
    """流式调用推理服务，yield SSE 解析后的 chunk"""
    payload = {
        "model": "minimind",
        "messages": messages,
        "tools": tools if tools else None,
        "stream": True,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    with httpx.Client(timeout=60) as client:
        with client.stream("POST", f"{self.server_url}/v1/chat/completions", json=payload) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # 去掉 "data: " 前缀
                if data_str == "[DONE]":
                    yield {"type": "done", "data": ""}
                    return
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                if "content" in delta:
                    yield {"type": "chunk", "data": delta["content"]}
                if "tool_calls" in delta:
                    yield {"type": "tool_calls", "data": delta["tool_calls"]}
```

`client.stream()` 打开 HTTP 长连接，`resp.iter_lines()` 逐行读取 SSE 数据。每行格式为 `data: {...}\n\n`。

### 8.7 main.py 流式显示

**CLI 模式**：

```python
def cli_mode(config, server_url):
    agent = create_agent(config, server_url)
    print("MiniMind Agent CLI (流式模式)")
    while True:
        user_input = input("\n你: ").strip()
        print("\nAgent: ", end="", flush=True)  # 不换行，等待流式输出
        for chunk in agent.chat_stream(user_input):
            if chunk["type"] == "chunk":
                print(chunk["data"], end="", flush=True)  # 逐字符打印
        print()  # 流结束后换行
```

`flush=True` 是关键——强制立即刷新输出缓冲区，否则字符会积攒后一起显示。

**WebUI 模式（Gradio generator）**：

```python
def respond(message, chat_history):
    chat_history.append((message, ""))  # 先加空回复
    for chunk in agent.chat_stream(message):
        if chunk["type"] == "chunk":
            # 逐步追加到最后一轮回复
            chat_history[-1] = (message, chat_history[-1][1] + chunk["data"])
            yield "", chat_history  # Gradio generator 返回值
```

Gradio 的 `yield` 机制：当响应函数是 generator 时，Gradio 会自动逐步更新 UI，实现逐字显示效果。

### 8.8 跨文件调用链（流式）

```
用户输入 "你好"
    │
    ▼ main.py
agent.chat_stream("你好")
    │
    ├─► memory.py       store("user", "你好")           ← 存用户输入
    │
    ├─► memory.py       search("你好", top_k=3)          ← 检索历史
    │
    ├─► tools.py        get_tools_schema()                ← 获取工具 Schema
    │
    ├─► agent._call_model_stream()
    │      ↓ httpx.stream("POST", serve.py)
    │
    │   serve.py        stream_chat()                     ← SSE 生成器
    │      ↓
    │   mock_model.stream_generate()                      ← 逐字符 yield
    │      ↓ yield {"type":"chunk","data":"你"}
    │      ↓ yield {"type":"chunk","data":"好"}
    │      ↓ yield {"type":"done"}
    │
    │   SSE: data: {"choices":[{"delta":{"content":"你"}}]}
    │   SSE: data: {"choices":[{"delta":{"content":"好"}}]}
    │   SSE: data: [DONE]
    │
    ├─► agent 解析 SSE → yield {"type":"chunk","data":"你"}
    │                 → yield {"type":"chunk","data":"好"}
    │                 → yield {"type":"done"}
    │
    ▼ main.py
CLI: print("你", flush=True) → print("好", flush=True)
WebUI: chat_history[-1] = ("你好","你") → yield
       chat_history[-1] = ("你好","你好") → yield
```

### 8.9 向后兼容设计

| 场景 | stream 参数 | 行为 |
|------|-------------|------|
| 默认调用 | `False` | 返回完整 JSON，行为不变 |
| 流式调用 | `True` | 返回 SSE 流，逐 token 推送 |
| tool_calls | `True` | tool_calls 仍一次性返回，不流式 |
| 向后兼容 | — | `chat()` 同步方法保留，`chat_stream()` 是新增 |

### 8.10 面试点

**Q: SSE 和 WebSocket 的区别？为什么选 SSE？**
> A: SSE 是单向推送（服务端 → 客户端），基于 HTTP，简单轻量；WebSocket 是双向通信，需要独立协议握手。Chat Completions API 是请求-响应模式，只需要服务端推送 token，SSE 足够。OpenAI 标准也是 SSE。

**Q: `data: [DONE]\n\n` 为什么重要？**
> A: 这是 SSE 协议的结束标记。客户端收到后必须关闭连接，否则会一直等待新数据。没有这个标记，客户端无法区分"生成结束"和"网络中断"。

**Q: `flush=True` 在 CLI 中的作用？**
> A: Python 的 `print()` 默认有缓冲区，字符会攒够一定量后才输出。`flush=True` 强制立即刷新，确保每个字符实时显示。没有这个参数，流式退化为块状输出。

**Q: Gradio generator 模式为什么能实现逐字显示？**
> A: Gradio 检测到响应函数是 generator 时，每次 `yield` 都会触发 UI 更新。`yield "", chat_history` 告诉 Gradio 用新的 `chat_history` 替换当前界面，实现逐步追加效果。

**Q: Mock 模式下 `time.sleep(0.05)` 的意义？**
> A: 模拟真实模型的 token 生成延迟（20-100ms/token）。没有延迟，所有 token 瞬间返回，无法验证前端流式显示逻辑是否正确。生产环境换成真实模型后，延迟由模型推理速度决定，不再需要 sleep。

**Q: 流式模式下工具调用为什么不流式？**
> A: 工具调用是结构化数据（函数名 + 参数 JSON），需要完整性保证。流式输出 JSON 片段会导致解析困难。OpenAI 标准中 tool_calls 也是一次性返回，只对纯文本内容做流式。

---

## 跨功能协作总结

### 完整调用链（一次用户输入的完整旅程）

```
用户输入 "现在几点了"
    │
    ▼ main.py:43 / main.py:62
agent.chat("现在几点了")
    │
    ├─► memory.py:46      store("user", "现在几点了")     ← 存用户输入
    │
    ├─► memory.py:66      search("现在几点了", top_k=3)   ← 检索相关历史
    │                         ↓ 注入 messages
    │
    ├─► tools.py:64       get_tools_schema()               ← 获取工具 JSON Schema
    │
    ├─► serve.py:118      POST /v1/chat/completions        ← HTTP 请求
    │      ↓
    │   serve.py:127      model.generate(messages, tools)  ← 调模型
    │      ↓
    │   mock_model.py:58  _generate_tool_call()            ← 关键词匹配
    │      ↓
    │   serve.py:130      json.dumps(arguments)            ← dict → JSON 字符串
    │      ↓
    │   返回 OpenAI 格式响应
    │
    ├─► agent.py:95       json.loads(args_str)             ← JSON 字符串 → dict
    │
    ├─► agent.py:131      execute_tool("get_time", {})     ← 执行工具
    │      ↓
    │   tools.py:90       _mock_execute("get_time", {})    ← 返回模拟数据
    │      ↓
    │   返回 '{"_mock": true, "result": "2024-01-15T12:00:00"}'
    │
    ├─► memory.py:46      store("tool", '{"_mock":true,}')  ← 存工具结果
    │
    ├─► messages 追加     {"role": "tool", "content": "..."}
    │
    ├─► 再调 serve.py:118  模型看到工具结果
    │      ↓
    │   mock_model.py:41  _generate_text()                  ← 返回最终文本
    │      ↓
    │   "现在是 2024年1月15日 12:00:00"
    │
    ├─► memory.py:46      store("assistant", "现在是...")   ← 存最终回复
    │
    ▼
返回给用户: "现在是 2024年1月15日 12:00:00"
```

### 层与层的依赖关系

```
依赖方向（上层依赖下层，下层不知道上层）：

main.py
  ↓ 依赖 agent.create_agent()
agent.py
  ↓ 依赖 memory.create_memory()     → memory.py
  ↓ 依赖 tools.get_tools_schema()   → tools.py
  ↓ 依赖 tools.execute_tool()       → tools.py
  ↓ 依赖 httpx POST                 → serve.py
serve.py
  ↓ 依赖 mock_model.MockModel       → mock_model.py
  ↓ 依赖 tools.get_tools_schema()   → tools.py（预留，当前未用）
  ↓ 依赖 config.yaml                → 全局配置

mock_model.py  → 独立，不依赖其他文件
tools.py       → 独立，不依赖其他文件
memory.py      → 独立，不依赖其他文件（只依赖 chromadb）
```

**面试总结**：这种分层设计遵循**依赖倒置原则**——高层模块不依赖低层实现，而是依赖接口。换模型、换记忆后端、加工具，都只改单一文件，不影响其他层。
