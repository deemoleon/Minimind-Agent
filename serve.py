"""OpenAI 兼容推理服务 — FastAPI 应用，实现 /health, /v1/models, /v1/chat/completions"""

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


async def stream_chat(request: ChatCompletionRequest):
    """SSE 流式生成器"""
    messages = [msg.dict(exclude_none=True) for msg in request.messages]
    tools_schema = request.tools
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    for chunk in model.stream_generate(messages, tools=tools_schema):
        if chunk["type"] == "tool_calls":
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
                "choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(sse_chunk, ensure_ascii=False)}\n\n"

        elif chunk["type"] == "chunk":
            sse_chunk = {
                "id": chat_id,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": chunk["data"]}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(sse_chunk, ensure_ascii=False)}\n\n"

        elif chunk["type"] == "done":
            yield "data: [DONE]\n\n"
            return


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.stream:
        return StreamingResponse(stream_chat(request), media_type="text/event-stream")

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
