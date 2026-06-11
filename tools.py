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