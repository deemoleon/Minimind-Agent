"""Mock 模型 — 微型 GPT 结构（2层/128维），CPU 可运行，用于全流程验证"""

import json
import random
import time
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

    def stream_generate(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
    ):
        """流式生成，yield 每个 token"""
        if tools:
            result = self.generate(messages, tools=tools)
            yield {"type": "tool_calls", "data": result}
            return

        last_msg = messages[-1] if messages else {}
        content = last_msg.get("content", "") if isinstance(last_msg, dict) else ""
        text = self._generate_text(content, messages)

        for char in text:
            yield {"type": "chunk", "data": char}
            time.sleep(0.05)

        yield {"type": "done", "data": ""}
