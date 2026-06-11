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
        try:
            history = self.memory.search(user_input, top_k=3, session_id=self.session_id)
            for h in history:
                if h["role"] in ("user", "assistant"):
                    messages.append({"role": h["role"], "content": h["content"]})
        except Exception as e:
            print(f"[警告] 记忆检索失败: {e}")

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
        try:
            self.memory.store({
                "role": role,
                "content": content,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": self.session_id,
            })
        except Exception as e:
            print(f"[警告] 记忆存储失败: {e}")

    def check_server(self) -> bool:
        """检查推理服务是否可用"""
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.server_url}/v1/models")
                return resp.status_code == 200
        except Exception:
            return False


def create_agent(config: dict, server_url: str = "http://localhost:8000") -> Agent:
    """工厂函数创建 Agent"""
    return Agent(config, server_url)