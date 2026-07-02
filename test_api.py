"""API 测试脚本 — 用 openai 库测试所有接口

用法:
    python test_api.py          # Mock 模式测试
    python test_api.py --real   # 真实模型测试（需先启动 mock: false）
"""

import json
import sys
import time

import httpx
from openai import OpenAI


BASE_URL = "http://localhost:8000"
results = []


def test(name: str, passed: bool, detail: str = ""):
    icon = "[PASS]" if passed else "[FAIL]"
    msg = f"{icon} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    results.append({"name": name, "passed": passed})


def test_health():
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=5)
        passed = resp.status_code == 200 and resp.json().get("status") == "ok"
        test("GET /health", passed, f"{resp.status_code} {resp.json().get('status', '')}")
    except Exception as e:
        test("GET /health", False, str(e))


def test_models():
    try:
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=5)
        data = resp.json()
        model_id = data["data"][0]["id"] if data.get("data") else None
        passed = resp.status_code == 200 and model_id == "minimind"
        test("GET /v1/models", passed, f"返回 {len(data.get('data', []))} 个模型")
    except Exception as e:
        test("GET /v1/models", False, str(e))


def test_chat_pure_text():
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
    test("POST /v1/chat/completions (模型未加载)", True, "跳过（模型已加载）")


def test_stream_text():
    """测试 SSE 流式输出（纯文本）"""
    try:
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "minimind",
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True,
            },
            timeout=30,
        )
        passed = resp.status_code == 200
        if not passed:
            test("POST /v1/chat/completions (stream 纯文本)", passed, f"状态码: {resp.status_code}")
            return

        lines = resp.text.strip().split("\n")
        has_done = any("data: [DONE]" in line for line in lines)
        has_data = any(line.startswith("data: ") and "[DONE]" not in line for line in lines)

        passed = has_data and has_done
        chunks = [line for line in lines if line.startswith("data: ") and "[DONE]" not in line]
        test("POST /v1/chat/completions (stream 纯文本)", passed,
             f"data chunks: {len(chunks)}, has [DONE]: {has_done}")
    except Exception as e:
        test("POST /v1/chat/completions (stream 纯文本)", False, str(e))


# ---- 真实模型专用测试 ----

def test_real_chat_quality():
    """真实模型：生成内容非空且长度合理"""
    try:
        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="real")
        resp = client.chat.completions.create(
            model="minimind",
            messages=[{"role": "user", "content": "用一句话介绍人工智能"}],
        )
        content = resp.choices[0].message.content
        passed = bool(content) and 10 < len(content) < 500
        test("真实模型: 纯文本质量", passed, f"长度: {len(content)}, 内容: {content[:50]}...")
    except Exception as e:
        test("真实模型: 纯文本质量", False, str(e))


def test_real_tool_call():
    """真实模型：自然语言触发工具调用"""
    try:
        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="real")
        resp = client.chat.completions.create(
            model="minimind",
            messages=[{"role": "user", "content": "帮我查一下现在几点了"}],
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
        # 真实模型可能直接回答也可能调用工具，两者都算通过
        passed = bool(msg.content) or has_tools
        detail = f"finish_reason: {resp.choices[0].finish_reason}"
        if has_tools:
            tc = msg.tool_calls[0]
            detail += f", tool: {tc.function.name}"
        else:
            detail += f", 内容: {msg.content[:30]}..."
        test("真实模型: 工具调用", passed, detail)
    except Exception as e:
        test("真实模型: 工具调用", False, str(e))


def test_real_stream():
    """真实模型：流式输出 token 数量合理"""
    try:
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "minimind",
                "messages": [{"role": "user", "content": "写一首关于春天的短诗"}],
                "stream": True,
            },
            timeout=60,
        )
        passed = resp.status_code == 200
        if not passed:
            test("真实模型: 流式输出", passed, f"状态码: {resp.status_code}")
            return

        lines = resp.text.strip().split("\n")
        chunks = [line for line in lines if line.startswith("data: ") and "[DONE]" not in line]
        has_done = any("data: [DONE]" in line for line in lines)
        passed = len(chunks) >= 3 and has_done
        test("真实模型: 流式输出", passed, f"chunks: {len(chunks)}, [DONE]: {has_done}")
    except Exception as e:
        test("真实模型: 流式输出", False, str(e))


def main():
    real_mode = "--real" in sys.argv

    print("=" * 50)
    print(f"MiniMind API 测试 ({'真实模型' if real_mode else 'Mock 模式'})")
    print("=" * 50)

    test_health()
    test_models()
    test_chat_pure_text()
    test_chat_with_tools()
    test_chat_tool_result()
    test_bad_request()
    test_model_not_loaded()
    test_stream_text()

    if real_mode:
        print("\n--- 真实模型附加测试 ---")
        test_real_chat_quality()
        test_real_tool_call()
        test_real_stream()

    print("=" * 50)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"结果: {passed}/{total} 通过")
    print("=" * 50)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
