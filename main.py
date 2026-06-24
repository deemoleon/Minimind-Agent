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
    """CLI 交互模式（流式）"""
    from agent import create_agent

    agent = create_agent(config, server_url)
    print("MiniMind Agent CLI (流式模式)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("再见！")
                break
            if not user_input:
                continue

            print("\nAgent: ", end="", flush=True)
            for chunk in agent.chat_stream(user_input):
                if chunk["type"] == "chunk":
                    print(chunk["data"], end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n[错误] {e}")


def webui_mode(config: dict, server_url: str):
    """Gradio WebUI 模式（流式）"""
    import gradio as gr
    from agent import create_agent

    agent = create_agent(config, server_url)

    def respond(message, chat_history):
        """处理用户消息并返回回复（流式）"""
        chat_history.append((message, ""))
        for chunk in agent.chat_stream(message):
            if chunk["type"] == "chunk":
                chat_history[-1] = (message, chat_history[-1][1] + chunk["data"])
                yield "", chat_history

    with gr.Blocks(title="MiniMind Agent") as demo:
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