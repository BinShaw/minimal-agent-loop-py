"""CLI 入口：加载配置和工具，进入交互式问答循环。

用法: uv run minimal-agent-loop  或  python -m minimal_agent_loop.cli
"""
from __future__ import annotations

import sys

from .agent import Agent, AgentOptions, AgentStep
from .llm import load_config
from .tools import create_default_tools

CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"

# Windows 旧终端（如 cmd）默认不识别 ANSI 转义，先尝试启用
if sys.platform == "win32":
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:  # noqa: BLE001
        pass


def _on_step(step: AgentStep, n: int) -> None:
    # 最后一步没有工具调用，其 text 就是最终答案，交给下方统一打印
    if not step.tool_calls:
        return
    # Reason：打印模型的思考文本，ReAct 的"想"也应该看得见
    if step.text:
        print(f"{DIM}[step {n}] 思考: {step.text}{RESET}")
    # Act + Observe：调用了哪个工具、参数是什么、观察到了什么
    for call in step.tool_calls:
        preview = step.results.get(call["id"], "")[:200]
        print(
            f"{DIM}[step {n}] 调用 {call['function']['name']}"
            f"({call['function']['arguments']}) → {preview}{RESET}"
        )


def main() -> int:
    try:
        config = load_config()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    agent = Agent(config, create_default_tools())

    print(f"{CYAN}minimal-agent-loop{RESET}  model={config['model']}  baseUrl={config['base_url']}")
    print(f"{DIM}输入问题开始对话，Ctrl+C 退出。{RESET}\n")

    try:
        while True:
            try:
                question = input(f"{YELLOW}你>{RESET} ").strip()
            except EOFError:
                break
            if not question:
                continue
            if question in ("exit", "quit"):
                break

            try:
                answer = agent.run(question, AgentOptions(on_step=_on_step))
                print(f"\n{CYAN}助手>{RESET} {answer}\n")
            except Exception as e:  # noqa: BLE001
                print(f"\n{YELLOW}出错:{RESET} {e}\n", file=sys.stderr)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
