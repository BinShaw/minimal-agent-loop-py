"""核心 ReAct 循环：推理（Reason）→ 行动（Act）→ 观察（Observe），循环直到模型给出最终回答。

一轮迭代：
  1. Reason: 把整个对话（含之前的工具观察）发给 LLM
  2. Act:    若模型返回 tool_calls → 执行工具，把结果作为 tool 消息追加进历史（Observe）
  3.         若无 tool_calls → 模型已得出答案，循环结束
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .llm import chat
from .types import ChatMessage, LLMConfig, ToolCall
from .tools import Tool

MAX_STEPS_DEFAULT = 10

# "问模型"这件事的签名——Agent 只依赖它，不依赖具体网络实现，
# 测试时替换成一个假函数即可离线跑通整个循环（见 tests/test_agent.py）。
ChatFn = Callable[[list[ChatMessage], list, LLMConfig], ChatMessage]


@dataclass
class AgentStep:
    """一轮迭代的可见产物，供 on_step 回调打印过程"""

    text: str
    """本轮模型的思考/回答文本（可能为空）"""
    tool_calls: list[ToolCall]
    """本轮请求执行的工具调用（空列表表示循环结束）"""
    results: dict[str, str]
    """各工具的执行结果，key 为 tool_call id"""


# onStep 的 Python 形态：可选回调，缺省为 None
OnStep = Callable[[AgentStep, int], None]


@dataclass
class AgentOptions:
    system_prompt: str | None = None
    max_steps: int | None = None
    on_step: OnStep | None = None


class Agent:
    def __init__(
        self,
        config: LLMConfig,
        tools: list[Tool],
        chat_fn: ChatFn = chat,
    ):
        """chat_fn 仅用于测试时替换真实 LLM，生产调用无需传入"""
        self._config = config
        self._tools = tools
        self._tool_map = {t.name: t for t in tools}
        self._chat = chat_fn

    def run(self, question: str, options: AgentOptions | None = None) -> str:
        """处理一个用户问题，返回最终回答文本"""
        opts = options or AgentOptions()
        max_steps = opts.max_steps if opts.max_steps is not None else MAX_STEPS_DEFAULT
        messages: list[ChatMessage] = [
            {"role": "system", "content": opts.system_prompt or "你是一个乐于助人的助手。"},
            {"role": "user", "content": question},
        ]

        # 这个循环只有一个正常出口：模型不再索要工具时，在循环内部 return。
        # 如果 for 自然跑完还没返回，说明模型一直在要工具（可能陷入死循环），属于异常。
        for step_num in range(1, max_steps + 1):
            # ---- Reason: 模型基于全部历史思考下一步 ----
            assistant = self._chat(messages, [t.schema for t in self._tools], self._config)

            tool_calls: list[ToolCall] = assistant.get("tool_calls") or []
            text = assistant.get("content") or ""

            # 没有 tool_calls → 模型认为任务完成，返回最终答案
            if not tool_calls:
                if opts.on_step:
                    opts.on_step(AgentStep(text=text, tool_calls=[], results={}), step_num)
                return text

            # ---- Act: 执行每个工具调用，结果作为 Observation ----
            results: dict[str, str] = {}
            # 必须先把 assistant 的 tool_calls 消息存进历史，再追加 tool 结果。
            # 少了这一步，下一轮请求里 tool 消息的 tool_call_id 就找不到对应的调用，
            # 服务端会直接报 400（这是自己复现 ReAct 时最常踩的坑）。
            messages.append(assistant)
            for call in tool_calls:
                output = self._execute_tool(call)
                results[call["id"]] = output
                messages.append(
                    {"role": "tool", "content": output, "tool_call_id": call["id"]}
                )

            if opts.on_step:
                opts.on_step(
                    AgentStep(text=text, tool_calls=tool_calls, results=results), step_num
                )

        raise RuntimeError(f"已达到最大步数 {max_steps}，仍未得到最终回答。可调大 max_steps 后重试。")

    def _execute_tool(self, call: ToolCall) -> str:
        """执行单个工具调用。

        核心原则：任何异常都转成文本观察值喂回模型，而不是抛出去。
        错误成为观察值，模型才有机会在下一轮自行修正；抛出去就等于替模型终止了对话。
        """
        name = call["function"]["name"]
        # 情况 1：模型幻觉出未注册的工具名，如实告诉它"没有这个工具"
        tool = self._tool_map.get(name)
        if tool is None:
            return f'错误: 未知工具 "{name}"'

        raw_args = call["function"]["arguments"]
        try:
            # 无参数工具有时返回空字符串，按 {} 处理
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            # 情况 2：参数不是合法 JSON，把原始参数一并回喂，便于它照着改正重试
            return f'错误: 工具 "{name}" 的参数不是合法 JSON: {raw_args}'

        try:
            result = tool.handler(args)
            # 无论工具返回什么，最终都要变成字符串才能进入对话历史
            return result if isinstance(result, str) else json.dumps(
                result, ensure_ascii=False
            )
        except Exception as e:  # noqa: BLE001
            # 情况 3：工具自身执行失败（业务错误），同样只作为观察值，不中断循环
            return f"错误: {e}"
