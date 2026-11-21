"""与 OpenAI 兼容 Chat Completions API 相关的共享类型。

本文件只有类型（TypedDict / 类型别名），不含任何运行时代码；
网络请求见 llm.py，主循环见 agent.py。
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Callable, TypedDict


class FunctionCall(TypedDict):
    """模型给出的函数调用明细（名字 + JSON 字符串形式的参数）"""

    name: str
    arguments: str


class ToolCall(TypedDict):
    """模型返回的一次工具调用请求"""

    id: str
    type: str  # 恒为 "function"
    function: FunctionCall


class ChatMessage(TypedDict, total=False):
    """对话消息（含 tool 角色），与 API 的 messages 数组结构一致。

    total=False：role/content 等字段在构造时总是给出，
    但 dict 在运行时天然可以缺字段，TypedDict 只做静态提示。
    """

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None
    tool_calls: list[ToolCall]
    tool_call_id: str


class FunctionSchema(TypedDict):
    """传给 API 的工具 JSON Schema 定义（function 部分）"""

    name: str
    description: str
    parameters: Mapping[str, Any]


class ToolSchema(TypedDict):
    """传给 API 的完整工具定义（外层 type + function）"""

    type: str  # 恒为 "function"
    function: FunctionSchema


# 工具执行函数的签名：接收参数 dict，返回任意可 JSON 化的结果（Agent 会把它字符串化）。
# 用 Callable 别名而非 Protocol——表达的是同一件事，对初学者少一个新概念。
ToolFunc = Callable[[Mapping[str, Any]], Any]


class LLMConfig(TypedDict):
    """LLM 服务配置，由 llm.py 的 load_config() 从环境变量构造"""

    base_url: str
    api_key: str
    model: str
