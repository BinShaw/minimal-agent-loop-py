"""工具层：定义 + 执行，故意保持极简。

每个工具 = schema（给模型看的 JSON Schema）+ handler（真正执行的函数）。
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .types import FunctionSchema, ToolFunc, ToolSchema


@dataclass(frozen=True)
class Tool:
    """一个已注册的工具。

    name:     本地路由 key，Agent 用它找到要执行的 handler
    schema:   发给模型的线上协议，其 function.name 与 name 是同一个名字
    handler:  真正执行的函数
    """

    name: str
    schema: ToolSchema
    handler: ToolFunc


@dataclass
class ToolDefinition:
    """define_tool 的入参：只关心业务字段，嵌套样板由工厂组装"""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolFunc


def define_tool(defn: ToolDefinition) -> Tool:
    """组装一个工具，把 JSON Schema 的嵌套样板收进工厂内部。

    name 同时填入本地路由 key 和 schema.function.name，避免两处手写不一致。
    """
    return Tool(
        name=defn.name,
        schema=ToolSchema(
            type="function",
            function=FunctionSchema(
                name=defn.name,
                description=defn.description,
                parameters=dict(defn.parameters),
            ),
        ),
        handler=defn.handler,
    )


def create_default_tools() -> list[Tool]:
    """内置工具集：calculate + get_current_time"""

    def calculate(args: Mapping[str, Any]) -> float:
        expression = str(args.get("expression", "")).strip()
        # 白名单只放行数学字符。禁掉字母是安全的关键：没有字母就拼不出任何
        # 标识符（__import__、os、open 等），表达式便够不到 Python 本身的能力。
        # 逗号也不放行：无字母的算式里，逗号唯一能拼出的是元组 (1,2)，没有正当用途
        if not re.fullmatch(r"[-+*/%().\d\s]+", expression):
            raise ValueError("表达式含非法字符，只支持数字、小数点和 + - * / % ( )")
        # 单字符白名单挡不住字符的「组合」：* 和 * 拼出 ** 后，9**9**9 会在
        # eval 阶段就耗尽 CPU/内存（轮不到下面的类型检查），// 同理，显式拒绝。
        if "**" in expression or "//" in expression:
            raise ValueError("不支持 ** 和 //（防止超大数运算卡死进程）")
        # 教学简化：真实项目请改用表达式解析器（如 sympy / ast 安全求值）。
        # 这里为了零依赖采用「白名单 + eval」，绝不能套用到未过滤的输入上。
        # 字符集只含数字、运算符和括号，eval 的结果必然是数值。
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        if not isinstance(result, (int, float)) or result != result:  # NaN 检查
            raise ValueError(f"无法求值: {expression}")
        return result

    def get_current_time(args: Mapping[str, Any]) -> str:
        return datetime.now(timezone.utc).isoformat()

    return [
        define_tool(
            ToolDefinition(
                name="calculate",
                description="计算一个数学表达式并返回结果",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "例如 2 + 3 * 4"},
                    },
                    "required": ["expression"],
                },
                handler=calculate,
            )
        ),
        define_tool(
            ToolDefinition(
                name="get_current_time",
                description="获取当前时间",
                parameters={"type": "object", "properties": {}},
                handler=get_current_time,
            )
        ),
    ]
