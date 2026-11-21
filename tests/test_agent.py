"""离线冒烟测试：用脚本化的假 chat 函数验证 ReAct 循环逻辑
（多轮工具调用、历史累积、错误观察、正常终止）。运行: uv run unittest

关键手法：Agent 构造器的第三个参数可以替换"问模型"这件事，
所以不需要 API key 就能测试整个循环。
"""
from __future__ import annotations

import copy
import json
import unittest
from typing import Any, Mapping

from minimal_agent_loop.agent import Agent, AgentOptions, AgentStep
from minimal_agent_loop.tools import Tool, ToolDefinition, define_tool
from minimal_agent_loop.types import ChatMessage, LLMConfig, ToolCall

FAKE_CONFIG: LLMConfig = {"base_url": "http://fake", "api_key": "fake", "model": "fake"}


def scripted_chat(*replies: ChatMessage):
    """假 LLM：第 n 次被调用就返回剧本里的第 n 条消息。

    剧本用完后一直重复最后一条（最终回答 → 正常结束；工具调用 → 可用来模拟死循环）。
    """
    i = 0

    def _chat(messages, tools, config):
        nonlocal i
        reply = replies[min(i, len(replies) - 1)]
        i += 1
        return copy.deepcopy(reply)

    return _chat


def make_tool(name: str) -> Tool:
    def handler(args: Mapping[str, Any]) -> str:
        return f"echo:{name}:{args.get('input', '')}"

    return define_tool(
        ToolDefinition(
            name=name,
            description=f"test tool {name}",
            parameters={"type": "object", "properties": {}},
            handler=handler,
        )
    )


def tool_call(id: str, name: str, args: dict[str, Any]) -> ToolCall:
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def tool_request(text: str, *calls: ToolCall) -> ChatMessage:
    """一条带工具调用的 assistant 消息：循环会继续"""
    return {"role": "assistant", "content": text, "tool_calls": list(calls)}


def text_reply(text: str) -> ChatMessage:
    """一条纯文本的 assistant 消息：没有 tool_calls，循环到此结束"""
    return {"role": "assistant", "content": text}


class MultiTurnToolTest(unittest.TestCase):
    def test_multi_turn_then_final_answer(self):
        """多轮工具调用后给出最终回答"""
        steps: list[AgentStep] = []
        agent = Agent(
            FAKE_CONFIG,
            [make_tool("get_time"), make_tool("calc")],
            scripted_chat(
                tool_request("先查时间", tool_call("c1", "get_time", {})),
                tool_request("再算个数", tool_call("c2", "calc", {"input": 42})),
                text_reply("最终答案"),
            ),
        )

        answer = agent.run(
            "问题", AgentOptions(on_step=lambda s, n: steps.append(s))
        )

        self.assertEqual(answer, "最终答案")
        self.assertEqual(len(steps), 3)  # 2 次工具步骤 + 1 次最终回答
        self.assertEqual(steps[0].text, "先查时间")  # Reason 也会传给 on_step
        self.assertEqual(steps[0].tool_calls[0]["function"]["name"], "get_time")
        self.assertEqual(steps[1].results["c2"], "echo:calc:42")
        self.assertEqual(steps[2].tool_calls, [])  # 最后一步没有工具调用


class HistoryAccumulationTest(unittest.TestCase):
    def test_history_accumulates_in_pairs(self):
        """对话历史随循环累积：assistant 的工具调用与 tool 观察值成对出现"""
        snapshots: list[list[ChatMessage]] = []
        replies = [
            tool_request("先算个数", tool_call("c1", "calc", {"input": 1})),
            text_reply("算完了"),
        ]
        i = 0

        # 除了按剧本回复，还把每轮发出去的历史拍个快照，看看循环到底给模型看了什么
        def recording_chat(messages, tools, config):
            nonlocal i
            snapshots.append(copy.deepcopy(messages))
            reply = replies[min(i, len(replies) - 1)]
            i += 1
            return copy.deepcopy(reply)

        agent = Agent(FAKE_CONFIG, [make_tool("calc")], recording_chat)
        agent.run("1+1 等于几")

        self.assertEqual(len(snapshots), 2)  # 工具一轮 + 最终回答一轮
        self.assertEqual(
            [m["role"] for m in snapshots[0]], ["system", "user"]
        )
        self.assertEqual(
            [m["role"] for m in snapshots[1]],
            ["system", "user", "assistant", "tool"],
        )

        assistant_msg, tool_msg = snapshots[1][2], snapshots[1][3]
        self.assertEqual(assistant_msg["tool_calls"][0]["id"], "c1")
        # tool 消息靠 tool_call_id 与上面的调用配对；缺了 assistant 那条，API 会报 400
        self.assertEqual(tool_msg["tool_call_id"], "c1")
        self.assertEqual(tool_msg["content"], "echo:calc:1")


class ToolErrorTest(unittest.TestCase):
    def test_error_becomes_observation(self):
        """工具抛错时错误作为观察值返回，模型仍能得到最终回答"""
        steps: list[AgentStep] = []

        def boom(args: Mapping[str, Any]) -> str:
            raise ValueError("炸了")

        failing = define_tool(
            ToolDefinition(
                name="boom",
                description="总是失败",
                parameters={"type": "object", "properties": {}},
                handler=boom,
            )
        )
        agent = Agent(
            FAKE_CONFIG,
            [failing],
            scripted_chat(
                tool_request("", tool_call("c1", "boom", {})),
                text_reply("看到错误了"),
            ),
        )

        answer = agent.run(
            "问题", AgentOptions(on_step=lambda s, n: steps.append(s))
        )

        self.assertEqual(answer, "看到错误了")
        # 关键：真的执行过一次工具，而不是第一轮就直接收尾
        self.assertEqual(len(steps), 2)
        self.assertIn("炸了", steps[0].results["c1"])


class MaxStepsTest(unittest.TestCase):
    def test_max_steps_raises(self):
        """达到 max_steps 上限时抛出错误"""
        # 剧本只有一条工具调用消息，之后每次都会重复它，模拟模型陷入死循环
        agent = Agent(
            FAKE_CONFIG,
            [make_tool("loop")],
            scripted_chat(tool_request("", tool_call("c1", "loop", {}))),
        )
        with self.assertRaisesRegex(RuntimeError, "最大步数"):
            agent.run("问题", AgentOptions(max_steps=3))


class BadRequestsTest(unittest.TestCase):
    def test_unknown_tool_and_bad_json(self):
        """未知工具与非法 JSON 参数都会返回错误观察值"""
        steps: list[AgentStep] = []
        agent = Agent(
            FAKE_CONFIG,
            [make_tool("calc")],
            scripted_chat(
                tool_request(
                    "",
                    {"id": "c1", "type": "function",
                     "function": {"name": "nope", "arguments": "{}"}},
                    {"id": "c2", "type": "function",
                     "function": {"name": "calc", "arguments": "{bad json"}},
                ),
                text_reply("done"),
            ),
        )

        answer = agent.run(
            "问题", AgentOptions(on_step=lambda s, n: steps.append(s))
        )

        self.assertEqual(answer, "done")
        self.assertIn('未知工具 "nope"', steps[0].results["c1"])
        self.assertIn("不是合法 JSON", steps[0].results["c2"])
        # 同一轮里每个失败的调用都各自得到观察值，互不影响
        self.assertEqual(len(steps[0].results), 2)


if __name__ == "__main__":
    unittest.main()
