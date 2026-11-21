"""与 LLM 服务通信：环境变量配置 + 一次 Chat Completions 请求。

不依赖任何 SDK，直接用标准库 urllib，可对接 OpenAI / DeepSeek / GLM / Ollama 等服务。
"""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .types import ChatMessage, LLMConfig, ToolSchema

TIMEOUT_SECONDS = 120


def load_config() -> LLMConfig:
    """从环境变量读取配置，缺失时给出可操作的提示"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少环境变量 OPENAI_API_KEY。\n"
            "  必填: OPENAI_API_KEY   API 密钥（本地 Ollama 可填任意非空值，如 ollama）\n"
            "  可选: OPENAI_BASE_URL  默认为 https://api.openai.com/v1\n"
            "        MODEL            默认为 gpt-4o-mini\n"
            "各平台（bash / PowerShell / cmd）的设置命令见 README「快速开始」。"
        )
    return LLMConfig(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        api_key=api_key,
        model=os.environ.get("MODEL", "gpt-4o-mini"),
    )


def chat(
    messages: list[ChatMessage],
    tools: list[ToolSchema],
    config: LLMConfig,
) -> ChatMessage:
    """调用一次 Chat Completions，返回 assistant 消息（可能含 tool_calls）。

    注意：工具调用的约定不在 system prompt 里，而在请求体的 tools 字段。
    tools 是 API 的一等参数，服务端会把这些 JSON Schema 按模型内部模板渲染，
    模型经过 function calling 微调后，才知道"需要外部信息时输出 tool_calls 而非自然语言"。
    所以：tools 决定"能不能调、叫什么名、参数长什么样"，
          system prompt 只影响"要不要调、什么时候调"。
    把 tools 传成空列表，模型基本就不会再返回 tool_calls。
    """
    body = json.dumps(
        {"model": config["model"], "messages": messages, "tools": tools}
    ).encode("utf-8")
    req = Request(
        f"{config['base_url']}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=TIMEOUT_SECONDS) as res:
            data = json.loads(res.read().decode("utf-8"))
    except HTTPError as e:
        # 服务端有响应但拒绝了：key 无效、模型名写错、余额不足等，把响应体带出来
        raise RuntimeError(f"LLM 请求失败 {e.code}: {e.read().decode('utf-8', 'replace')}") from e
    except URLError as e:
        # 连不上服务：Ollama 没启动、BASE_URL 写错、断网等。
        # 注意 HTTPError 是 URLError 的子类，必须先捕具体的 HTTPError，
        # 否则上面带响应体的报错会被这个分支吞掉。
        raise RuntimeError(
            f"无法连接 {config['base_url']}，请检查服务是否已启动、地址是否正确（{e.reason}）"
        ) from e

    # 教学简化：直接断言返回结构。生产代码应做运行时校验（如 pydantic），
    # 否则服务返回异常结构时只会在更远处报错，难以定位。
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"LLM 返回结构异常: {json.dumps(data, ensure_ascii=False)}") from e
    return message
