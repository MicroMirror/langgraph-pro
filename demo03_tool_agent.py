from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# 1. 创建 Tools
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Args:
        expression: Mathematical expression, e.g. "128 * 256"
    """
    try:
        # Demo only.
        # Production environment should NOT directly use eval().
        result = eval(expression, {"__builtins__": {}})

        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.

    Args:
        city: City name
    """

    weather = {
        "上海": "26°C，晴",
        "北京": "20°C，多云",
        "深圳": "29°C，雷阵雨",
        "广州": "30°C，晴",
        "杭州": "25°C，小雨",
    }

    return weather.get(
        city,
        f"{city}：暂无天气数据",
    )

@tool
def get_local_time(timezone: str = "Asia/Shanghai") -> str:
    """
    获取指定时区的当前本地时间。

    Args:
        timezone:
            IANA 时区名称，例如：
            Asia/Shanghai
            Asia/Singapore
            Asia/Tokyo
            America/New_York
            Europe/London
    """

    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)

        return (
            f"{now.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{now.tzname()} "
            f"(UTC{now.strftime('%z')})"
        )

    except Exception:
        return (
            f"不支持的时区：{timezone}。"
            "请使用 IANA 时区，例如 Asia/Shanghai。"
        )

tools = [
    calculator,
    get_weather,
	get_local_time,
]


# ============================================================
# 2. 创建 LLM
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)


# ============================================================
# 3. 绑定 Tools
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 4. 定义 State
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 5. Agent Node
# ============================================================

def chatbot(state: State):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 6. 判断是否需要调用 Tool
# ============================================================

def should_continue(state: State):

    last_message = state["messages"][-1]

    # 如果 LLM 请求调用工具
    if last_message.tool_calls:
        return "tools"

    # 否则结束
    return END


# ============================================================
# 7. 创建 Graph
# ============================================================

builder = StateGraph(State)


# Agent
builder.add_node(
    "chatbot",
    chatbot,
)


# Tools
builder.add_node(
    "tools",
    ToolNode(tools),
)


# START → chatbot
builder.add_edge(
    START,
    "chatbot",
)


# chatbot → tools / END
builder.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)


# tools → chatbot
builder.add_edge(
    "tools",
    "chatbot",
)


# ============================================================
# 8. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 9. 测试
# ============================================================

questions = [
    "请计算 128 * 256",
    "上海今天天气怎么样？",
    "请解释一下什么是 Agent，不需要调用工具。",
	"当前具体什么时间？"
]


for question in questions:

    print("\n" + "=" * 60)

    print("USER:")
    print(question)

    print("-" * 60)

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=question
                )
            ]
        }
    )

    print("ASSISTANT:")

    print(
        result["messages"][-1].content
    )
