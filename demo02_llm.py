import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import Annotated
from typing_extensions import TypedDict


# ============================================================
# 1. 加载环境变量
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY 未设置")

if not BASE_URL:
    raise RuntimeError("OPENAI_BASE_URL 未设置")

if not MODEL:
    raise RuntimeError("OPENAI_MODEL 未设置")


# ============================================================
# 2. 创建 LLM
# ============================================================

llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ============================================================
# 3. 定义 LangGraph State
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 4. 定义 LLM Node
# ============================================================

def chatbot(state: State):

    response = llm.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 5. 创建 Graph
# ============================================================

builder = StateGraph(State)

builder.add_node(
    "chatbot",
    chatbot
)

builder.add_edge(
    START,
    "chatbot"
)

builder.add_edge(
    "chatbot",
    END
)


# ============================================================
# 6. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 7. Invoke
# ============================================================

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="请用三句话解释什么是 Agent。"
            )
        ]
    }
)


# ============================================================
# 8. 输出
# ============================================================

print("\n========== Agent Response ==========\n")

print(
    result["messages"][-1].content
)

print("\n====================================\n")
