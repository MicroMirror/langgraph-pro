import os
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.1,
    use_responses_api=False,
)

# ============================================================
# 2. Chatbot Node
# ============================================================

def chatbot(state: MessagesState):

    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


# ============================================================
# 3. 创建 Graph
# ============================================================

builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)


# ============================================================
# 4. 创建 Checkpointer
# ============================================================

memory = InMemorySaver()


# ============================================================
# 5. 编译 Graph
# ============================================================

graph = builder.compile(
    checkpointer=memory
)


# ============================================================
# 6. Thread 1
# ============================================================

config_1 = {
    "configurable": {
        "thread_id": "user-jerry"
    }
}


print("\n==============================")
print("第一次对话")
print("==============================")

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="你好，我叫 Jerry，是一名 CTO。"
            )
        ]
    },
    config_1
)

print(
    "AI:",
    result["messages"][-1].content
)


# ============================================================
# 7. 第二次调用
# ============================================================

print("\n==============================")
print("第二次对话")
print("==============================")

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="你还记得我的名字和职业吗？"
            )
        ]
    },
    config_1
)

print(
    "AI:",
    result["messages"][-1].content
)


# ============================================================
# 8. Thread 2
# ============================================================

config_2 = {
    "configurable": {
        "thread_id": "user-tom"
    }
}


print("\n==============================")
print("切换到另一个用户")
print("==============================")

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="你知道我叫什么吗？"
            )
        ]
    },
    config_2
)

print(
    "AI:",
    result["messages"][-1].content
)
