from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    message: str


def hello(state: State):
    return {
        "message": state["message"] + " Hello LangGraph!"
    }


# 创建 Graph
builder = StateGraph(State)

# 添加 Node
builder.add_node("hello", hello)

# 定义 Edge
builder.add_edge(START, "hello")
builder.add_edge("hello", END)

# 编译 Graph
graph = builder.compile()

# 执行
result = graph.invoke({
    "message": "Hello"
})

print(result)
