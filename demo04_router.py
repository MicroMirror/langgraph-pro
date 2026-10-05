from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. State
# ============================================================

class State(TypedDict):
    user_input: str
    route: str
    result: str


# ============================================================
# 2. Router
# ============================================================

def router(state: State) -> dict:
    """
    根据用户输入决定下一步走哪个节点。
    """

    text = state["user_input"].lower()

    if any(keyword in text for keyword in ["天气", "气温", "下雨", "weather"]):
        route = "weather"

    elif any(keyword in text for keyword in ["时间", "几点", "time"]):
        route = "time"

    elif any(keyword in text for keyword in ["计算", "多少", "+", "-", "*", "/", "calculate"]):
        route = "calculator"

    else:
        route = "general"

    print(f"[Router] route = {route}")

    return {
        "route": route
    }


# ============================================================
# 3. Weather Node
# ============================================================

def weather_node(state: State) -> dict:
    print("[Weather Node]")

    return {
        "result": "上海：晴，28°C，湿度 65%"
    }


# ============================================================
# 4. Time Node
# ============================================================

def time_node(state: State) -> dict:
    print("[Time Node]")

    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Asia/Shanghai"))

    return {
        "result": now.strftime("%Y-%m-%d %H:%M:%S")
    }


# ============================================================
# 5. Calculator Node
# ============================================================

def calculator_node(state: State) -> dict:
    print("[Calculator Node]")

    expression = state["user_input"]

    # Demo only
    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )
    except Exception as e:
        result = f"计算失败：{e}"

    return {
        "result": str(result)
    }


# ============================================================
# 6. General Node
# ============================================================

def general_node(state: State) -> dict:
    print("[General Node]")

    return {
        "result": f"这是一个普通问题：{state['user_input']}"
    }


# ============================================================
# 7. Conditional Routing Function
# ============================================================

def route_decision(state: State) -> Literal[
    "weather",
    "time",
    "calculator",
    "general",
]:
    """
    Conditional Edge 根据 route 决定下一步。
    """

    return state["route"]


# ============================================================
# 8. Build Graph
# ============================================================

builder = StateGraph(State)


# Nodes

builder.add_node("router", router)
builder.add_node("weather", weather_node)
builder.add_node("time", time_node)
builder.add_node("calculator", calculator_node)
builder.add_node("general", general_node)


# START → Router

builder.add_edge(START, "router")


# Router → Conditional Routing

builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "weather": "weather",
        "time": "time",
        "calculator": "calculator",
        "general": "general",
    },
)


# 各分支 → END

builder.add_edge("weather", END)
builder.add_edge("time", END)
builder.add_edge("calculator", END)
builder.add_edge("general", END)


# Compile

graph = builder.compile()


# ============================================================
# 9. Test
# ============================================================

if __name__ == "__main__":

    test_cases = [
        "上海天气怎么样？",
        "现在几点？",
        "128 * 256",
        "什么是 Agent？",
    ]

    for question in test_cases:

        print("\n" + "=" * 60)
        print(f"User: {question}")

        result = graph.invoke({
            "user_input": question,
            "route": "",
            "result": "",
        })

        print(f"Result: {result['result']}")

