"""
Demo11 - Jev + LangGraph Agent

核心架构：

User
  ↓
LangGraph
  ↓
Jev Decision Layer
  ↓
Conditional Router
  ├── technical
  ├── business
  ├── urgent
  └── general
  ↓
LLM Agent
  ↓
Final Answer

学习重点：

1. .env 环境变量
2. Jev Choice
3. Jev Noul
4. Jev confidence
5. LangGraph State
6. Conditional Routing
7. Jev + LLM 协同
"""

import os
from typing import TypedDict

from dotenv import load_dotenv

from typesafe_sdk import Choice, Noul, TypeSafeClient

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. Load .env
# ============================================================

load_dotenv()


# ============================================================
# 2. Environment Check
# ============================================================

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")


if not TYPESAFE_API_KEY:
    raise RuntimeError(
        "未找到 TYPESAFE_API_KEY，请检查 .env"
    )

if not OPENAI_API_KEY:
    raise RuntimeError(
        "未找到 OPENAI_API_KEY，请检查 .env"
    )


# ============================================================
# 3. Agent State
# ============================================================

class AgentState(TypedDict):

    # 用户问题
    question: str

    # Jev 决策
    route: str

    # Jev Choice confidence
    route_confidence: float

    # Jev Noul probability
    urgent_probability: float

    # 最终答案
    answer: str


# ============================================================
# 4. LLM
# ============================================================

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0,
    use_responses_api=False,
)


# ============================================================
# 5. Jev Decision
# ============================================================

def call_jev(question: str):
    """
    调用 Jev。

    一次 System One 请求同时提出：

    1. Choice
       判断应该进入哪个 Agent

    2. Noul
       判断是否紧急

    """

    with TypeSafeClient() as client:

        response = client.system_one(

            state={
                "user_question": question
            },

            questions={

                # ------------------------------------------------
                # Choice
                # ------------------------------------------------

                "route": Choice(

                    instructions="""
判断这个用户问题最适合由哪个 Agent 处理。

必须从以下四个选项中选择一个：

technical：
技术、架构、代码、AI、模型、云计算、
基础设施、系统设计等问题。

business：
企业经营、数字化转型、管理、组织、
业务流程、商业模式等问题。

urgent：
生产故障、重大系统异常、安全事件、
业务中断、需要立即处理的问题。

general：
不属于以上三类的普通问题。
""",

                    criteria={

                        "technical":
                            "技术、架构、代码、AI、模型、云计算、系统、基础设施",

                        "business":
                            "企业业务、经营、管理、数字化转型、组织、商业模式",

                        "urgent":
                            "生产故障、重大异常、安全事件、业务中断、紧急事件",

                        "general":
                            "其他普通问题",
                    },
                ),

                # ------------------------------------------------
                # Noul
                # ------------------------------------------------

                "urgent": Noul(

                    instructions="""
判断用户问题是否明确涉及需要立即处理的紧急事件。

如果涉及生产事故、重大系统故障、安全事件、
业务中断或明确的紧急处理要求，则为 true。

普通技术咨询、架构设计、业务咨询不属于紧急事件。
"""
                ),
            },
        )

    return response


# ============================================================
# 6. Jev Node
# ============================================================

def jev_node(state: AgentState):

    question = state["question"]

    print()
    print("=" * 70)
    print("JEV DECISION")
    print("=" * 70)

    print()
    print("User:")
    print(question)

    # --------------------------------------------------------
    # 调用 Jev
    # --------------------------------------------------------

    response = call_jev(question)

    # --------------------------------------------------------
    # Choice
    # --------------------------------------------------------

    route_result = response.choices["route"]

    route = route_result.choice

    route_confidence = route_result.confidence

    # --------------------------------------------------------
    # Noul
    #
    # 注意：
    # Noul 没有 confidence。
    # 它返回 0～1 的 probability。
    # --------------------------------------------------------

    urgent_result = response.nouls["urgent"]

    urgent_probability = urgent_result.noul

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    print()
    print("Jev Route:")
    print(route)

    print()
    print("Route Confidence:")
    print(route_confidence)

    print()
    print("Urgent Probability:")
    print(urgent_probability)

    print("=" * 70)

    return {
        "route": route,
        "route_confidence": route_confidence,
        "urgent_probability": urgent_probability,
    }


# ============================================================
# 7. Technical Agent
# ============================================================

def technical_agent(state: AgentState):

    question = state["question"]

    print()
    print("[Technical Agent]")

    prompt = f"""
你是一名资深企业技术架构师。

用户问题：

{question}

请从以下角度回答：

1. 问题理解
2. 技术原理
3. 架构设计
4. 可选技术方案
5. 优缺点
6. 实施建议

要求：

- 技术上严谨
- 不要编造事实
- 如果信息不足，明确指出
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# 8. Business Agent
# ============================================================

def business_agent(state: AgentState):

    question = state["question"]

    print()
    print("[Business Agent]")

    prompt = f"""
你是一名企业数字化与AI战略顾问。

用户问题：

{question}

请从以下角度分析：

1. 业务目标
2. 核心问题
3. 业务价值
4. 组织影响
5. 实施路径
6. 风险
7. 衡量指标

避免空泛表达。
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# 9. Urgent Agent
# ============================================================

def urgent_agent(state: AgentState):

    question = state["question"]

    print()
    print("[Urgent Incident Agent]")

    prompt = f"""
你是一名企业生产事故响应专家。

当前事件：

{question}

请按照事故响应优先级回答：

1. 立即确认什么
2. 首先采取什么措施
3. 如何控制影响范围
4. 如何排查根因
5. 如何进行业务沟通
6. 什么情况下升级处理

不要做没有依据的推测。
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# 10. General Agent
# ============================================================

def general_agent(state: AgentState):

    question = state["question"]

    print()
    print("[General Agent]")

    response = llm.invoke(question)

    return {
        "answer": response.content
    }


# ============================================================
# 11. Human Review
# ============================================================

def human_review(state: AgentState):

    print()
    print("[Human Review]")

    print()
    print("Jev 判断置信度较低。")

    print(
        "Route:",
        state["route"]
    )

    print(
        "Confidence:",
        state["route_confidence"]
    )

    print(
        "Question:",
        state["question"]
    )

    return {
        "answer": (
            "Jev 对该问题的分类置信度较低，"
            "当前进入人工审核流程。"
        )
    }


# ============================================================
# 12. Router
# ============================================================

def route_after_jev(state: AgentState):

    route = state["route"]

    confidence = state["route_confidence"]

    urgent_probability = state[
        "urgent_probability"
    ]

    print()
    print("=" * 70)
    print("LANGGRAPH ROUTER")
    print("=" * 70)

    print("route =", route)

    print(
        "confidence =",
        confidence
    )

    print(
        "urgent_probability =",
        urgent_probability
    )

    # --------------------------------------------------------
    # 第一优先级：
    # 紧急事件
    # --------------------------------------------------------

    if urgent_probability >= 0.90:

        return "urgent"

    # --------------------------------------------------------
    # 第二优先级：
    # Jev 对 Choice 判断不够确定
    #
    # 0.70 只是 Demo 阈值。
    # 生产环境应该通过自己的 eval 数据确定。
    # --------------------------------------------------------

    if confidence < 0.70:

        return "human"

    # --------------------------------------------------------
    # 正常路由
    # --------------------------------------------------------

    if route == "technical":

        return "technical"

    if route == "business":

        return "business"

    if route == "urgent":

        return "urgent"

    return "general"


# ============================================================
# 13. Build LangGraph
# ============================================================

def build_graph():

    builder = StateGraph(
        AgentState
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    builder.add_node(
        "jev",
        jev_node
    )

    builder.add_node(
        "technical",
        technical_agent
    )

    builder.add_node(
        "business",
        business_agent
    )

    builder.add_node(
        "urgent",
        urgent_agent
    )

    builder.add_node(
        "general",
        general_agent
    )

    builder.add_node(
        "human",
        human_review
    )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    builder.add_edge(
        START,
        "jev"
    )

    # --------------------------------------------------------
    # Conditional Routing
    # --------------------------------------------------------

    builder.add_conditional_edges(

        "jev",

        route_after_jev,

        {
            "technical": "technical",

            "business": "business",

            "urgent": "urgent",

            "general": "general",

            "human": "human",
        }
    )

    # --------------------------------------------------------
    # END
    # --------------------------------------------------------

    builder.add_edge(
        "technical",
        END
    )

    builder.add_edge(
        "business",
        END
    )

    builder.add_edge(
        "urgent",
        END
    )

    builder.add_edge(
        "general",
        END
    )

    builder.add_edge(
        "human",
        END
    )

    return builder.compile()


# ============================================================
# 14. Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("Demo11 - Jev + LangGraph Agent")
    print("=" * 70)

    print()
    print("示例：")
    print()
    print("1. 如何设计企业AI Agent平台？")
    print("2. 企业AI数字化转型应该怎么做？")
    print("3. 生产环境突然出现大量500错误怎么办？")
    print()

    question = input("> ").strip()

    if not question:

        print("问题不能为空。")

        return

    # --------------------------------------------------------
    # Initial State
    # --------------------------------------------------------

    initial_state: AgentState = {

        "question": question,

        "route": "",

        "route_confidence": 0.0,

        "urgent_probability": 0.0,

        "answer": "",
    }

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    graph = build_graph()

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    result = graph.invoke(
        initial_state
    )

    # --------------------------------------------------------
    # Final Result
    # --------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print()
    print(result["answer"])

    print()
    print("=" * 70)
    print("JEV RESULT")
    print("=" * 70)

    print(
        "Route:",
        result["route"]
    )

    print(
        "Route Confidence:",
        result["route_confidence"]
    )

    print(
        "Urgent Probability:",
        result["urgent_probability"]
    )

    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()