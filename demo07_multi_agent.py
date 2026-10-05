import os

from typing import TypedDict, Literal

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langgraph.graph import (
    StateGraph,
    START,
    END,
)


# ============================================================
# 1. Load Environment
# ============================================================

load_dotenv()


# ============================================================
# 2. LLM
# ============================================================

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
    use_responses_api=False,
)


# ============================================================
# 3. State
# ============================================================

class State(TypedDict):

    user_request: str

    research: str

    draft: str

    final_answer: str

    next_agent: str


# ============================================================
# 4. Supervisor
# ============================================================

def supervisor(state: State):

    print("\n")
    print("=" * 60)
    print("SUPERVISOR")
    print("=" * 60)

    user_request = state["user_request"]

    prompt = f"""
你是一个企业级 Multi-Agent 系统的 Supervisor。

用户任务：

{user_request}

你的职责是判断当前应该由哪个 Agent 工作。

可选 Agent：

1. research
   负责研究、分析、拆解问题。

2. writer
   根据 research 结果生成最终报告。

如果目前还没有 research：
返回 research

如果已经有 research，但是没有 draft：
返回 writer

如果已经完成 draft：
返回 finish

只允许返回：

research
writer
finish
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().lower()

    print("Supervisor Decision:", decision)

    if "research" in decision:
        return {
            "next_agent": "research"
        }

    if "writer" in decision:
        return {
            "next_agent": "writer"
        }

    return {
        "next_agent": "finish"
    }


# ============================================================
# 5. Research Agent
# ============================================================

def research_agent(state: State):

    print("\n")
    print("=" * 60)
    print("RESEARCH AGENT")
    print("=" * 60)

    prompt = f"""
你是 Research Agent。

请对下面的问题进行深入分析：

{state["user_request"]}

要求：

1. 提炼核心问题
2. 分析关键趋势
3. 分析技术发展
4. 分析商业模式
5. 分析未来机会
6. 给出结构化研究结论

暂时不需要写成正式报告。

请输出研究结果。
"""

    response = llm.invoke(prompt)

    print("\nResearch Result:\n")

    print(response.content)

    return {
        "research": response.content
    }


# ============================================================
# 6. Writer Agent
# ============================================================

def writer_agent(state: State):

    print("\n")
    print("=" * 60)
    print("WRITER AGENT")
    print("=" * 60)

    prompt = f"""
你是 Writer Agent。

用户原始任务：

{state["user_request"]}

Research Agent 的研究结果：

{state["research"]}

请根据研究结果生成一份高质量最终回答。

要求：

1. 结构清晰
2. 逻辑严谨
3. 避免重复
4. 提炼核心观点
5. 给出明确结论
6. 使用 Markdown
"""

    response = llm.invoke(prompt)

    print("\nDraft:\n")

    print(response.content)

    return {
        "draft": response.content,
        "final_answer": response.content
    }


# ============================================================
# 7. Router
# ============================================================

def router(
    state: State,
) -> Literal[
    "research",
    "writer",
    "finish",
]:

    return state["next_agent"]


# ============================================================
# 8. Build Graph
# ============================================================

builder = StateGraph(State)


builder.add_node(
    "supervisor",
    supervisor,
)

builder.add_node(
    "research",
    research_agent,
)

builder.add_node(
    "writer",
    writer_agent,
)


# ============================================================
# 9. Edges
# ============================================================

builder.add_edge(
    START,
    "supervisor",
)


builder.add_conditional_edges(
    "supervisor",
    router,
    {
        "research": "research",
        "writer": "writer",
        "finish": END,
    },
)


builder.add_edge(
    "research",
    "supervisor",
)


builder.add_edge(
    "writer",
    "supervisor",
)


# ============================================================
# 10. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 11. Run
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("LANGGRAPH DEMO07")
    print("MULTI-AGENT / SUPERVISOR")
    print("=" * 60)

    user_request = input(
        "\n请输入任务："
    )

    result = graph.invoke(
        {
            "user_request": user_request,
            "research": "",
            "draft": "",
            "final_answer": "",
            "next_agent": "",
        }
    )

    print("\n")
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(
        result["final_answer"]
    )