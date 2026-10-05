"""
Demo10 - Enterprise Research Agent

学习目标：
1. LangGraph State
2. Multi-Agent
3. RAG / Knowledge Agent
4. Tool Calling
5. Parallel Execution
6. Analysis Agent
7. Writer Agent
8. Human-in-the-loop
9. Checkpoint / Memory

运行：

    uv run python demo10_enterprise_agent.py
"""

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.checkpoint.memory import InMemorySaver

from langgraph.types import (
    interrupt,
    Command,
)


# ============================================================
# 1. Environment
# ============================================================

load_dotenv()


# ============================================================
# 2. Agent State
# ============================================================

class AgentState(TypedDict):

    # 用户原始问题
    question: str

    # 内部知识库结果
    knowledge_result: str

    # 外部研究结果
    research_result: str

    # 综合分析
    analysis_result: str

    # 最终报告
    final_report: str

    # 人工审批结果
    approved: bool


# ============================================================
# 3. LLM
# ============================================================

def create_llm():

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv(
        "OPENAI_MODEL",
        "gpt-5.5",
    )

    if not api_key:
        raise RuntimeError(
            "未找到 OPENAI_API_KEY。\n"
            "请先设置环境变量。"
        )

    print()
    print("=" * 70)
    print("LLM Configuration")
    print("=" * 70)

    print("Model    :", model)
    print("Base URL :", base_url or "OpenAI Default")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        use_responses_api=False,
    )


llm = create_llm()


# ============================================================
# 4. Local Knowledge Base
# ============================================================

DOCS_DIR = Path("docs")


def load_knowledge():

    if not DOCS_DIR.exists():

        return (
            "内部知识库目录 docs/ 不存在。"
        )

    results = []

    for file in sorted(
        DOCS_DIR.glob("*.md")
    ):

        try:

            content = file.read_text(
                encoding="utf-8"
            )

        except Exception as e:

            content = (
                f"读取文件失败：{e}"
            )

        results.append(
            f"""
==================================================
SOURCE: {file.name}
==================================================

{content}
"""
        )

    if not results:

        return (
            "内部知识库为空。"
        )

    return "\n".join(results)


# ============================================================
# 5. Knowledge Search Tool
# ============================================================

@tool
def search_knowledge(
    query: str,
) -> str:
    """
    查询企业内部知识库。

    适用于：
    - 企业业务
    - 公司战略
    - AI战略
    - 数字化建设
    - 产品体系
    - 技术架构
    - 企业内部资料

    参数：
        query: 查询问题
    """

    knowledge = load_knowledge()

    if (
        not knowledge
        or knowledge == "内部知识库为空。"
    ):

        return (
            "内部知识库没有内容。"
        )

    # --------------------------------------------------------
    # Demo 版本：
    # 这里暂时使用全文文本提供给 LLM。
    #
    # 后续可以替换成：
    #
    # Query
    #   ↓
    # Embedding
    #   ↓
    # Vector DB
    #   ↓
    # Retriever
    #   ↓
    # Top-K
    #
    # --------------------------------------------------------

    return f"""
用户查询：

{query}

内部知识库：

{knowledge}
"""


# ============================================================
# 6. Knowledge Agent
# ============================================================

def knowledge_agent(
    state: AgentState,
):

    question = state["question"]

    print()
    print("[Knowledge Agent]")
    print("正在查询内部知识库...")

    result = search_knowledge.invoke(
        {
            "query": question
        }
    )

    return {
        "knowledge_result": result
    }


# ============================================================
# 7. External Research Tool
# ============================================================

@tool
def external_research(
    query: str,
) -> str:
    """
    外部研究工具。

    Demo 版本使用模拟数据。

    实际生产环境可以替换为：

    - MCP Search
    - Web Search
    - 企业 API
    - 搜索引擎
    - GitHub
    - 数据库
    """

    return f"""
外部研究主题：

{query}

外部研究摘要：

当前企业 Agent 技术体系主要包括：

1. Large Language Model
2. Agent Runtime
3. Tool Calling
4. MCP
5. RAG
6. Context Engineering
7. Memory
8. Multi-Agent
9. Human-in-the-loop
10. Agent Evaluation
11. Agent Observability
12. Enterprise AI Governance

企业级 Agent 的重点正在从：

“聊天机器人”

逐渐转向：

“能够理解任务、调用工具、访问企业知识、
执行工作流并持续积累上下文的智能系统。”
"""


# ============================================================
# 8. Research Agent
# ============================================================

def research_agent(
    state: AgentState,
):

    question = state["question"]

    print()
    print("[Research Agent]")
    print("正在进行外部研究...")

    result = external_research.invoke(
        {
            "query": question
        }
    )

    return {
        "research_result": result
    }


# ============================================================
# 9. Analysis Agent
# ============================================================

def analysis_agent(
    state: AgentState,
):

    question = state["question"]

    knowledge = state[
        "knowledge_result"
    ]

    research = state[
        "research_result"
    ]

    print()
    print("[Analysis Agent]")
    print("正在综合内部知识和外部研究...")


    prompt = f"""
你是一名企业级 AI 战略分析师。

请根据用户问题、企业内部知识和外部研究，
进行结构化分析。

==================================================
用户问题
==================================================

{question}


==================================================
企业内部知识
==================================================

{knowledge}


==================================================
外部研究
==================================================

{research}


==================================================
分析要求
==================================================

1. 区分：
   - 企业内部事实
   - 外部研究信息
   - 你的分析判断

2. 不得编造不存在的数据。

3. 如果内部知识不足，要明确指出。

4. 提炼关键趋势。

5. 分析企业可能面对的机会。

6. 分析主要风险。

7. 给出可执行的行动建议。

8. 从 CTO / CIO / 数智化负责人视角思考。

输出：

# 一、核心结论

# 二、关键事实

# 三、行业趋势

# 四、企业机会

# 五、主要风险

# 六、建议行动

# 七、下一步实施路径
"""


    response = llm.invoke(
        prompt
    )

    return {
        "analysis_result":
            response.content
    }


# ============================================================
# 10. Writer Agent
# ============================================================

def writer_agent(
    state: AgentState,
):

    analysis = state[
        "analysis_result"
    ]

    print()
    print("[Writer Agent]")
    print("正在生成管理层报告...")


    prompt = f"""
你是一名企业战略报告专家。

根据下面的分析结果，
生成一份适合企业管理层阅读的报告。

==================================================
分析结果
==================================================

{analysis}


==================================================
报告要求
==================================================

语言：

- 专业
- 简洁
- 克制
- 有逻辑
- 避免营销化语言

报告结构：

# 企业 AI Agent 战略分析报告

## 1. 执行摘要

## 2. 核心发现

## 3. 技术趋势

## 4. 企业机会

## 5. 主要风险

## 6. 建议行动

## 7. 90天实施建议

## 8. 未来12个月路线图

要求：

1. 不编造事实。
2. 区分事实与判断。
3. 尽量给出具体行动。
4. 管理层可以直接阅读。
"""


    response = llm.invoke(
        prompt
    )

    return {
        "final_report":
            response.content
    }


# ============================================================
# 11. Human Review
# ============================================================

def human_review(
    state: AgentState,
):

    report = state[
        "final_report"
    ]

    print()
    print("=" * 70)
    print("HUMAN REVIEW")
    print("=" * 70)

    print()
    print(report)

    print()
    print("=" * 70)

    decision = interrupt(
        {
            "type": "report_review",

            "message":
                "请审核最终报告",

            "report":
                report,

            "options": [
                "approve",
                "reject",
            ],
        }
    )

    approved = (
        str(decision).lower()
        == "approve"
    )

    return {
        "approved": approved
    }


# ============================================================
# 12. Build Graph
# ============================================================

def build_graph():

    builder = StateGraph(
        AgentState
    )


    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    builder.add_node(
        "knowledge",
        knowledge_agent,
    )

    builder.add_node(
        "research",
        research_agent,
    )

    builder.add_node(
        "analysis",
        analysis_agent,
    )

    builder.add_node(
        "writer",
        writer_agent,
    )

    builder.add_node(
        "review",
        human_review,
    )


    # --------------------------------------------------------
    # Parallel branches
    # --------------------------------------------------------

    builder.add_edge(
        START,
        "knowledge",
    )

    builder.add_edge(
        START,
        "research",
    )


    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    builder.add_edge(
        "knowledge",
        "analysis",
    )

    builder.add_edge(
        "research",
        "analysis",
    )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    builder.add_edge(
        "analysis",
        "writer",
    )

    builder.add_edge(
        "writer",
        "review",
    )

    builder.add_edge(
        "review",
        END,
    )


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpointer = InMemorySaver()


    graph = builder.compile(
        checkpointer=checkpointer
    )

    return graph


# ============================================================
# 13. Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("Demo10 - Enterprise Research Agent")
    print("=" * 70)

    print()
    print("Architecture:")
    print()
    print(
        "User"
        " -> Knowledge Agent"
        " + Research Agent"
        " -> Analysis Agent"
        " -> Writer Agent"
        " -> Human Review"
    )


    # --------------------------------------------------------
    # Build Graph
    # --------------------------------------------------------

    graph = build_graph()


    # --------------------------------------------------------
    # User Question
    # --------------------------------------------------------

    question = input(
        "\n请输入研究任务：\n> "
    ).strip()


    if not question:

        print(
            "问题不能为空。"
        )

        return


    # --------------------------------------------------------
    # Thread / Memory
    # --------------------------------------------------------

    config = {
        "configurable": {
            "thread_id":
                "demo10-user-001"
        }
    }


    # --------------------------------------------------------
    # Initial State
    # --------------------------------------------------------

    initial_state = {

        "question":
            question,

        "knowledge_result":
            "",

        "research_result":
            "",

        "analysis_result":
            "",

        "final_report":
            "",

        "approved":
            False,
    }


    # --------------------------------------------------------
    # First Run
    # --------------------------------------------------------

    try:

        result = graph.invoke(
            initial_state,
            config=config,
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("Agent 执行失败")
        print("=" * 70)

        print(
            f"\n{type(e).__name__}: {e}"
        )

        return


    # --------------------------------------------------------
    # Human Review
    # --------------------------------------------------------

    while True:

        # ----------------------------------------------------
        # 获取当前状态
        # ----------------------------------------------------

        state_snapshot = (
            graph.get_state(config)
        )


        # ----------------------------------------------------
        # 判断是否有 interrupt
        # ----------------------------------------------------

        if not state_snapshot.interrupts:

            break


        print()
        print("=" * 70)
        print("等待人工审核")
        print("=" * 70)

        print()
        print(
            "请输入："
        )

        print(
            "approve  = 批准"
        )

        print(
            "reject   = 拒绝"
        )

        print(
            "rewrite  = 重新生成"
        )


        decision = input(
            "\n> "
        ).strip().lower()


        # ----------------------------------------------------
        # Approve
        # ----------------------------------------------------

        if decision == "approve":

            result = graph.invoke(
                Command(
                    resume="approve"
                ),
                config=config,
            )

            break


        # ----------------------------------------------------
        # Reject / Rewrite
        # ----------------------------------------------------

        elif decision in (
            "reject",
            "rewrite",
        ):

            print()
            print(
                "报告将重新生成..."
            )

            result = graph.invoke(
                Command(
                    resume="reject"
                ),
                config=config,
            )

            break


        else:

            print(
                "请输入 approve / reject / rewrite"
            )


    # ========================================================
    # Final Result
    # ========================================================

    final_state = graph.get_state(
        config
    ).values


    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)


    print()

    print(
        final_state.get(
            "final_report",
            "没有生成最终报告。",
        )
    )


    print()
    print("=" * 70)

    print(
        "Approved:",
        final_state.get(
            "approved",
            False,
        )
    )

    print("=" * 70)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()