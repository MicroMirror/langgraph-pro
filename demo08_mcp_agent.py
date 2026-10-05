import os
import asyncio

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import create_react_agent


load_dotenv()


# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model=os.getenv(
        "OPENAI_MODEL",
        "gpt-5.5",
    ),
    api_key=os.getenv(
        "OPENAI_API_KEY"
    ),
    base_url=os.getenv(
        "OPENAI_BASE_URL"
    ),
    temperature=0,
    use_responses_api=False,
)


# ============================================================
# 2. MCP Client
# ============================================================

client = MultiServerMCPClient(
    {
        "demo08": {
            "command": "uv",
            "args": [
                "run",
                "python",
                "demo08_mcp_server.py",
            ],
            "transport": "stdio",
        }
    }
)


# ============================================================
# 3. Main
# ============================================================

async def main():

    print("=" * 60)
    print("Demo08 - MCP + LangGraph")
    print("=" * 60)


    # --------------------------------------------------------
    # 获取 MCP Tools
    # --------------------------------------------------------

    tools = await client.get_tools()


    print("\n发现 MCP Tools：")

    for tool in tools:

        print(
            f"- {tool.name}: "
            f"{tool.description}"
        )


    # --------------------------------------------------------
    # 创建 Agent
    # --------------------------------------------------------

    agent = create_react_agent(
        model=llm,
        tools=tools,
    )


    # --------------------------------------------------------
    # 用户问题
    # --------------------------------------------------------

    user_input = input(
        "\n请输入问题："
    )


    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        }
    )


    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("Agent Final Answer")
    print("=" * 60)

    print(
        result["messages"][-1].content
    )


if __name__ == "__main__":

    asyncio.run(main())