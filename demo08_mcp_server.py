from datetime import datetime
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("demo08-tools")


@mcp.tool()
def get_current_time() -> str:
    """
    获取当前本地时间。
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


@mcp.tool()
def calculate(
    expression: str,
) -> str:
    """
    计算简单数学表达式。

    例如：
    100 * 0.15
    1000 / 8
    20 + 30
    """

    try:

        # Demo 学习用。
        # 生产环境不要直接 eval 用户输入。
        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {},
        )

        return str(result)

    except Exception as e:

        return f"计算失败：{e}"


if __name__ == "__main__":

    mcp.run(
        transport="stdio"
    )