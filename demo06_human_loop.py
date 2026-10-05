from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# ============================================================
# 1. State
# ============================================================

class State(TypedDict):
    applicant: str
    amount: float
    purpose: str
    approved: bool
    result: str


# ============================================================
# 2. 创建报销申请
# ============================================================

def create_expense(state: State):

    print("\n================================")
    print("创建报销申请")
    print("================================")

    print(f"申请人：{state['applicant']}")
    print(f"金额：{state['amount']} 元")
    print(f"用途：{state['purpose']}")

    return {
        "result": "等待人工审批"
    }


# ============================================================
# 3. Human-in-the-Loop
# ============================================================

def human_approval(state: State):

    print("\n================================")
    print("等待人工审批")
    print("================================")

    decision = interrupt({
        "type": "expense_approval",
        "applicant": state["applicant"],
        "amount": state["amount"],
        "purpose": state["purpose"],
        "message": "请审批这笔报销：批准 / 拒绝",
    })

    print(f"\n人工审批结果：{decision}")

    if decision == "批准":
        return {
            "approved": True,
            "result": "审批通过"
        }

    return {
        "approved": False,
        "result": "审批拒绝"
    }


# ============================================================
# 4. 执行报销
# ============================================================

def execute_expense(state: State):

    print("\n================================")
    print("执行报销")
    print("================================")

    if state["approved"]:

        print(
            f"正在为 {state['applicant']} "
            f"执行 {state['amount']} 元报销..."
        )

        return {
            "result": "报销执行成功"
        }

    print("报销未执行。")

    return {
        "result": "报销未执行"
    }


# ============================================================
# 5. 构建 Graph
# ============================================================

builder = StateGraph(State)

builder.add_node(
    "create_expense",
    create_expense
)

builder.add_node(
    "human_approval",
    human_approval
)

builder.add_node(
    "execute_expense",
    execute_expense
)


# ============================================================
# 6. Graph Routing
# ============================================================

builder.add_edge(
    START,
    "create_expense"
)

builder.add_edge(
    "create_expense",
    "human_approval"
)

builder.add_edge(
    "human_approval",
    "execute_expense"
)

builder.add_edge(
    "execute_expense",
    END
)


# ============================================================
# 7. Checkpointer
# ============================================================

memory = InMemorySaver()


graph = builder.compile(
    checkpointer=memory
)


# ============================================================
# 8. Thread
# ============================================================

config = {
    "configurable": {
        "thread_id": "expense-jerry-001"
    }
}


# ============================================================
# 9. 第一次执行
# ============================================================

print("\n================================")
print("第一次运行 Agent")
print("================================")

result = graph.invoke(
    {
        "applicant": "Jerry",
        "amount": 5000,
        "purpose": "上海出差",
        "approved": False,
        "result": "",
    },
    config
)


print("\nAgent 当前状态：")

print(
    result
)


# ============================================================
# 10. 获取当前状态
# ============================================================

snapshot = graph.get_state(config)

print("\n================================")
print("当前 Graph State")
print("================================")

print(snapshot.values)


# ============================================================
# 11. 人工恢复 Agent
# ============================================================

print("\n================================")
print("人工输入审批结果")
print("================================")

decision = input(
    "请输入：批准 / 拒绝："
).strip()


result = graph.invoke(
    Command(
        resume=decision
    ),
    config
)


# ============================================================
# 12. 最终结果
# ============================================================

print("\n================================")
print("最终结果")
print("================================")

print(result)