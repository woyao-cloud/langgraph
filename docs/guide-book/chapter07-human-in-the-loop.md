# 第七章：人机交互

## 为什么需要人机交互

大语言模型虽然强大，但并非万能。在实际业务中，许多场景需要人类介入：

- **LLM 的局限性**：模型可能产生幻觉、做出错误决策或遗漏关键信息，需要人类审核、确认和补充。
- **合规需求**：在金融、医疗等领域，关键决策必须经过人工确认后才能执行。例如，自动审批系统在拒绝贷款前需要人工复核。
- **渐进式交互**：用户逐步提供信息，而非一次性提供所有输入。例如多步骤表单，每一步都暂停等待用户补充。

LangGraph 提供了完整的人机交互机制，核心包括 `interrupt()` 函数和 `Command` 类。

## interrupt() 函数详解

### 基本用法

`interrupt(value)` 在节点中暂停图执行，将 `value` 传递给客户端，等待人类响应后恢复执行：

```python
import uuid
from typing import Optional
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command


class State(TypedDict):
    question: str
    answer: Optional[str]


def ask_human(state: State):
    # 暂停执行，将问题发送给客户端
    # 首次调用时抛出 GraphInterrupt 异常
    # 恢复后返回人类提供的值
    human_answer = interrupt("请回答这个问题：" + state["question"])
    return {"answer": human_answer}


builder = StateGraph(State)
builder.add_node("ask_human", ask_human)
builder.add_edge(START, "ask_human")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": uuid.uuid4()}}

# 第一次调用：触发中断
for chunk in graph.stream({"question": "1+1等于几？"}, config):
    print(chunk)
# 输出: {'__interrupt__': (Interrupt(value='请回答这个问题：1+1等于几？', id='...'),)}

# 人类提供答案后，恢复执行
for chunk in graph.stream(Command(resume="2"), config):
    print(chunk)
# 输出: {'ask_human': {'answer': '2'}}
```

> **重要**：使用 `interrupt()` 必须启用 checkpointer，因为中断机制依赖持久化状态。

### 多次中断

同一节点中可以多次调用 `interrupt()`。LangGraph 按调用顺序匹配恢复值：

```python
class FormState(TypedDict):
    name: Optional[str]
    age: Optional[int]


def collect_form(state: FormState):
    # 第一次中断：收集姓名
    name = interrupt("请输入您的姓名")
    # 第二次中断：收集年龄
    age = interrupt("请输入您的年龄")
    return {"name": name, "age": int(age)}


builder = StateGraph(FormState)
builder.add_node("collect_form", collect_form)
builder.add_edge(START, "collect_form")

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "form-1"}}

# 首次调用：在第一个 interrupt 处暂停
graph.stream({"name": None, "age": None}, config)

# 恢复第一个中断，然后在第二个 interrupt 处暂停
graph.stream(Command(resume="张三"), config)

# 恢复第二个中断，节点执行完成
for chunk in graph.stream(Command(resume="25"), config):
    print(chunk)
# 输出: {'collect_form': {'name': '张三', 'age': 25}}
```

### 中断值的类型

`interrupt()` 的 `value` 参数支持任意 Python 对象，不限于字符串：

```python
def review_node(state):
    # 发送结构化信息供人类审核
    review_info = interrupt({
        "type": "approval_required",
        "document": state["document"],
        "risk_level": state["risk_level"],
        "suggested_action": state["suggested_action"],
    })
    return {"approved": review_info["approved"], "comment": review_info["comment"]}
```

## Command 类详解

### Command(resume=value)：恢复执行

最基本的用法是恢复被 `interrupt()` 暂停的执行：

```python
# 恢复单个中断
graph.stream(Command(resume="用户的回答"), config)

# 恢复多个中断（按顺序匹配）
graph.stream(Command(resume=["第一个回答", "第二个回答"]), config)

# 按 ID 恢复特定中断
graph.stream(
    Command(resume={"interrupt_id": "特定的恢复值"}),
    config,
)
```

### Command(update=..., goto=...)：同时更新状态并路由

`Command` 不仅能恢复中断，还能同时更新图的状态并指定下一个执行的节点：

```python
from langgraph.types import Command
from typing import Literal


class State(TypedDict):
    messages: list
    next_action: str


def router(state: State) -> Command[Literal["agent", "__end__"]]:
    if state["next_action"] == "continue":
        # 同时更新状态并路由到 agent 节点
        return Command(update={"messages": ["继续对话"]}, goto="agent")
    else:
        return Command(goto=END)


builder = StateGraph(State)
builder.add_node("router", router)
builder.add_node("agent", lambda s: s)
builder.add_edge(START, "router")
builder.add_edge("agent", END)

graph = builder.compile()
```

### Command(graph=Command.PARENT)：子图向父图发送命令

在子图中，可以使用 `Command(graph=Command.PARENT)` 向父图发送路由指令：

```python
from langgraph.types import Command


class ParentState(TypedDict):
    value: str


class ChildState(TypedDict):
    data: str


def child_node(state: ChildState):
    # 子图节点决定跳转到父图的特定节点
    return Command(graph=Command.PARENT, goto="parent_target")


child_builder = StateGraph(ChildState)
child_builder.add_node("child_node", child_node)
child_builder.add_edge(START, "child_node")
child_graph = child_builder.compile()

parent_builder = StateGraph(ParentState)
parent_builder.add_node("parent_first", lambda s: s)
parent_builder.add_node("parent_target", lambda s: {"value": "reached"})
parent_builder.add_node("call_child", child_graph)
parent_builder.add_edge(START, "parent_first")
parent_builder.add_edge("parent_first", "call_child")
parent_builder.add_edge("parent_target", END)

parent_graph = parent_builder.compile()
```

### 组合用法

`Command` 的各参数可以自由组合：

```python
# 更新状态 + 恢复中断 + 路由
Command(
    update={"approved": True, "reviewer": "admin"},
    resume="approved",
    goto="execute_step",
)

# 更新状态 + 路由到多个目标
Command(
    update={"analysis": "complete"},
    goto=["step_a", "step_b"],
)
```

## interrupt_before / interrupt_after

除了在节点内部调用 `interrupt()` 外，还可以在编译图时通过 `interrupt_before` 和 `interrupt_after` 参数声明式中断：

### 在节点执行前/后中断

```python
builder = StateGraph(State)
builder.add_node("review", review_node)
builder.add_node("approve", approve_node)
builder.add_edge(START, "review")
builder.add_edge("review", "approve")
builder.add_edge("approve", END)

# 在 approve 节点执行前中断
graph = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["approve"],
)

# 在 review 节点执行后中断
graph2 = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_after=["review"],
)

# 在所有节点执行后中断
graph3 = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_after="*",
)
```

### 调试用例

`interrupt_before` 和 `interrupt_after` 在调试时非常有用，可以在每个步骤暂停查看状态：

```python
# 调试图：每个节点执行后暂停
debug_graph = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_after="*",
)

config = {"configurable": {"thread_id": "debug-1"}}

# 第一步
result = debug_graph.invoke({"input": "test"}, config)
state = debug_graph.get_state(config)
print(f"步骤1完成后: {state.values}")

# 继续下一步
debug_graph.invoke(None, config)
state = debug_graph.get_state(config)
print(f"步骤2完成后: {state.values}")
```

### 与 interrupt() 函数的区别

| 特性 | `interrupt()` | `interrupt_before/after` |
|------|-------------|------------------------|
| 定义位置 | 节点函数内部 | 编译图时 |
| 灵活性 | 可传递自定义值 | 仅暂停，不传递值 |
| 恢复方式 | `Command(resume=...)` | 直接调用 `invoke(None, config)` |
| 适用场景 | 需要收集用户输入 | 调试、审批关口 |

## 完整示例

### 示例 1：人类审核批准流程

```python
import uuid
from typing import Optional
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command


class ApprovalState(TypedDict):
    document: str
    risk_level: str
    approved: Optional[bool]
    comment: Optional[str]


def analyze_risk(state: ApprovalState) -> dict:
    """分析文档风险等级"""
    doc = state["document"]
    if "紧急" in doc or "重要" in doc:
        risk = "high"
    else:
        risk = "low"
    return {"risk_level": risk}


def human_review(state: ApprovalState) -> dict:
    """人类审核：高风险文档需要人工批准"""
    if state["risk_level"] == "high":
        result = interrupt({
            "message": "高风险文档需要审核批准",
            "document": state["document"],
            "risk_level": state["risk_level"],
        })
        return {"approved": result["approved"], "comment": result.get("comment", "")}
    else:
        return {"approved": True, "comment": "低风险，自动通过"}


def execute_action(state: ApprovalState) -> dict:
    """根据审核结果执行操作"""
    if state["approved"]:
        return {"comment": f"已执行: {state['comment']}"}
    else:
        return {"comment": f"已拒绝: {state['comment']}"}


builder = StateGraph(ApprovalState)
builder.add_node("analyze_risk", analyze_risk)
builder.add_node("human_review", human_review)
builder.add_node("execute_action", execute_action)
builder.add_edge(START, "analyze_risk")
builder.add_edge("analyze_risk", "human_review")
builder.add_edge("human_review", "execute_action")
builder.add_edge("execute_action", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# --- 使用示例 ---
config = {"configurable": {"thread_id": "approval-1"}}

# 第一步：提交高风险文档
for chunk in graph.stream({"document": "紧急：系统升级方案", "risk_level": "", "approved": None, "comment": None}, config):
    print(chunk)
# 输出包含中断信息

# 第二步：人类审核并恢复
for chunk in graph.stream(
    Command(resume={"approved": True, "comment": "同意，请执行"}),
    config,
):
    print(chunk)
# 输出: {'execute_action': {'comment': '已执行: 同意，请执行'}}
```

### 示例 2：多步骤表单收集

```python
import uuid
from typing import Optional
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command


class RegistrationState(TypedDict):
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    confirmed: bool


def collect_name(state: RegistrationState) -> dict:
    name = interrupt("请输入您的姓名")
    return {"name": name}


def collect_contact(state: RegistrationState) -> dict:
    email = interrupt("请输入您的邮箱")
    phone = interrupt("请输入您的手机号")
    return {"email": email, "phone": phone}


def confirm_info(state: RegistrationState) -> dict:
    summary = f"姓名: {state['name']}\n邮箱: {state['email']}\n手机: {state['phone']}"
    confirmed = interrupt(f"请确认以下信息:\n{summary}\n\n确认吗？(yes/no)")
    return {"confirmed": confirmed.lower() == "yes"}


builder = StateGraph(RegistrationState)
builder.add_node("collect_name", collect_name)
builder.add_node("collect_contact", collect_contact)
builder.add_node("confirm_info", confirm_info)
builder.add_edge(START, "collect_name")
builder.add_edge("collect_name", "collect_contact")
builder.add_edge("collect_contact", "confirm_info")
builder.add_edge("confirm_info", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "reg-1"}}

# 步骤1：输入姓名
graph.stream({"name": None, "email": None, "phone": None, "confirmed": False}, config)
for chunk in graph.stream(Command(resume="李明"), config):
    print(chunk)

# 步骤2：输入邮箱
for chunk in graph.stream(Command(resume="liming@example.com"), config):
    print(chunk)

# 步骤3：输入手机号
for chunk in graph.stream(Command(resume="13800138000"), config):
    print(chunk)

# 步骤4：确认信息
for chunk in graph.stream(Command(resume="yes"), config):
    print(chunk)
# 输出: {'confirm_info': {'confirmed': True}}
```

### 示例 3：子图中的中断向父图传播

```python
import uuid
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command


class ParentState(TypedDict):
    task: str
    result: str


class ChildState(TypedDict):
    sub_task: str
    approval: str


def child_process(state: ChildState):
    # 子图中的中断会自动传播到父图
    approval = interrupt("子任务需要确认：" + state["sub_task"])
    return {"approval": approval}


child_builder = StateGraph(ChildState)
child_builder.add_node("child_process", child_process)
child_builder.add_edge(START, "child_process")
child_builder.add_edge("child_process", END)
child_graph = child_builder.compile(checkpointer=True)


def parent_node(state: ParentState):
    result = child_graph.invoke({"sub_task": state["task"]})
    return {"result": result["approval"]}


builder = StateGraph(ParentState)
builder.add_node("parent_node", parent_node)
builder.add_edge(START, "parent_node")
builder.add_edge("parent_node", END)

# 父图也需要 checkpointer
graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "parent-child-1"}}

# 调用：子图的中断会传播到父图
for chunk in graph.stream({"task": "部署生产环境", "result": ""}, config):
    print(chunk)
# 输出包含子图的中断信息

# 从父图恢复子图的中断
for chunk in graph.stream(Command(resume="已确认"), config):
    print(chunk)
```

### 示例 4：带状态更新的恢复

```python
import uuid
from typing import Optional
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Command


class WorkflowState(TypedDict):
    query: str
    search_results: Optional[list]
    refined_query: Optional[str]
    final_answer: Optional[str]


def search(state: WorkflowState) -> dict:
    # 模拟搜索
    results = [f"结果1: {state['query']}", f"结果2: {state['query']}"]
    return {"search_results": results}


def refine_and_continue(state: WorkflowState) -> Command:
    # 使用 Command 同时更新状态并指定下一步
    return Command(
        update={"refined_query": state["query"] + " 详细"},
        goto="final_answer",
    )


def final_answer(state: WorkflowState) -> dict:
    return {"final_answer": f"基于 {state['refined_query']} 的最终答案"}


builder = StateGraph(WorkflowState)
builder.add_node("search", search)
builder.add_node("refine_and_continue", refine_and_continue)
builder.add_node("final_answer", final_answer)
builder.add_edge(START, "search")
builder.add_edge("search", "refine_and_continue")
builder.add_edge("final_answer", END)

graph = builder.compile()

result = graph.invoke({"query": "LangGraph", "search_results": None, "refined_query": None, "final_answer": None})
print(result)
# {'query': 'LangGraph', 'search_results': ['结果1: LangGraph', '结果2: LangGraph'],
#  'refined_query': 'LangGraph 详细', 'final_answer': '基于 LangGraph 详细 的最终答案'}
```

## 小结

- `interrupt(value)` 在节点内暂停执行，`value` 可以是任意 Python 对象
- `Command(resume=...)` 恢复被中断的执行，`resume` 值会作为 `interrupt()` 的返回值
- `Command(update=..., goto=...)` 可以同时更新状态和路由到指定节点
- `Command(graph=Command.PARENT, goto=...)` 用于子图向父图发送路由指令
- `interrupt_before/after` 适合调试和声明式审批关口，`interrupt()` 适合需要传递值的交互式场景
- 子图中的 `interrupt()` 会自动传播到父图，从父图用 `Command(resume=...)` 恢复