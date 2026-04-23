# 第八章：高级路由

## 条件边深入

### 路由函数的签名和返回类型

条件边通过 `add_conditional_edges` 添加，路由函数接收当前状态并返回下一个节点的名称：

```python
from langgraph.graph import StateGraph, START, END
from typing import Literal


class State(dict):
    query: str
    category: str


def classifier(state: State) -> Literal["tech_support", "billing", "__end__"]:
    if "技术" in state["query"]:
        return "tech_support"
    elif "账单" in state["query"]:
        return "billing"
    else:
        return END


builder = StateGraph(State)
builder.add_node("classifier", lambda s: s)
builder.add_node("tech_support", lambda s: {"category": "tech"})
builder.add_node("billing", lambda s: {"category": "billing"})

# 路由函数返回字符串，对应目标节点名称
builder.add_conditional_edges("classifier", classifier)

builder.add_edge(START, "classifier")
builder.add_edge("tech_support", END)
builder.add_edge("billing", END)
```

路由函数的返回类型可以是：
- 单个字符串（节点名称或 `END`）
- 字符串列表（fan-out，并行执行多个节点）
- `Send` 对象（动态 fan-out，带自定义输入）
- `Command` 对象（同时更新状态和路由）

### Literal 自动推断目标节点

当路由函数的返回类型注解使用 `Literal` 时，LangGraph 会自动推断可能的目标节点：

```python
from typing import Literal


def router(state: State) -> Literal["node_a", "node_b", "__end__"]:
    """返回类型注解让 LangGraph 知道可能的目标节点"""
    if state["value"] > 0:
        return "node_a"
    elif state["value"] < 0:
        return "node_b"
    else:
        return END


builder.add_conditional_edges("source_node", router)
```

如果不使用 `Literal` 类型注解，建议提供 `path_map`，否则图可视化会假设所有节点都是可能的目标。

### path_map 字典映射

`path_map` 用于将路由函数的返回值映射到实际节点名称：

```python
def get_category(state: State) -> str:
    # 返回逻辑分类码
    if state["score"] > 80:
        return "high"
    elif state["score"] > 50:
        return "medium"
    else:
        return "low"


# path_map 将分类码映射到实际节点名称
builder.add_conditional_edges(
    "source_node",
    get_category,
    path_map={"high": "fast_track", "medium": "normal_track", "low": "review"},
)
```

`path_map` 也可以传入字符串列表，此时将自动生成从索引到节点名的映射：

```python
# 等价于 path_map={"0": "node_a", "1": "node_b", "2": "node_c"}
builder.add_conditional_edges("source", router, path_map=["node_a", "node_b", "node_c"])
```

### 返回多个目标（fan-out）

路由函数可以返回列表，让多个节点并行执行：

```python
from typing import Annotated
import operator


class State(TypedDict):
    query: str
    results: Annotated[list, operator.add]


def fan_out(state: State) -> list[str]:
    """返回多个目标节点名称，它们将并行执行"""
    return ["search_web", "search_db", "search_cache"]


builder = StateGraph(State)
builder.add_node("search_web", lambda s: {"results": ["web_result"]})
builder.add_node("search_db", lambda s: {"results": ["db_result"]})
builder.add_node("search_cache", lambda s: {"results": ["cache_result"]})

builder.add_conditional_edges(START, fan_out)
builder.add_edge("search_web", END)
builder.add_edge("search_db", END)
builder.add_edge("search_cache", END)

graph = builder.compile()
result = graph.invoke({"query": "test", "results": []})
# result["results"] == ["web_result", "db_result", "cache_result"]
```

## Send：动态 fan-out

### 基本用法

`Send` 允许在条件边中动态地向目标节点发送自定义输入，而不是使用全局状态：

```python
from langgraph.types import Send


def route_to_workers(state: State) -> list[Send]:
    """动态创建并行任务，每个任务有独立的输入"""
    return [
        Send("worker", {"task": f"任务{i}"})
        for i in range(state["task_count"])
    ]


builder.add_conditional_edges(START, route_to_workers)
```

### Map-Reduce 模式详解

Map-Reduce 是 `Send` 最典型的应用场景——将大任务拆分为多个子任务并行执行，再汇总结果：

```python
from typing import Annotated, TypedDict
import operator
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send


class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]


class JokeState(TypedDict):
    subject: str


def generate_joke(state: JokeState) -> dict:
    """为单个主题生成笑话"""
    return {"jokes": [f"关于{state['subject']}的笑话"]}


def continue_to_jokes(state: OverallState) -> list[Send]:
    """Map 阶段：为每个主题创建并行任务"""
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


builder = StateGraph(OverallState)
builder.add_node("generate_joke", generate_joke)
builder.add_conditional_edges(START, continue_to_jokes)
builder.add_edge("generate_joke", END)

graph = builder.compile()
result = graph.invoke({"subjects": ["猫", "狗", "程序员"], "jokes": []})
print(result["jokes"])
# ['关于猫的笑话', '关于狗的笑话', '关于程序员的笑话']
```

### Send vs 条件边：何时用哪个

| 特性 | 条件边（返回字符串） | Send |
|------|-------------------|------|
| 输入 | 使用全局状态 | 传递自定义输入 |
| 并行度 | 固定（返回列表时） | 动态（运行时决定） |
| 目标节点 | 按名称路由 | 按名称路由 + 自定义 arg |
| 适用场景 | 固定分支逻辑 | Map-Reduce、动态并行 |

- **使用条件边**：当你需要根据状态选择不同的处理路径（如分类、审批流程）
- **使用 Send**：当你需要动态数量的并行任务，且每个任务需要不同的输入（如批量处理文档）

### Send 的 arg 参数

`Send` 的 `arg` 参数会作为目标节点的输入状态传入，而不是与全局状态合并：

```python
# Send 的 arg 传递给目标节点
Send("summarize", {"doc": "文档内容...", "length": "short"})

# 目标节点接收的是 arg 中的内容，而非全局状态
def summarize(state):
    # state 是 {"doc": "文档内容...", "length": "short"}
    return {"summary": state["doc"][:state["length"]]}
```

## Command：灵活控制

### Command(goto=...)：显式路由

`Command` 提供比条件边更灵活的路由方式。节点可以直接返回 `Command` 来指定下一步：

```python
from langgraph.types import Command
from typing import Literal


class State(TypedDict):
    value: int
    result: str


def decide(state: State) -> Command[Literal["process_high", "process_low", "__end__"]]:
    if state["value"] > 100:
        return Command(goto="process_high")
    elif state["value"] > 0:
        return Command(goto="process_low")
    else:
        return Command(goto=END)


builder = StateGraph(State)
builder.add_node("decide", decide)
builder.add_node("process_high", lambda s: {"result": "high"})
builder.add_node("process_low", lambda s: {"result": "low"})
builder.add_edge(START, "decide")
builder.add_edge("process_high", END)
builder.add_edge("process_low", END)
```

### Command(update=..., goto=...)：同时更新和路由

这是 `Command` 最强大的用法——在一个操作中同时更新状态并路由：

```python
def process(state: State) -> Command[Literal["next_step", "__end__"]]:
    # 先更新状态，然后路由到 next_step
    return Command(
        update={"result": "processed", "value": state["value"] * 2},
        goto="next_step",
    )
```

### Command(goto=[Send(...), "node"])：混合路由

`Command.goto` 可以混合使用节点名称和 `Send` 对象：

```python
def dispatch(state: State) -> Command:
    return Command(
        update={"dispatched": True},
        goto=[
            Send("worker", {"task_id": i}) for i in range(3)
        ] + ["collector"],
    )
```

### Command(graph=Command.PARENT)：子图向父图发送

在嵌套子图中，`Command(graph=Command.PARENT)` 可以让子图节点直接影响父图的执行流程：

```python
from langgraph.types import Command


class ParentState(TypedDict):
    jump_from_idx: int


class ChildState(TypedDict):
    jump: bool


# 子图节点决定跳转到父图的节点
def child_node(state: ChildState) -> Command | ChildState:
    if state["jump"]:
        return Command(graph=Command.PARENT, goto="parent_target")
    return state


child_builder = StateGraph(ChildState)
child_builder.add_node("child_node", child_node)
child_builder.add_edge(START, "child_node")
child_graph = child_builder.compile()

parent_builder = StateGraph(ParentState)
parent_builder.add_node("parent_first", lambda s: s)
parent_builder.add_node("parent_target", lambda s: {"jump_from_idx": s["jump_from_idx"]})
parent_builder.add_node("call_child", child_graph)
parent_builder.add_edge(START, "parent_first")
parent_builder.add_edge("parent_first", "call_child")
parent_builder.add_edge("parent_target", END)

parent_graph = parent_builder.compile()
```

## 子图

### 什么是子图：图中的图

子图是将一个编译好的图作为节点添加到另一个图中：

```python
# 子图
child_builder = StateGraph(ChildState)
child_builder.add_node("process", process_node)
child_builder.add_edge(START, "process")
child_builder.add_edge("process", END)
child_graph = child_builder.compile()

# 父图
parent_builder = StateGraph(ParentState)
parent_builder.add_node("call_child", child_graph)  # 子图作为节点
parent_builder.add_edge(START, "call_child")
parent_builder.add_edge("call_child", END)
parent_graph = parent_builder.compile()
```

### 子图的 Checkpointer：None / True / False

子图的 checkpointer 有三种配置：

```python
# 继承父图的 checkpointer（默认行为）
child_graph = child_builder.compile(checkpointer=None)

# 启用独立持久化
child_graph = child_builder.compile(checkpointer=True)

# 禁用持久化（即使父图有 checkpointer）
child_graph = child_builder.compile(checkpointer=False)

# 使用自定义 checkpointer
from langgraph.checkpoint.memory import InMemorySaver
child_graph = child_builder.compile(checkpointer=InMemorySaver())
```

- `None`（默认）：继承父图的 checkpointer，共享持久化
- `True`：启用独立持久化，子图有自己的检查点
- `False`：完全禁用子图的持久化

### 子图的状态隔离

子图有独立的状态类型，与父图状态通过键名自动映射：

```python
class ParentState(TypedDict):
    message: str
    child_result: str


class ChildState(TypedDict):
    message: str  # 同名字段会自动映射
    processed: str


def child_process(state: ChildState):
    return {"processed": f"已处理: {state['message']}"}


child_builder = StateGraph(ChildState)
child_builder.add_node("process", child_process)
child_builder.add_edge(START, "process")
child_builder.add_edge("process", END)
child_graph = child_builder.compile()

# 父图调用子图时，同名字段（如 message）会自动传递
parent_builder = StateGraph(ParentState)
parent_builder.add_node("call_child", child_graph)
parent_builder.add_edge(START, "call_child")
parent_builder.add_edge("call_child", END)
```

### 子图中的 interrupt 如何传播到父图

子图中的 `interrupt()` 会自动传播到父图。当父图恢复执行时，子图也会从中断点继续：

```python
# 子图中的中断会传播到父图
def child_node_with_interrupt(state: ChildState):
    approval = interrupt("子任务需要确认")
    return {"processed": f"已确认: {approval}"}

# 父图只需正常恢复，子图会自动继续
graph.invoke(Command(resume="确认"), config)
```

### 输入/输出模式在子图中的应用

通过 `add_node` 的 `input` 和 `output` 参数可以控制父子图之间的状态转换：

```python
from langgraph.graph import add_node

# 只传入特定字段给子图
parent_builder.add_node(
    "call_child",
    child_graph,
    input={"message": "parent_message"},  # 映射字段名
)
```

## 完整示例

### 示例 1：Map-Reduce 文档摘要

```python
from typing import Annotated, TypedDict
import operator
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send


class OverallState(TypedDict):
    documents: list[str]
    summaries: Annotated[list[str], operator.add]
    final_summary: str


class DocState(TypedDict):
    doc: str


def summarize_doc(state: DocState) -> dict:
    """对单个文档生成摘要"""
    doc = state["doc"]
    # 模拟摘要生成
    summary = f"摘要: {doc[:20]}..."
    return {"summaries": [summary]}


def map_to_summarize(state: OverallState) -> list[Send]:
    """Map：为每个文档创建并行摘要任务"""
    return [Send("summarize_doc", {"doc": doc}) for doc in state["documents"]]


def reduce_summaries(state: OverallState) -> dict:
    """Reduce：汇总所有摘要"""
    combined = "\n".join(state["summaries"])
    return {"final_summary": f"综合摘要:\n{combined}"}


builder = StateGraph(OverallState)
builder.add_node("summarize_doc", summarize_doc)
builder.add_node("reduce", reduce_summaries)
builder.add_conditional_edges(START, map_to_summarize)
builder.add_edge("summarize_doc", "reduce")
builder.add_edge("reduce", END)

graph = builder.compile()

result = graph.invoke({
    "documents": ["文档A的完整内容...", "文档B的完整内容...", "文档C的完整内容..."],
    "summaries": [],
    "final_summary": "",
})
print(result["final_summary"])
```

### 示例 2：多 Agent 协作（子图 + Command）

```python
from typing import Literal, Optional
from typing_extensions import TypedDict
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Command


class OrchestratorState(TypedDict):
    task: str
    research_result: Optional[str]
    writing_result: Optional[str]
    final_output: str


# 研究员子图
class ResearchState(TypedDict):
    topic: str
    findings: str


def research_node(state: ResearchState) -> dict:
    return {"findings": f"关于{state['topic']}的研究发现"}


research_builder = StateGraph(ResearchState)
research_builder.add_node("do_research", research_node)
research_builder.add_edge(START, "do_research")
research_builder.add_edge("do_research", END)
research_graph = research_builder.compile()

# 写手子图
class WritingState(TypedDict):
    research: str
    article: str


def write_node(state: WritingState) -> dict:
    return {"article": f"基于 {state['research']} 撰写的文章"}


writing_builder = StateGraph(WritingState)
writing_builder.add_node("write", write_node)
writing_builder.add_edge(START, "write")
writing_builder.add_edge("write", END)
writing_graph = writing_builder.compile()


# 编排器
def call_research(state: OrchestratorState) -> dict:
    result = research_graph.invoke({"topic": state["task"], "findings": ""})
    return {"research_result": result["findings"]}


def call_writing(state: OrchestratorState) -> dict:
    result = writing_graph.invoke({"research": state["research_result"], "article": ""})
    return {"writing_result": result["article"]}


def route_task(state: OrchestratorState) -> Command[Literal["call_research", "call_writing", "__end__"]]:
    if not state.get("research_result"):
        return Command(goto="call_research")
    elif not state.get("writing_result"):
        return Command(goto="call_writing")
    else:
        return Command(goto=END)


def finalize(state: OrchestratorState) -> dict:
    return {"final_output": state["writing_result"]}


builder = StateGraph(OrchestratorState)
builder.add_node("call_research", call_research)
builder.add_node("call_writing", call_writing)
builder.add_node("finalize", finalize)
builder.add_edge(START, "call_research")
builder.add_edge("call_research", "call_writing")
builder.add_edge("call_writing", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()
result = graph.invoke({"task": "AI发展趋势", "research_result": None, "writing_result": None, "final_output": ""})
print(result["final_output"])
```

### 示例 3：嵌套子图的状态传递

```python
from typing import Optional
from typing_extensions import TypedDict
from langgraph.constants import START, END
from langgraph.graph import StateGraph


# 最内层子图：执行具体操作
class WorkerState(TypedDict):
    task_detail: str
    task_result: str


def execute_task(state: WorkerState) -> dict:
    return {"task_result": f"完成: {state['task_detail']}"}


worker_builder = StateGraph(WorkerState)
worker_builder.add_node("execute", execute_task)
worker_builder.add_edge(START, "execute")
worker_builder.add_edge("execute", END)
worker_graph = worker_builder.compile()


# 中间层子图：协调多个 worker
class ManagerState(TypedDict):
    task_name: str
    sub_results: str


def delegate_task(state: ManagerState) -> dict:
    result = worker_graph.invoke({"task_detail": state["task_name"], "task_result": ""})
    return {"sub_results": result["task_result"]}


manager_builder = StateGraph(ManagerState)
manager_builder.add_node("delegate", delegate_task)
manager_builder.add_edge(START, "delegate")
manager_builder.add_edge("delegate", END)
manager_graph = manager_builder.compile()


# 顶层图：项目协调
class ProjectState(TypedDict):
    project: str
    report: str


def run_manager(state: ProjectState) -> dict:
    result = manager_graph.invoke({"task_name": state["project"], "sub_results": ""})
    return {"report": result["sub_results"]}


project_builder = StateGraph(ProjectState)
project_builder.add_node("run_manager", run_manager)
project_builder.add_edge(START, "run_manager")
project_builder.add_edge("run_manager", END)

project_graph = project_builder.compile()

result = project_graph.invoke({"project": "数据分析", "report": ""})
print(result["report"])
# 输出: 完成: 数据分析
```

## 小结

- 条件边通过返回值（字符串、列表、`Send`、`Command`）控制路由
- `Literal` 类型注解让 LangGraph 自动推断目标节点，`path_map` 用于值映射
- `Send` 实现动态 fan-out，每个任务可携带自定义输入，适合 Map-Reduce 模式
- `Command` 提供最灵活的控制：同时更新状态和路由，支持混合 `Send` 和节点名
- `Command(graph=Command.PARENT)` 让子图节点可以影响父图执行流程
- 子图有独立状态，同名字段自动映射，`interrupt` 自动传播到父图