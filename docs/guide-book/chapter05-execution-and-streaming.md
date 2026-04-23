# 第五章 执行与流式输出

LangGraph 基于 Pregel 模型（Bulk Synchronous Parallel，即 BSP 超步模型）执行图。理解执行模型是掌握流式输出和状态管理的基础。本章将深入讲解执行原理与七种流式输出模式。

## 5.1 BSP 超步详解

LangGraph 的执行被组织为一系列"超步"（superstep）。每个超步包含以下阶段：

1. **准备任务**：根据上一超步的通道更新，确定哪些节点需要执行。
2. **检查中断**：如果在 `interrupt_before` 列表中的节点即将执行，则暂停当前超步，将中断信息保存到检查点。
3. **执行**：并行执行所有待执行节点。每个节点读取当前状态，计算更新，写入输出。
4. **应用写入**：将所有节点的输出合并到状态中。每个状态字段根据其 reducer 函数聚合写入值。
5. **保存检查点**：如果配置了 checkpointer，持久化当前状态快照。

**关键特性：同一超步中的节点看不到彼此的写入。** 这是因为写入在超步结束时才统一应用。这种设计保证了并行节点之间不会产生依赖冲突。

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    results: Annotated[list[str], operator.add]  # reducer: 列表拼接

def node_a(state: State) -> dict:
    # node_a 看不到 node_b 在同一超步的写入
    return {"results": ["A"]}

def node_b(state: State) -> dict:
    # node_b 看不到 node_a 在同一超步的写入
    return {"results": ["B"]}

def aggregate(state: State) -> dict:
    # 在下一个超步中，两个写入都已合并
    return {"results": [f"聚合：{state['results']}"]}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("agg", aggregate)
builder.add_edge(START, "a")
builder.add_edge(START, "b")  # a 和 b 在同一超步并行执行
builder.add_edge("a", "agg")
builder.add_edge("b", "agg")
builder.add_edge("agg", END)

graph = builder.compile()
result = graph.invoke({"results": []})
# result["results"] = ["A", "B", "聚合：['A', 'B']"]
```

### 递归限制

`recursion_limit` 是 `RunnableConfig` 中的一个参数，默认值为 25，限制图执行的最大超步数。超过此限制会抛出 `GraphRecursionError`。

```python
result = graph.invoke(
    {"query": "hello"},
    config={"recursion_limit": 10},  # 最多 10 个超步
)
```

## 5.2 invoke() vs stream() vs astream()

### invoke()：一次性返回最终结果

```python
result = graph.invoke(
    {"query": "你好"},
    config={"configurable": {"thread_id": "t1"}},
)
# result 是最终状态的字典
```

`invoke()` 阻塞执行直到完成，直接返回最终结果。适合不需要中间过程的场景。

### stream()：逐步返回中间结果

```python
for chunk in graph.stream(
    {"query": "你好"},
    config={"configurable": {"thread_id": "t1"}},
    stream_mode="updates",
):
    print(chunk)
```

`stream()` 是同步生成器，每完成一个超步就产出一次结果。适合需要实时展示进度的场景。

### astream()：异步流式

```python
async for chunk in graph.astream(
    {"query": "你好"},
    config={"configurable": {"thread_id": "t1"}},
    stream_mode="updates",
):
    print(chunk)
```

`astream()` 是 `stream()` 的异步版本，在 async 上下文中使用，不会阻塞事件循环。

## 5.3 七种 StreamMode 详解

`stream_mode` 参数控制流式输出的格式。LangGraph 提供七种模式，可以单独使用也可以组合使用。

### 5.3.1 "values"：完整状态快照

每步执行后输出完整的当前状态。这是默认模式。

```python
for state in graph.stream(
    {"query": "LangGraph 是什么？", "log": []},
    stream_mode="values",
):
    print(state)
# 第一步后：{"query": "LangGraph 是什么？", "research_result": "...", ...}
# 第二步后：{"query": "LangGraph 是什么？", "research_result": "...", "draft": "...", ...}
# 最终步后：完整最终状态
```

### 5.3.2 "updates"：增量更新

只输出每个节点产生的增量更新，而不是完整状态。格式为 `{节点名: 更新字典}`。

```python
for update in graph.stream(
    {"query": "LangGraph 是什么？", "log": []},
    stream_mode="updates",
):
    print(update)
# {"research": {"research_result": "...", "log": ["research"]}}
# {"draft": {"draft": "草稿：...", "log": ["draft"]}}
```

如果在同一超步中有多个节点并行执行，它们的更新会分别产出。

### 5.3.3 "messages"：LLM token 流

专门为 LLM 场景设计。逐 token 输出 LLM 的响应，同时附带元数据。元数据包含 `langgraph_step`、`langgraph_node`、`langgraph_triggers` 等信息。

```python
from langchain_core.messages import HumanMessage

for msg_and_meta in graph.stream(
    {"messages": [HumanMessage(content="写一首关于AI的诗")]},
    config={"configurable": {"thread_id": "t1"}},
    stream_mode="messages",
):
    msg, metadata = msg_and_meta
    if isinstance(msg, AIMessageChunk):
        print(msg.content, end="", flush=True)
# 逐 token 输出：在数据之巅我站立...
```

### 5.3.4 "custom"：自定义流输出

通过 `StreamWriter` 在节点内部产出自定义数据。`StreamWriter` 是一个注入的 `Callable`，只在 `stream_mode="custom"` 时产出数据。

```python
from langgraph.types import StreamWriter

def my_node(state: State, *, writer: StreamWriter) -> dict:
    writer({"step": 1, "msg": "开始处理"})
    result = do_expensive_computation(state)
    writer({"step": 2, "msg": "处理完成"})
    return {"result": result}

builder = StateGraph(State)
builder.add_node("my_node", my_node)

for custom_data in graph.stream(
    input_data,
    stream_mode="custom",
):
    print(custom_data)
# {"step": 1, "msg": "开始处理"}
# {"step": 2, "msg": "处理完成"}
```

### 5.3.5 "checkpoints"：检查点事件

每当检查点被创建时，输出一个事件。数据格式与 `get_state()` 返回的 `StateSnapshot` 一致。

```python
for checkpoint in graph.stream(
    input_data,
    config={"configurable": {"thread_id": "t1"}},
    stream_mode="checkpoints",
):
    print(f"步骤 {checkpoint['metadata']['step']}: {checkpoint['values']}")
```

### 5.3.6 "tasks"：任务生命周期事件

输出任务级别的开始和完成事件。

```python
for task_event in graph.stream(
    input_data,
    stream_mode="tasks",
):
    if "error" in task_event:
        # 任务完成（可能失败）
        print(f"任务 {task_event['name']} 失败: {task_event['error']}")
    elif "result" in task_event:
        # 任务成功完成
        print(f"任务 {task_event['name']} 完成")
    else:
        # 任务开始
        print(f"任务 {task_event['name']} 开始，输入: {task_event['input']}")
```

### 5.3.7 "debug"：调试信息

等价于同时启用 `"checkpoints"` 和 `"tasks"`，输出最详细的执行信息。

```python
for debug_event in graph.stream(
    input_data,
    config={"configurable": {"thread_id": "t1"}},
    stream_mode="debug",
):
    print(debug_event)
# 包含 step、timestamp、type("checkpoint" | "task" | "task_result") 和 payload
```

## 5.4 流式输出的 v1 vs v2

`version` 参数控制流式输出的数据格式：

- **v1**（默认）：`stream()` 返回原始字典或值，格式取决于 `stream_mode`。
- **v2**：返回结构化的 `StreamPart` TypedDict，包含 `type` 字段用于区分不同流模式的数据。

```python
# v1 格式（默认）
for chunk in graph.stream(input_data, stream_mode="updates"):
    print(chunk)
# {"node_name": {"key": "value"}}

# v2 格式
for part in graph.stream(input_data, stream_mode="updates", version="v2"):
    print(part["type"])   # "updates"
    print(part["data"])   # {"node_name": {"key": "value"}}
    print(part["ns"])     # 命名空间元组
```

v2 格式的优势在于：
- 每个 chunk 都有明确的 `type` 字段，便于多流模式组合时区分数据来源。
- 包含 `ns`（命名空间）字段，用于子图场景下定位事件来源。
- 类型安全：`StreamPart` 是 TypedDict，IDE 可以提供更好的类型提示。

## 5.5 多流模式组合

`stream_mode` 参数支持传入列表，同时启用多种流模式。此时输出为 `(mode, data)` 元组。

```python
for mode, data in graph.stream(
    input_data,
    config={"configurable": {"thread_id": "t1"}},
    stream_mode=["messages", "updates"],
):
    if mode == "messages":
        msg, metadata = data
        if hasattr(msg, "content"):
            print(msg.content, end="", flush=True)
    elif mode == "updates":
        print(f"\n节点更新: {data}")
```

使用 v2 格式时，多流模式的每个 chunk 是 `StreamPart` TypedDict，通过 `part["type"]` 区分：

```python
for part in graph.stream(
    input_data,
    config={"configurable": {"thread_id": "t1"}},
    stream_mode=["messages", "updates"],
    version="v2",
):
    match part["type"]:
        case "messages":
            msg, metadata = part["data"]
            print(msg.content, end="")
        case "updates":
            print(f"\n更新: {part['data']}")
        case "custom":
            print(f"自定义: {part['data']}")
```

## 5.6 完整示例：同时获取 messages 和 custom 流

以下示例展示如何在代理中同时获取 LLM 的 token 流和自定义进度信息：

```python
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StreamWriter

class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

def chatbot(state: State, *, writer: StreamWriter) -> dict:
    writer({"type": "status", "message": "正在调用 LLM..."})
    response = llm.invoke(state["messages"])
    writer({"type": "status", "message": "LLM 调用完成"})
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 同时获取 messages 和 custom 流
config = {"configurable": {"thread_id": "demo"}}

for mode, data in graph.stream(
    {"messages": [HumanMessage(content="解释一下量子计算")]},
    config,
    stream_mode=["messages", "custom"],
):
    if mode == "messages":
        msg, metadata = data
        if isinstance(msg, AIMessageChunk) and msg.content:
            print(msg.content, end="", flush=True)
    elif mode == "custom":
        print(f"\n[状态] {data['message']}")
```

此示例中，`stream_mode=["messages", "custom"]` 让你既能实时显示 LLM 的 token 流，又能通过 `StreamWriter` 输出自定义的进度信息。这是构建交互式代理应用的推荐模式。