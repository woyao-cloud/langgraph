# 第三章 图构建——声明与编译

## 3.1 功能概览

`graph/` 模块是 LangGraph 的核心声明层。用户通过 `StateGraph` 以声明式 API 描述"有哪些节点""节点之间如何连接""条件路由走哪条路"，最后调用 `.compile()` 将构建器状态转化为可执行的 `CompiledStateGraph`（即 `Pregel` 实例）。整个流程可以概括为三步：

1. **定义状态模式** — 用 `TypedDict` 或 `Annotated` 声明状态字段及其 reducer
2. **声明图结构** — 通过 `add_node`、`add_edge`、`add_conditional_edges` 注册节点与边
3. **编译执行** — `.compile()` 将声明式结构转换为 Pregel 引擎能理解的 channels、nodes、writers

编译产物 `CompiledStateGraph` 继承自 `Pregel`，拥有 `invoke`、`stream`、`get_state` 等方法，可以立即运行。

## 3.2 应用场景

### (a) 简单线性流水线

最常见的场景——数据沿固定路径依次流过各节点：

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    text: str

def preprocess(state: State) -> dict:
    return {"text": state["text"].strip().lower()}

def analyze(state: State) -> dict:
    return {"text": f"[ANALYZED] {state['text']}"}

builder = StateGraph(State)
builder.add_node("preprocess", preprocess)
builder.add_node("analyze", analyze)
builder.add_edge(START, "preprocess")
builder.add_edge("preprocess", "analyze")
builder.add_edge("analyze", END)

graph = builder.compile()
result = graph.invoke({"text": "  Hello World  "})
# {'text': '[ANALYZED] hello world'}
```

`add_edge(START, "preprocess")` 将入口与预处理节点相连，`add_edge("analyze", END)` 标记终点。

### (b) 条件路由——Router 模式

当需要根据状态动态决定下一步走向时，使用 `add_conditional_edges`：

```python
from typing import Literal

class State(TypedDict):
    query: str
    category: str

def router(state: State) -> Literal["billing", "technical", "general"]:
    if "账单" in state["query"]:
        return "billing"
    elif "故障" in state["query"]:
        return "technical"
    return "general"

builder = StateGraph(State)
builder.add_node("router_node", lambda s: s)  # 占位节点
builder.add_node("billing", lambda s: {"category": "billing"})
builder.add_node("technical", lambda s: {"category": "technical"})
builder.add_node("general", lambda s: {"category": "general"})
builder.add_edge(START, "router_node")
builder.add_conditional_edges("router_node", router)
builder.add_edge("billing", END)
builder.add_edge("technical", END)
builder.add_edge("general", END)
```

`add_conditional_edges` 的 `path` 参数是一个可调用对象，返回值决定目标节点。通过 `Literal` 类型注解或 `path_map` 参数，图可视化工具可以精确绘制路由分支。

### (c) Map-Reduce：用 Send 实现并行扇出

当条件路由需要将同一输入派发给多个节点并行处理时，`path` 函数可以返回 `Send` 对象列表：

```python
from langgraph.types import Send

class State(TypedDict):
    topics: list[str]
    summaries: list[str]

def route_to_summarizers(state: State) -> list[Send]:
    return [Send("summarizer", {"topic": t}) for t in state["topics"]]

def summarizer(state: dict) -> dict:
    return {"summaries": [f"Summary of {state['topic']}"]}

builder = StateGraph(State)
builder.add_node("route", lambda s: s)
builder.add_node("summarizer", summarizer)
builder.add_conditional_edges(START, route_to_summarizers)
builder.add_edge("summarizer", END)
```

`Send` 将不同的输入分别路由到 `summarizer` 节点的多个实例，实现 map-reduce 模式。

### (d) 扇入：多节点汇聚写入同一状态

多个节点可以写入同一个 state key，reducer 负责合并：

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    results: Annotated[list, add]

def worker_a(state: State) -> dict:
    return {"results": ["A的结果"]}

def worker_b(state: State) -> dict:
    return {"results": ["B的结果"]}

builder = StateGraph(State)
builder.add_node("worker_a", worker_a)
builder.add_node("worker_b", worker_b)
builder.add_edge(START, "worker_a")
builder.add_edge(START, "worker_b")
builder.add_edge("worker_a", "aggregator")
builder.add_edge("worker_b", "aggregator")
```

使用 `add_edge(["worker_a", "worker_b"], "aggregator")` 可确保两个 worker 都完成后才执行 aggregator。`Annotated[list, add]` 让 `results` 字段使用 `operator.add` 作为 reducer，自动合并来自不同节点的列表。

### (e) 自定义输入/输出模式

当图的内部状态比外部暴露的接口更复杂时，可以用 `input_schema` 和 `output_schema` 控制出入：

```python
class InternalState(TypedDict):
    query: str
    intermediate: str
    result: str

class Input(TypedDict):
    query: str

class Output(TypedDict):
    result: str

builder = StateGraph(
    InternalState,
    input_schema=Input,
    output_schema=Output,
)
```

编译后 `graph.invoke({"query": "hello"})` 只需提供 `Input` 字段，返回结果只包含 `Output` 字段。`intermediate` 等内部字段对外部不可见。

### (f) add_messages reducer：聊天历史去重、编辑、删除

`add_messages` 是 LangGraph 内置的聊天消息 reducer，支持按 ID 去重、覆盖更新和删除：

```python
from typing import Annotated
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import add_messages, MessagesState

# 去重：相同 ID 的消息会被替换
msgs1 = [HumanMessage(content="你好", id="1")]
msgs2 = [HumanMessage(content="你好（修改后）", id="1")]
result = add_messages(msgs1, msgs2)
# [HumanMessage(content='你好（修改后）', id='1')]

# 删除：用 RemoveMessage 标记要删除的消息
msgs = [HumanMessage(content="要删除的消息", id="2")]
add_messages(msgs, [RemoveMessage(id="2")])
# [] — 消息被删除

# 在 StateGraph 中使用
class State(MessagesState):  # 等价于 messages: Annotated[list, add_messages]
    pass
```

`add_messages` 还支持 `format="langchain-openai"` 参数，自动将消息转换为 OpenAI API 兼容格式。

### (g) Context schema：运行作用域不可变数据

`context_schema` 暴露运行作用域的不可变上下文，如用户 ID、数据库连接等：

```python
from langgraph.runtime import Runtime

class Context(TypedDict):
    user_id: str
    db_url: str

class State(TypedDict):
    query: str
    result: str

def my_node(state: State, runtime: Runtime[Context]) -> dict:
    user = runtime.context["user_id"]
    return {"result": f"为用户 {user} 处理: {state['query']}"}

builder = StateGraph(State, context_schema=Context)
builder.add_node("my_node", my_node)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)
graph = builder.compile()
result = graph.invoke({"query": "hello"}, context={"user_id": "u123", "db_url": "..."})
```

与 `state` 不同，`context` 在整个运行期间不可变，不会在节点间传播修改。

### (h) Command 路由 vs 条件边

除了 `add_conditional_edges`，节点还可以返回 `Command` 对象来路由：

```python
from langgraph.types import Command

def decide_node(state: State) -> Command:
    if state["score"] > 80:
        return Command(goto="pass", update={"status": "approved"})
    return Command(goto="fail", update={"status": "rejected"})
```

`Command` 同时声明了"状态更新"和"下一步去向"，适合需要先更新状态再路由的场景。而 `add_conditional_edges` 是纯路由，不修改状态。

### (i) Node defer：延迟执行清理任务

`add_node` 的 `defer=True` 参数将节点标记为延迟执行，它会在运行即将结束时才执行，适合清理、日志等任务：

```python
builder.add_node("cleanup", cleanup_fn, defer=True)
```

在编译时，`defer=True` 的节点对应的 channel 使用 `NamedBarrierValueAfterFinish`，只有当所有前置节点完成后才触发。

## 3.3 Python 进阶

### Protocol（结构化子类型）

`_node.py` 定义了 9 个 `Protocol` 变体来描述节点函数签名：

```python
class _Node(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra) -> Any: ...

class _NodeWithConfig(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, config: RunnableConfig) -> Any: ...

class _NodeWithWriter(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, writer: StreamWriter) -> Any: ...
```

`Protocol` 实现了结构化子类型（structural subtyping）——任何具有匹配 `__call__` 签名的函数都自动满足协议，无需显式继承。这使得 LangGraph 可以接受纯函数、带 `config` 参数的函数、带 `writer` 的函数等多种形式。

### TypeAlias 与联合类型

`StateNode` 使用 `TypeAlias` 将 9 种 Protocol 变体与 `Runnable` 组合成联合类型：

```python
StateNode: TypeAlias = (
    _Node[NodeInputT]
    | _NodeWithConfig[NodeInputT]
    | _NodeWithWriter[NodeInputT]
    | ...
    | Runnable[NodeInputT, Any]
)
```

这让类型检查器接受任意合法节点形式，同时保持精确的类型推断。

### @overload

`add_node` 方法定义了 4 个 `@overload` 签名，区分"只传函数""传名称+函数""是否指定 input_schema"等组合。这使得 IDE 能根据参数组合提供精确的类型提示和自动补全。

### get_type_hints() 与 get_origin/get_args

`add_node` 内部使用 `get_type_hints()` 推断节点的输入类型，并从返回值类型 `Literal["node_a", "node_b"]` 中提取 `Command` 路由目标：

```python
if rtn_origin is Command and (rargs := get_args(rtn)):
    if get_origin(rargs[0]) is Literal:
        ends = get_args(rargs[0])
```

同样，`BranchSpec.from_path` 用 `get_type_hints` 推断路由函数返回的 `Literal` 类型以自动生成 `path_map`。

### functools.partial

`add_messages` 通过 `@_add_messages_wrapper` 装饰器将 `format` 等关键字参数部分应用（partial），使 reducer 可以带参数地用在 `Annotated` 中：

```python
@_add_messages_wrapper  # 使 add_messages(left, right) 和 add_messages(format="langchain-openai") 都合法
def add_messages(left, right, *, format=None): ...
```

### defaultdict 与 cast()

`StateGraph.branches` 使用 `defaultdict(dict)` 自动为每个源节点创建分支字典。`cast()` 在多处用于安抚类型检查器，如 `cast(type[InputT], input_schema or state_schema)`。

### Unpack

`add_node` 签名中的 `**kwargs: Unpack[DeprecatedKwargs]` 允许方法接受已弃用但未移除的关键字参数（如 `retry`），同时类型检查器不会报错。

### inspect.signature

`_get_branch_path_input_schema` 使用 `signature(callable_).parameters` 获取路由函数的首参数名，再配合 `get_type_hints` 推断输入 schema。

## 3.4 代码走读

### _node.py：9 种 Protocol 与 StateNodeSpec

`_node.py` 是类型系统的核心。它定义了：

- **_Node** — 最简形式 `(state) -> Any`
- **_NodeWithConfig** — `(state, config) -> Any`
- **_NodeWithWriter** — `(state, *, writer) -> Any`
- **_NodeWithStore** — `(state, *, store) -> Any`
- **_NodeWithWriterStore** — `(state, *, writer, store) -> Any`
- **_NodeWithConfigWriter** — `(state, *, config, writer) -> Any`
- **_NodeWithConfigStore** — `(state, *, config, store) -> Any`
- **_NodeWithConfigWriterStore** — `(state, *, config, writer, store) -> Any`
- **_NodeWithRuntime** — `(state, *, runtime) -> Any`

这些 Protocol 共享 `NodeInputT_contra`（逆变类型参数），确保输入类型的类型安全。运行时通过 `inspect.signature` 检查实际函数的参数名来决定注入哪些依赖。

`StateNodeSpec` 是一个 `@dataclass(slots=True)`，记录每个节点的元信息：

```python
@dataclass(slots=True)
class StateNodeSpec(Generic[NodeInputT, ContextT]):
    runnable: StateNode[NodeInputT, ContextT]
    metadata: dict[str, Any] | None
    input_schema: type[NodeInputT]
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    ends: tuple[str, ...] | dict[str, str] | None = EMPTY_SEQ
    defer: bool = False
```

`ends` 字段记录 `Command` 路由的目标节点（仅用于图可视化），`defer` 标记延迟执行。

### _branch.py：BranchSpec 与条件路由

`BranchSpec` 是一个 `NamedTuple`，封装了条件边的三要素：

```python
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]  # 路由函数
    ends: dict[Hashable, str] | None                 # 路由结果到节点名的映射
    input_schema: type[Any] | None = None             # 路由函数的输入类型
```

`from_path` 类方法做了两件关键的事：

1. **推断 path_map**：如果 `path_map` 未提供，则从路由函数的返回类型注解 `Literal["a", "b"]` 中提取候选目标
2. **推断 input_schema**：通过 `_get_branch_path_input_schema` 解析路由函数首参数的类型注解

`BranchSpec.run()` 将路由逻辑包装为 `RunnableCallable`，注册写入器。`_route/_aroute` 执行路由：先读取状态（通过 `reader`），调用 `path.invoke()`，然后通过 `writer` 将结果写入对应 channel。

### state.py：StateGraph 的构建逻辑

**`__init__`** 初始化时接受 `state_schema`、`context_schema`、`input_schema`、`output_schema`，并调用 `_add_schema` 为每个 schema 注册 channel：

```python
self.nodes = {}
self.edges = set()
self.branches = defaultdict(dict)
self.channels = {}
self.managed = {}
```

**`_add_schema`** 解析 TypedDict 的字段注解，将 `Annotated[type, reducer]` 转换为 `BinaryOperatorAggregate` 或 `LastValue` channel，存入 `self.channels`。

**`add_node`** 接受多种调用形式（函数直接传入或 `name, func` 元组），自动推断节点名称，创建 `StateNodeSpec` 并存入 `self.nodes`。如果函数有返回类型注解 `Command[Literal[...]]`，会自动提取目标节点到 `ends`。

**`add_edge`** 处理两种情况：
- 单一起点：`add_edge("A", "B")` 简单地加入 `self.edges` 集合
- 多起点：`add_edge(["A", "B"], "C")` 加入 `self.waiting_edges`，表示扇入

**`add_conditional_edges`** 将路由函数包装为 `BranchSpec.from_path()`，存入 `self.branches[source][name]`。

**`compile`** 是最关键的方法：

1. 调用 `validate()` 检查所有源/目标节点是否存在
2. 创建 `CompiledStateGraph` 实例（即 Pregel），初始化 channels（含 `START: EphemeralValue(input_schema)`）
3. 调用 `attach_node(START, None)` 注册入口节点
4. 遍历 `self.nodes`，对每个节点调用 `attach_node`
5. 遍历 `self.edges` 和 `self.waiting_edges`，对每条边调用 `attach_edge`
6. 遍历 `self.branches`，对每条条件边调用 `attach_branch`
7. 调用 `compiled.validate()` 完成最终验证

### message.py：add_messages reducer

`add_messages` 是一个装饰了 `@_add_messages_wrapper` 的函数。核心逻辑：

1. 将输入强制转为 `BaseMessage` 列表，用 `uuid.uuid4()` 为无 ID 消息分配 ID
2. 遇到 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 时清空所有消息，只保留其后的新消息
3. 遍历右侧消息：如果 ID 已存在于左侧，则覆盖（替换）；如果是 `RemoveMessage`，则标记删除
4. 新 ID 的消息直接追加
5. 最后按 `format` 参数做格式转换（如 `"langchain-openai"`）

`MessagesState` 是一个便捷 TypedDict，等价于 `messages: Annotated[list[AnyMessage], add_messages]`。

## 3.5 实现原理

### add_edge 如何变成 Pregel 的 channel + writer

编译时，`attach_edge` 根据边类型创建不同 channel：

**简单边（A → B）**：为源节点 A 添加一个 writer，写入 `branch:to:B` 的 `EphemeralValue` channel。B 节点订阅此 channel 作为触发器。

```python
# attach_edge 中简单边逻辑
self.nodes[starts].writers.append(
    ChannelWrite((ChannelWriteEntry(_CHANNEL_BRANCH_TO.format(end), None),))
)
```

**扇入边（[A, B] → C）**：创建一个 `NamedBarrierValue` channel（名称为 `join:A+B:C`），要求 A 和 B 都写入后才触发 C。如果 C 节点设置了 `defer=True`，则使用 `NamedBarrierValueAfterFinish`，等待所有前置节点完全结束后才触发。

```python
channel_name = f"join:{'+'.join(starts)}:{end}"
self.channels[channel_name] = NamedBarrierValue(str, set(starts))
self.nodes[end].triggers.append(channel_name)
for start in starts:
    self.nodes[start].writers.append(
        ChannelWrite((ChannelWriteEntry(channel_name, start),))
    )
```

### add_conditional_edges 如何变成 BranchSpec.run()

`attach_branch` 做了三件事：

1. 为路由函数创建一个 `ChannelRead` reader，从状态 channel 读取当前状态
2. 调用 `branch.run(get_writes, reader)` 生成 `RunnableCallable`，其中 `get_writes` 是将路由结果映射到 channel 写入的函数
3. 将这个 `RunnableCallable` 添加到源节点的 writers 列表

运行时，路由函数的返回值（如 `"billing"`）会被 `get_writes` 转换为 `ChannelWriteEntry("branch:to:billing", None)`，从而触发目标节点。

### compile 的完整转换链

`compile()` 将声明式结构转换为 Pregel 可执行结构：

| 声明式 API            | Pregel 结构                                           |
|----------------------|------------------------------------------------------|
| `add_node("A", fn)`  | `nodes["A"] = PregelNode(bound=fn, triggers=[...], writers=[...])` |
| `add_edge("A", "B")` | `channels["branch:to:B"] = EphemeralValue(...)` + writer on A |
| `add_edge(["A","B"], "C")` | `channels["join:A+B:C"] = NamedBarrierValue(...)` + writers on A, B |
| `add_conditional_edges` | `BranchSpec.run()` → writer on source node           |
| `START`              | `channels["__start__"] = EphemeralValue(input_schema)` |
| State key with reducer | `channels[key] = BinaryOperatorAggregate(type, reducer)` |
| State key without reducer | `channels[key] = LastValue(type)`                  |

## 3.6 动手实验

### 实验 1：构建简单线性图并检查编译结构

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int

def double(state: State) -> dict:
    return {"value": state["value"] * 2}

builder = StateGraph(State)
builder.add_node("double", double)
builder.add_edge(START, "double")
builder.add_edge("double", END)

graph = builder.compile()

# 检查编译后的 channel 结构
print("Channels:", list(graph.channels.keys()))
# ['value', '__start__']

# 检查节点及其触发器和写入器
for name, node in graph.nodes.items():
    print(f"Node: {name}, triggers: {node.triggers}")
    for w in node.writers:
        print(f"  Writer entries: {w.entries}")
```

### 实验 2：条件路由与 Literal 推断

```python
from typing import Literal

class State(TypedDict):
    score: int
    result: str

def grade(state: State) -> Literal["pass", "fail"]:
    return "pass" if state["score"] >= 60 else "fail"

def pass_node(state: State) -> dict:
    return {"result": "及格"}

def fail_node(state: State) -> dict:
    return {"result": "不及格"}

builder = StateGraph(State)
builder.add_node("grade", grade)
builder.add_node("pass", pass_node)
builder.add_node("fail", fail_node)
builder.add_edge(START, "grade")
builder.add_conditional_edges("grade", grade)
builder.add_edge("pass", END)
builder.add_edge("fail", END)

graph = builder.compile()
# 检查 branches 结构
for source, branches in builder.branches.items():
    for name, spec in branches.items():
        print(f"Branch from {source}: ends={spec.ends}")
```

### 实验 3：扇入与 add_messages

```python
from typing import Annotated
from operator import add
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages, MessagesState

# 测试 add_messages 去重与覆盖
msgs1 = [HumanMessage(content="Hi", id="1"), AIMessage(content="Hello", id="2")]
msgs2 = [HumanMessage(content="Hi (edited)", id="1")]  # 相同 ID，覆盖
result = add_messages(msgs1, msgs2)
print(f"去重后数量: {len(result)}")  # 2（不是 3）
print(f"内容: {[m.content for m in result]}")  # ['Hi (edited)', 'Hello']

# 测试扇入 reducer
class State(TypedDict):
    results: Annotated[list, add]

builder = StateGraph(State)
builder.add_node("a", lambda s: {"results": ["A"]})
builder.add_node("b", lambda s: {"results": ["B"]})
builder.add_node("merge", lambda s: {"results": [f"Merged: {s['results']}"]})
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge(["a", "b"], "merge")
builder.add_edge("merge", END)

graph = builder.compile()
print(graph.invoke({"results": []}))
# {'results': ["Merged: ['A', 'B']"]}
```

### 实验 4：检查 NamedBarrierValue channel

```python
# 构建扇入图后检查 channels
from langgraph.channels.named_barrier_value import NamedBarrierValue

graph = builder.compile()
for name, channel in graph.channels.items():
    if isinstance(channel, NamedBarrierValue):
        print(f"Barrier channel: {name}, names: {channel.names}")
```

这会显示扇入产生的 `join:a+b:merge` channel，以及它等待哪些节点完成。