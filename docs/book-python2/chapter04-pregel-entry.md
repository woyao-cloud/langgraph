# 第四章 Pregel 引擎——入口与执行协议

## 4.1 功能概览

`Pregel` 是 LangGraph 的运行时引擎。`StateGraph.compile()` 的产物 `CompiledStateGraph` 就是 `Pregel` 的子类——用户调用的 `invoke`、`stream`、`get_state`、`update_state` 等方法全部定义在 `Pregel` 上。`PregelProtocol` 则定义了这些方法的抽象协议。

从架构上看，Pregel 实现了 **BSP（Bulk Synchronous Parallel）模型**：

- **Plan**：根据上一步更新的 channel，决定下一步需要执行哪些节点
- **Execute**：并行执行所有选中的节点
- **Update**：将节点输出写入 channel，更新状态

循环直到没有新节点被触发或达到递归上限。

`Pregel` 也可以不通过 `StateGraph` 直接使用——通过 `NodeBuilder` 的流式 API 手动构建 channels 和 nodes，适合高级用户对执行模型进行细粒度控制。

## 4.2 应用场景

### (a) 简单 invoke：一次性执行

最基础的用法——执行图并获取最终结果：

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    count: int

def increment(state: State) -> dict:
    return {"count": state["count"] + 1}

builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)
graph = builder.compile()

result = graph.invoke({"count": 0})
print(result)  # {'count': 1}
```

`invoke` 内部实际上是调用 `stream` 并收集最后一个 chunk。对于一次性执行（无需逐步观察中间状态），这是最简洁的方式。

### (b) stream mode "values"：每步后的完整状态

```python
for state in graph.stream({"count": 0}, stream_mode="values"):
    print(state)
# {'count': 0}    ← 初始状态
# {'count': 1}    ← increment 节点执行后
```

`"values"` 模式在每一步执行后输出完整状态快照，适合需要观察状态逐步变化的场景。

### (c) stream mode "updates"：节点级增量

```python
for chunk in graph.stream({"count": 0}, stream_mode="updates"):
    print(chunk)
# {'increment': {'count': 1}}
```

`"updates"` 模式只输出每个节点的增量更新（节点名 → 更新内容），不包含其他未修改的字段。这在多节点并行执行时尤为有用——每个节点完成后立即发出自己的更新。

### (d) stream mode "messages"：逐 token LLM 输出

```python
for chunk in graph.stream(
    {"messages": [HumanMessage(content="讲个笑话")]},
    stream_mode="messages"
):
    # chunk 是 (token, metadata) 元组
    if isinstance(chunk[0], str):
        print(chunk[0], end="", flush=True)
```

`"messages"` 模式特别为 LLM 流式输出设计。当节点内调用 LLM 时，每个 token 生成后都会立即推送，而不需要等待整个响应完成。这是实现聊天机器人"打字机效果"的关键。

### (e) stream mode "custom"：StreamWriter 自定义输出

```python
from langgraph.types import StreamWriter

def my_node(state: State, *, writer: StreamWriter) -> dict:
    for i in range(5):
        writer(f"进度: {i+1}/5")
    return {"result": "完成"}

for chunk in graph.stream({"count": 0}, stream_mode="custom"):
    print(chunk)
# 进度: 1/5
# 进度: 2/5
# ...
```

`"custom"` 模式允许节点通过 `writer` 回调函数在执行过程中推送自定义数据，与状态更新完全解耦。适合实时进度报告、中间结果展示等场景。

### (f) get_state / update_state：调试与手动状态操控

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thread-1"}}
graph.invoke({"count": 0}, config)

# 获取当前状态快照
snapshot = graph.get_state(config)
print(snapshot.values)    # {'count': 1}
print(snapshot.next)      # 下一步要执行的节点

# 手动修改状态
graph.update_state(config, {"count": 100}, as_node="increment")
snapshot = graph.get_state(config)
print(snapshot.values)    # {'count': 100}
```

`get_state` 返回 `StateSnapshot`，包含当前值、下一步节点、配置、元数据、待执行任务和中断信息。`update_state` 允许手动修改任何字段，如同某个节点产出了这个更新。这对调试、人机协作、错误恢复至关重要。

### (g) interrupt_before / interrupt：断点调试

```python
graph = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["increment"],  # 在 increment 节点执行前中断
)

config = {"configurable": {"thread_id": "t1"}}
result = graph.invoke({"count": 0}, config)
print(result)  # None — 被中断了

# 查看中断状态
snapshot = graph.get_state(config)
print(snapshot.next)  # ('increment',)

# 恢复执行
result = graph.invoke(None, config)
print(result)  # {'count': 1}
```

`interrupt_before` 和 `interrupt_after` 参数指定在哪些节点前后暂停执行。配合 `checkpointer`，图可以在任意断点暂停、检查状态、修改状态后恢复执行。这是实现人类审批（Human-in-the-Loop）的核心机制。

### (h) 不同 Checkpointer 选择与持久化模式

```python
# 内存中（开发/测试）
from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(checkpointer=InMemorySaver())

# SQLite（轻量持久化）
from langgraph.checkpoint.sqlite import SqliteSaver
graph = builder.compile(checkpointer=SqliteSaver.from_conn_string("db.sqlite"))

# PostgreSQL（生产环境）
from langgraph.checkpoint.postgres import AsyncPostgresSaver
# async with AsyncPostgresSaver.from_conn_string(DB_URI) as saver:
#     graph = builder.compile(checkpointer=saver)
```

Pregel 支持 `durability` 参数控制持久化时机：
- `"sync"`：每步执行前同步保存检查点，最安全但最慢
- `"async"`：异步保存，下一步与保存并行，平衡性能与安全
- `"exit"`：仅在退出时保存，最快但可能丢失中间状态

### (i) 子图执行与独立 Checkpointer

```python
def subgraph_node(state: SubState) -> dict:
    return {"result": "subgraph done"}

sub_builder = StateGraph(SubState)
sub_builder.add_node("process", subgraph_node)
sub_builder.add_edge(START, "process")
sub_builder.add_edge("process", END)
sub_graph = sub_builder.compile()  # 子图可以有自己的 checkpointer

# 父图中引用子图
builder.add_node("sub", sub_graph)
```

当子图有自己的 `checkpointer` 时，`get_state(subgraphs=True)` 可以递归获取子图状态。子图的检查点命名空间格式为 `"parent_node:<task_id>"`，在父图中通过 `checkpoint_ns` 定位。

## 4.3 Python 进阶

### Protocol 与 @abstractmethod

`PregelProtocol` 继承自 `Runnable[InputT, Any]` 和 `Generic[StateT, ContextT, InputT, OutputT]`，定义了所有核心方法的 `@abstractmethod`：

```python
class PregelProtocol(Runnable[InputT, Any], Generic[StateT, ContextT, InputT, OutputT]):
    @abstractmethod
    def get_state(self, config: RunnableConfig, *, subgraphs: bool = False) -> StateSnapshot: ...
    
    @abstractmethod
    def stream(self, input, config, *, ...) -> Iterator[...]: ...
```

4 个类型参数使 `PregelProtocol` 可以精确表达状态类型、上下文类型、输入类型和输出类型。

### @overload 在抽象方法上的应用

`PregelProtocol.stream` 有 3 个 `@overload` 签名：

1. `version: Literal["v2"]` → 返回 `Iterator[StreamPart]`
2. `version: Literal["v1"] = ...` → 返回 `Iterator[dict | Any]`
3. 实际实现签名 `version: Literal["v1", "v2"] = "v1"`

这样 IDE 可以根据 `version` 参数提供精确的返回类型提示。`invoke` 和 `ainvoke` 同理。

### Generic 与 4 个类型参数

`Pregel` 类声明为：

```python
class Pregel(PregelProtocol[StateT, ContextT, InputT, OutputT], Generic[StateT, ContextT, InputT, OutputT]):
```

- **StateT**：完整状态类型（如 `TypedDict`）
- **ContextT**：运行时上下文类型（如 `TypedDict`）
- **InputT**：图的输入类型（可能窄于 StateT）
- **OutputT**：图的输出类型（可能窄于 StateT）

这让 `CompiledStateGraph[MyState, None, MyInput, MyOutput]` 具有精确的输入输出类型。

### Self 类型

`Pregel.with_config` 和 `NodeBuilder` 的链式方法返回 `Self`：

```python
def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Self:
    return self.copy({"config": merge_configs(self.config, config, ...)})
```

`Self`（从 `typing_extensions` 导入）确保链式调用的返回类型是子类自身，而非硬编码的基类。

### __init__ 的关键字参数与 Unpack

`Pregel.__init__` 接受大量关键字参数（nodes, channels, input_channels, output_channels, checkpointer 等），全部为关键字形式（`*` 后声明）。`StateGraph.add_node` 签名中的 `**kwargs: Unpack[DeprecatedKwargs]` 允许接受已弃用的参数名而不触发类型错误。

### @property

`Pregel` 使用多个 `@property`：

```python
@property
def InputType(self) -> Any:
    if isinstance(self.input_channels, str):
        channel = self.channels[self.input_channels]
        if isinstance(channel, BaseChannel):
            return channel.UpdateType

@property
def stream_channels_list(self) -> Sequence[str]:
    stream_channels = self.stream_channels_asis
    return [stream_channels] if isinstance(stream_channels, str) else stream_channels
```

这些属性将内部表示转化为便于外部使用的形式，同时避免存储冗余数据。

### __init_subclass__

`CompiledStateGraph` 作为 `Pregel` 的子类，通过 `super().__init__(**kwargs)` 初始化，但额外存储 `builder` 和 `schema_to_mapper`。这种模式利用了 Python 的 `__init__` 链式调用，而非 `__init_subclass__`。

## 4.4 代码走读

### PregelProtocol（protocol.py）

`PregelProtocol` 是 Pregel 引擎的接口协议，定义了 20+ 个抽象方法。关键方法按功能分组：

**执行方法**：
- `invoke` / `ainvoke`：同步/异步一次性执行
- `stream` / `astream`：同步/异步流式执行

**状态方法**：
- `get_state` / `aget_state`：获取当前状态快照
- `get_state_history` / `aget_state_history`：获取历史状态
- `update_state` / `aupdate_state`：手动更新状态
- `bulk_update_state` / `abulk_update_state`：批量更新

**图方法**：
- `get_graph` / `aget_graph`：获取可绘制图结构
- `with_config`：创建带新配置的副本

每个方法都有 `version` 参数区分 `"v1"` 和 `"v2"` 输出格式。`v2` 格式返回 `StreamPart` 类型化字典，提供更好的类型安全。

`StreamProtocol` 是一个轻量辅助类，将流写入回调与流模式集合绑定：

```python
class StreamProtocol:
    __slots__ = ("modes", "__call__")
    modes: set[StreamMode]
    __call__: Callable[[StreamChunk], None]
```

### Pregel 类（main.py）

**字段**：`Pregel` 定义了完整的运行时配置：

```python
nodes: dict[str, PregelNode]        # 所有执行节点
channels: dict[str, BaseChannel | ManagedValueSpec]  # 所有通信通道
stream_mode: StreamMode = "values"  # 默认流式模式
output_channels: str | Sequence[str]  # 输出通道
input_channels: str | Sequence[str]    # 输入通道
checkpointer: Checkpointer = None   # 检查点保存器
store: BaseStore | None = None      # 持久化存储
cache: BaseCache | None = None      # 缓存
```

**__init__**：初始化时将 `NodeBuilder` 实例自动 `.build()` 为 `PregelNode`，注册 `TASKS` channel（用于 `Send` 消息），并默认调用 `validate()`。

**_defaults 方法**：统一处理 `stream`/`astream`/`invoke`/`ainvoke` 的默认参数：

```python
def _defaults(self, config, *, stream_mode, print_mode, output_keys, 
              interrupt_before, interrupt_after, durability):
    # 确保 stream_mode 是集合
    stream_modes = {stream_mode} if isinstance(stream_mode, str) else set(stream_mode)
    # 确定检查点器
    checkpointer = self._resolve_checkpointer(config)
    # 确定存储和缓存
    store = self.store; cache = self.cache
    return (stream_modes, output_keys, interrupt_before, interrupt_after, 
            checkpointer, store, cache, durability)
```

**stream 方法**：核心执行入口，流程如下：

1. 调用 `_defaults()` 解析参数
2. 创建 `SyncQueue` 作为流缓冲区
3. 通过 `get_callback_manager_for_config` 设置回调
4. 如果 `stream_mode` 含 `"messages"`，添加 `StreamMessagesHandler`
5. 如果含 `"custom"`，设置 `stream_writer` 函数
6. 构建 `Runtime` 对象，合并父级运行时
7. 创建 `SyncPregelLoop` 上下文管理器
8. 在循环中调用 `loop.tick()` 执行每一步
9. 通过 `stream.put()` 推送每步结果

**invoke 方法**：委托给 `stream`，收集最终结果：

```python
def invoke(self, input, config=None, *, stream_mode="values", ...):
    latest = None
    for chunk in self.stream(input, config, stream_mode=stream_mode, ...):
        if stream_mode == "values":
            latest = chunk  # 只保留最新值
        else:
            chunks.append(chunk)  # 收集所有 chunk
    if stream_mode == "values":
        return latest  # 返回最终状态
    return chunks  # 返回所有事件列表
```

### NodeBuilder（流式构建 API）

`NodeBuilder` 提供链式 API 构建 `PregelNode`：

```python
node1 = (
    NodeBuilder()
    .subscribe_only("a")     # 订阅单一 channel
    .do(lambda x: x + x)      # 执行函数
    .write_to("b")            # 写入 channel b
)
```

关键方法：
- `subscribe_only(channel)`：订阅单一 channel（值直接传入，不包字典）
- `subscribe_to(*channels)`：订阅多个 channel（值以字典传入）
- `read_from(*channels)`：只读取但不触发
- `do(runnable)`：设置执行函数
- `write_to(*channels)`：写入指定 channels
- `meta(*tags, **metadata)`：添加标签和元数据
- `add_retry_policies(*policies)`：添加重试策略
- `build()`：构建为 `PregelNode`

## 4.5 实现原理

### invoke → stream 委托模式

`invoke` 不直接实现执行逻辑，而是完全委托给 `stream`：

```
invoke(input) → stream(input, stream_mode="values") → 收集最后一个值
invoke(input, stream_mode="updates") → stream(input, stream_mode="updates") → 收集所有 chunk
```

这种设计的优势是：执行逻辑只维护一份（在 `stream` 中），`invoke` 只负责"如何收集结果"。

### stream → PregelLoop 创建

`stream` 的核心流程：

```
1. _defaults() → 解析参数、确定 checkpointer/store/cache
2. 创建 SyncQueue（流缓冲）
3. 配置回调管理器
4. 设置 StreamMessagesHandler（messages 模式）
5. 设置 stream_writer（custom 模式）
6. 构建 Runtime 对象
7. with SyncPregelLoop(...) as loop:
8.     创建 PregelRunner
9.     loop.tick() 循环执行
10.    每次 tick 通过 stream.put() 推送结果
```

`SyncPregelLoop` 是执行的核心容器，管理 channel 状态、任务调度、检查点保存。每一步 `tick()` 执行：
- **Plan**：从 `prepare_next_tasks()` 获取待执行任务
- **Execute**：通过 `PregelRunner` 并行执行任务
- **Update**：`apply_writes()` 将任务输出写入 channel
- **Checkpoint**：按 `durability` 设置保存检查点

### PregelLoop 的三个阶段

```
┌─────────┐     ┌──────────┐     ┌──────────┐
│  Plan   │────→│ Execute  │────→│  Update  │
│         │     │          │     │          │
│ 选择待  │     │ 并行执行 │     │ 写入     │
│ 执行任务│     │ 所有任务 │     │ channels │
└─────────┘     └──────────┘     └──────────┘
      ↑                                │
      └────────────────────────────────┘
              （循环直到无新任务）
```

**Plan 阶段**：`prepare_next_tasks()` 检查哪些 channel 有新数据，触发订阅这些 channel 的节点。

**Execute 阶段**：`PregelRunner` 使用线程池（同步）或 asyncio（异步）并行执行所有选中节点。

**Update 阶段**：`apply_writes()` 将每个节点的输出（dict 或 Command）解析为 `(channel_key, value)` 元组，写入对应 channel。reducer channel 会调用 reducer 函数合并新旧值。

### get_state 与 StateSnapshot

`get_state` 的流程：

1. 从 `config` 中获取 `checkpointer` 和 `thread_id`
2. 调用 `checkpointer.get_tuple(config)` 获取保存的检查点
3. 用 `channels_from_checkpoint()` 从检查点恢复 channel 状态
4. 调用 `prepare_next_tasks()` 计算下一步待执行任务
5. 组装 `StateSnapshot`：包含 values、next、config、metadata、created_at、parent_config、tasks、interrupts

如果 `subgraphs=True`，还会递归获取每个子图任务的状态。

### update_state 的实现

`update_state` 委托给 `bulk_update_state`：

```python
def update_state(self, config, values, as_node=None, task_id=None):
    return self.bulk_update_state(config, [[StateUpdate(values, as_node, task_id)]])
```

它将用户提供的值当作"来自 as_node 节点的输出"写入 channel，然后保存检查点。这实现了"时光机"功能——可以回退到任意历史状态并从那里继续执行。

## 4.6 动手实验

### 实验 1：创建最小 Pregel 实例

```python
from langgraph.channels import EphemeralValue
from langgraph.pregel import Pregel, NodeBuilder

# 用 NodeBuilder 直接构建
node1 = (
    NodeBuilder()
    .subscribe_only("input")
    .do(lambda x: x.upper())
    .write_to("output")
)

graph = Pregel(
    nodes={"upper": node1},
    channels={
        "input": EphemeralValue(str),
        "output": EphemeralValue(str),
    },
    input_channels="input",
    output_channels="output",
)

result = graph.invoke({"input": "hello"})
print(result)  # {'output': 'HELLO'}
```

### 实验 2：不同 stream mode 对比

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    value: int

def step_a(state: State) -> dict:
    return {"value": state["value"] + 1}

def step_b(state: State) -> dict:
    return {"value": state["value"] * 2}

builder = StateGraph(State)
builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)
builder.add_edge(START, "step_a")
builder.add_edge("step_a", "step_b")
builder.add_edge("step_b", END)

graph = builder.compile()

# values 模式
print("=== values ===")
for chunk in graph.stream({"value": 1}, stream_mode="values"):
    print(chunk)
# {'value': 1}    ← 初始状态
# {'value': 2}    ← step_a 后
# {'value': 4}    ← step_b 后

# updates 模式
print("=== updates ===")
for chunk in graph.stream({"value": 1}, stream_mode="updates"):
    print(chunk)
# {'step_a': {'value': 2}}
# {'step_b': {'value': 4}}
```

### 实验 3：get_state 与 update_state 调试

```python
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "debug-1"}}

# 第一次执行
result = graph.invoke({"value": 1}, config)
print(f"结果: {result}")  # {'value': 4}

# 获取状态快照
snapshot = graph.get_state(config)
print(f"当前值: {snapshot.values}")    # {'value': 4}
print(f"下一步: {snapshot.next}")       # ()
print(f"历史步数: {snapshot.metadata}")

# 查看状态历史
for state in graph.get_state_history(config):
    print(f"步骤 {state.metadata.get('step', '?')}: 值={state.values}")

# 手动修改状态
graph.update_state(config, {"value": 999}, as_node="step_a")
snapshot = graph.get_state(config)
print(f"修改后: {snapshot.values}")  # {'value': 999}

# 从修改后的状态继续执行
result = graph.invoke(None, config)
print(f"继续后: {result}")  # {'value': 1998}（999 * 2）
```

### 实验 4：interrupt 断点调试

```python
def human_approval(state: State) -> dict:
    return {"value": state["value"] + 100}

builder2 = StateGraph(State)
builder2.add_node("process", step_a)
builder2.add_node("approve", human_approval)
builder2.add_edge(START, "process")
builder2.add_edge("process", "approve")
builder2.add_edge("approve", END)

graph2 = builder2.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["approve"],  # 在 approve 前中断
)

config2 = {"configurable": {"thread_id": "approval"}}

# 第一次调用——会在 approve 前中断
result = graph2.invoke({"value": 1}, config2)
print(f"中断后结果: {result}")  # None 或不完整的值

# 检查中断状态
snapshot = graph2.get_state(config2)
print(f"下一步: {snapshot.next}")  # ('approve',)
print(f"中断信息: {snapshot.interrupts}")

# 人工审核后继续执行
result = graph2.invoke(None, config2)
print(f"继续后结果: {result}")  # {'value': 102}（(1+1) + 100）
```

### 实验 5：多 stream mode 组合

```python
# 同时获取 values 和 updates
for chunk in graph.stream({"value": 1}, stream_mode=["values", "updates"]):
    mode, data = chunk
    if mode == "values":
        print(f"[完整状态] {data}")
    elif mode == "updates":
        print(f"[增量更新] {data}")
```

`stream_mode` 参数接受字符串或列表。传入列表时，每个 chunk 是 `(namespace, mode, data)` 三元组（有子图时）或 `(mode, data)` 二元组，允许同时观察多种输出格式。

### 实验 6：NodeBuilder 流式 API 与 BinaryOperatorAggregate

```python
from langgraph.channels import EphemeralValue, BinaryOperatorAggregate

def reducer(current, update):
    return current + " | " + update if current else update

node1 = (
    NodeBuilder()
    .subscribe_only("input")
    .do(lambda x: x + x)
    .write_to("intermediate", "result")
)

node2 = (
    NodeBuilder()
    .subscribe_to("intermediate")
    .do(lambda x: x["intermediate"] + x["intermediate"])
    .write_to("result")
)

graph = Pregel(
    nodes={"double": node1, "quad": node2},
    channels={
        "input": EphemeralValue(str),
        "intermediate": EphemeralValue(str),
        "result": BinaryOperatorAggregate(str, operator=reducer),
    },
    input_channels="input",
    output_channels="result",
)

result = graph.invoke({"input": "foo"})
# result = "foofoo | foofoofoofoo"
# 两个节点都写入了 result，reducer 将它们合并
```

这个实验展示了 Pregel 底层 API 如何直接操作 channel 和 reducer，不依赖 `StateGraph` 的声明式抽象。