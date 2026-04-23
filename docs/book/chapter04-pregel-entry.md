# 第四章 Pregel 执行引擎（上）— pregel/main.py, pregel/protocol.py

> 从协议定义到引擎核心，理解 LangGraph 的运行骨架

本章进入 LangGraph 的核心引擎——Pregel。它遵循 Bulk Synchronous Parallel（BSP）模型，将图的执行组织为一系列"超级步"：每个超级步分为 Plan（选择待执行的 actor）、Execute（并行执行）、Update（更新通道值）三个阶段，循环直到没有 actor 被选中或达到最大步数。

---

## 4.1 执行协议 — `pregel/protocol.py`

### Java 桥梁

Java 开发者习惯用 `abstract class` 或 `interface` 定义行为契约。Python 的 `Protocol` 提供了一种更灵活的方式——**结构化子类型（structural subtyping）**。不需要 `implements` 声明，只要方法签名匹配，类型检查器就认为满足协议。

### Python 概念速查

| Python 概念 | Java 对应 | 说明 |
|---|---|---|
| `Protocol`（结构化子类型） | `interface`（名义子类型） | Python Protocol 不需要显式声明继承，签名匹配即可；Java interface 必须 `implements` |
| `@abstractmethod` | `abstract` 方法 | 必须由子类实现，否则实例化时抛 `TypeError` |
| `@overload` + `@abstractmethod` | 接口方法重载 | Protocol 中可声明多个 overload 签名 |

### 代码走读

**PregelProtocol** 是整个 LangGraph 执行层的核心协议：

```python
class PregelProtocol(Runnable[InputT, Any], Generic[StateT, ContextT, InputT, OutputT]):
    @abstractmethod
    def with_config(self, config, **kwargs) -> Self: ...

    @abstractmethod
    def get_graph(self, config, *, xray=False) -> DrawableGraph: ...

    @abstractmethod
    async def aget_graph(self, config, *, xray=False) -> DrawableGraph: ...

    @abstractmethod
    def get_state(self, config, *, subgraphs=False) -> StateSnapshot: ...

    @abstractmethod
    async def aget_state(self, config, *, subgraphs=False) -> StateSnapshot: ...

    @abstractmethod
    def get_state_history(self, config, ...) -> Iterator[StateSnapshot]: ...

    @abstractmethod
    async def aget_state_history(self, config, ...) -> AsyncIterator[StateSnapshot]: ...

    @abstractmethod
    def update_state(self, config, values, as_node=None) -> RunnableConfig: ...

    @abstractmethod
    async def aupdate_state(self, config, values, as_node=None) -> RunnableConfig: ...
```

注意 `PregelProtocol` 同时继承了 `Runnable[InputT, Any]`——这是 LangChain 的核心抽象，类似于 Java 的 `Supplier<T>` + `Consumer<T>` 组合体。`Runnable` 提供 `invoke`、`stream`、`ainvoke`、`astream` 等执行方法。

**Protocol vs Java Interface 的关键差异**：

在 Java 中，`class Pregel implements PregelProtocol` 必须显式声明。在 Python 中，只要 `Pregel` 类拥有 `PregelProtocol` 要求的所有方法签名，类型检查器（mypy/pyright）就认为 `Pregel` 满足 `PregelProtocol`——这叫"鸭子类型"的静态版本。但 `PregelProtocol` 使用了 `@abstractmethod`，意味着**名义子类**如果不实现这些方法，实例化时会抛 `TypeError`。所以这里的 `Protocol` 实际上混合了结构化子类型（对类型检查器）和名义子类型（对运行时）的语义。

**stream/ainvoke 的 @overload**：`PregelProtocol` 为 `stream`、`astream`、`invoke`、`ainvoke` 各声明了 3 个 overload 签名。以 `stream` 为例：

```python
@overload
@abstractmethod
def stream(self, input, config=None, *, version: Literal["v2"]) -> Iterator[StreamPart]: ...

@overload
@abstractmethod
def stream(self, input, config=None, *, version: Literal["v1"] = ...) -> Iterator[dict]: ...

@abstractmethod
def stream(self, input, config=None, *, version: Literal["v1","v2"] = "v1") -> Iterator: ...
```

Java 开发者注意：这里的 `Literal["v1", "v2"]` 是 Python 3.8+ 的字面量类型——它约束参数只能是特定字符串值。结合 `@overload`，类型检查器能根据 `version` 的值推断返回类型。Java 中这需要用不同的方法名（如 `streamV1` / `streamV2`）或泛型技巧来实现。

**StreamProtocol 与 StreamChunk**：

```python
StreamChunk = tuple[tuple[str, ...], str, Any]

class StreamProtocol:
    __slots__ = ("modes", "__call__")
    modes: set[StreamMode]
    __call__: Callable[[Self, StreamChunk], None]

    def __init__(self, __call__, modes):
        self.__call__ = cast(Callable[[Self, StreamChunk], None], __call__)
        self.modes = modes
```

`StreamChunk` 是一个三元组：(命名空间路径, 流模式, 数据)。`StreamProtocol` 用 `__slots__` 限制属性为 `modes` 和 `__call__`——这等价于 Java 中用 `final` 字段声明的紧凑对象，避免动态 `__dict__` 的开销。

`__call__` 被赋值为一个可调用对象，使得 `StreamProtocol` 实例本身可以被"调用"——这是 Python 的"可调用协议"，类似 Java 的 `@FunctionalInterface` + `call()` 方法。

### 运行原理

`PregelProtocol` 是所有 LangGraph 可执行图的抽象基类。`Pregel` 类和 `CompiledStateGraph` 类都满足此协议。它定义了图的执行接口（invoke/stream）、状态查询接口（get_state/update_state）和可视化接口（get_graph）。上层 SDK 和 CLI 只依赖 `PregelProtocol`，不依赖具体实现。

---

## 4.2 Pregel 引擎 — `pregel/main.py`

### Java 桥梁

Java 开发者遇到"大量构造参数"时，通常用 Builder 模式：

```java
Pregel app = Pregel.builder()
    .nodes(nodes)
    .channels(channels)
    .inputChannels("input")
    .outputChannels("output")
    .checkpointer(saver)
    .build();
```

Python 没有 Builder 模式的传统，因为 Python 支持**关键字参数**——所有参数都可以用 `name=value` 传入，天然解决了参数多、顺序难记的问题。

### Python 概念速查

| Python 概念 | Java 对应 | 说明 |
|---|---|---|
| 全部关键字参数构造 | Builder 模式 | Python 用 `*` 强制关键字参数，替代 Builder |
| `async def` / `await` | `CompletableFuture` / `async` (Project Loom) | Python 的协程是语言级特性，比 Java 的 CompletableFuture 更直观 |
| `contextmanager` | `try-with-resources` / AutoCloseable | Python 用 `with` 语句管理资源生命周期 |
| `__init_subclass__` | 无直接对应 | 子类创建时自动调用的钩子方法 |

### 代码走读

**Pregel 类声明**：

```python
class Pregel(
    PregelProtocol[StateT, ContextT, InputT, OutputT],
    Generic[StateT, ContextT, InputT, OutputT],
):
```

`Pregel` 同时继承 `PregelProtocol` 和声明 `Generic`——前者提供行为约束和抽象方法，后者提供类型参数化。Java 中这等价于 `class Pregel<S,C,I,O> implements PregelProtocol<S,C,I,O>`。

**核心字段**：

```python
nodes: dict[str, PregelNode]           # 节点名 → PregelNode
channels: dict[str, BaseChannel | ManagedValueSpec]  # 通道名 → 通道实例
stream_mode: StreamMode = "values"     # 默认流模式
output_channels: str | Sequence[str]   # 输出通道
input_channels: str | Sequence[str]    # 输入通道
step_timeout: float | None = None      # 超级步超时
checkpointer: Checkpointer = None      # 状态持久化
store: BaseStore | None = None         # 长期记忆存储
cache: BaseCache | None = None         # 节点结果缓存
retry_policy: Sequence[RetryPolicy] = ()  # 重试策略
```

**__init__ 构造器**：

```python
def __init__(
    self,
    *,
    nodes: dict[str, PregelNode | NodeBuilder],
    channels: dict[str, BaseChannel | ManagedValueSpec] | None,
    auto_validate: bool = True,
    stream_mode: StreamMode = "values",
    output_channels: str | Sequence[str],
    input_channels: str | Sequence[str],
    step_timeout: float | None = None,
    checkpointer: Checkpointer = None,
    store: BaseStore | None = None,
    cache: BaseCache | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] = (),
    context_schema: type[ContextT] | None = None,
    name: str = "LangGraph",
    **deprecated_kwargs: Unpack[DeprecatedKwargs],
) -> None:
```

注意 `*` 之后的所有参数都必须用关键字传入——这等价于 Java Builder 的"只提供 setter 不提供位置构造参数"。`nodes` 参数接受 `PregelNode | NodeBuilder`，构造器内自动调用 `v.build()` 将 NodeBuilder 转为 PregelNode：

```python
self.nodes = {k: v.build() if isinstance(v, NodeBuilder) else v for k, v in nodes.items()}
```

**自动注册 TASKS 通道**：

```python
if TASKS in self.channels and not isinstance(self.channels[TASKS], Topic):
    raise ValueError(f"Channel '{TASKS}' is reserved")
else:
    self.channels[TASKS] = Topic(Send, accumulate=False)
```

`TASKS` 是一个保留通道名，用于 `Send` 对象的传递。Pregel 自动创建一个 `Topic` 类型的通道——类似于消息总线中的 topic 发布/订阅。

### 运行原理

Pregel 构造器做了三件事：
1. 将 NodeBuilder 转为 PregelNode
2. 注册保留通道（如 TASKS）
3. 如果 `auto_validate=True`，调用 `validate()` 检查图的完整性

`validate()` 检查：所有 input/output 通道存在、trigger-to-nodes 映射正确、无悬空引用等。

---

## 4.3 NodeBuilder — 流式节点构建器

### Java 桥梁

Java 的 Builder 模式通过 `return this` 实现链式调用。NodeBuilder 完全遵循这一模式，但用 Python 的类型系统（`Self` 返回类型）提供更好的类型推断。

### 代码走读

```python
class NodeBuilder:
    __slots__ = (
        "_channels", "_triggers", "_tags", "_metadata",
        "_writes", "_bound", "_retry_policy", "_cache_policy",
    )

    def subscribe_only(self, channel: str) -> Self:
        """订阅单个通道，节点接收该通道的裸值"""
        if not self._channels:
            self._channels = channel
        else:
            raise ValueError("Cannot subscribe to single channels when others are already subscribed")
        self._triggers.append(channel)
        return self

    def subscribe_to(self, *channels: str, read: bool = True) -> Self:
        """订阅多个通道，节点接收 dict[channel_name, value]"""
        if isinstance(self._channels, str):
            raise ValueError("Cannot subscribe to channels when subscribed to a single channel")
        if read:
            self._channels.extend(channels)
        self._triggers.extend(channels)
        return self

    def do(self, node: RunnableLike) -> Self:
        """绑定执行函数"""
        if self._bound is not DEFAULT_BOUND:
            self._bound = RunnableSeq(self._bound, coerce_to_runnable(node))
        else:
            self._bound = coerce_to_runnable(node)
        return self

    def write_to(self, *channels: str | ChannelWriteEntry, **kwargs) -> Self:
        """声明写入目标通道"""
        self._writes.extend(
            ChannelWriteEntry(c) if isinstance(c, str) else c for c in channels
        )
        self._writes.extend(
            ChannelWriteEntry(k, mapper=v) if callable(v)
            else ChannelWriteEntry(k, value=v) for k, v in kwargs.items()
        )
        return self

    def build(self) -> PregelNode:
        """构建 PregelNode"""
        return PregelNode(
            channels=self._channels,
            triggers=self._triggers,
            tags=self._tags,
            metadata=self._metadata,
            writers=[ChannelWrite(self._writes)],
            bound=self._bound,
            retry_policy=self._retry_policy,
            cache_policy=self._cache_policy,
        )
```

**subscribe_only vs subscribe_to**：这是一个重要的区分：
- `subscribe_only("a")`：节点只订阅一个通道，接收到的是该通道的**裸值**（如字符串）
- `subscribe_to("a", "b")`：节点订阅多个通道，接收到的是 `dict["a": val_a, "b": val_b]`

Java 开发者可以类比：`subscribe_only` 像接收 `Message<String>`，`subscribe_to` 像接收 `Map<String, Object>`。

**do() 方法支持链式组合**：如果多次调用 `do()`，执行函数会组成 `RunnableSeq`（顺序执行链），而非覆盖。

**write_to() 的双模式**：支持位置参数（通道名）和关键字参数（通道名 + 值/映射器）：

```python
node = (
    NodeBuilder()
    .subscribe_only("input")
    .do(lambda x: x.upper())
    .write_to("output", uppercase=lambda x: x.upper())  # 两种写法
    .build()
)
```

### 运行原理

NodeBuilder 是一个"延迟构建"模式——它收集配置但不立即创建 PregelNode，直到 `build()` 被调用。`__slots__` 声明确保实例没有 `__dict__`，属性访问更快且防止意外添加属性——类似 Java 中将字段声明为 `private final` 且不提供 setter。

---

## 4.4 Pregel 的执行方法

### Java 桥梁

Java 中执行图通常用 `CompletableFuture<Output> invoke(Input)` 或响应式流的 `Flux<Output> stream(Input)`。Python 的 `async def` / `await` 让异步代码看起来像同步代码——这是 Java 开发者最需要适应的范式转变。

### Python 概念速查

| Python 概念 | Java 对应 | 说明 |
|---|---|---|
| `Iterator` (生成器) | `Stream` / `Flux` | Python 用 `yield` 返回惰性序列 |
| `AsyncIterator` (异步生成器) | `Flux` / `Publisher` | `async for` + `yield` 组合 |
| `contextlib` | `try-with-resources` | Python 用 `with` / `async with` 管理上下文 |
| `SyncQueue` / `AsyncQueue` | `BlockingQueue` / 队列 | 用于流式输出的内部通信 |

### 代码走读

**invoke() 方法**：

```python
def invoke(self, input, config=None, *, stream_mode="values", version="v1", ...) -> dict | Any:
    output_keys = output_keys if output_keys is not None else self.output_channels
    latest: dict[str, Any] | Any = None
    chunks: list = []
    interrupts: list[Interrupt] = []

    if version == "v2":
        for chunk in self.stream(input, config, ...):
            if stream_mode == "values":
                latest = chunk["data"]
                if chunk_ints := chunk.get("interrupts", ()):
                    interrupts.extend(chunk_ints)
            else:
                chunks.append(chunk)
    else:
        for chunk in self.stream(input, config, ...):
            # v1: 收集 values 和 interrupts
            ...
```

关键发现：**invoke 内部调用 stream**。`invoke` 并非独立实现，它只是 `stream` 的"聚合消费者"——遍历流中的所有事件，收集最终值。这与 Java 中 `flux.blockLast()` 的思路一致。

**stream() 方法**：

```python
def stream(self, input, config=None, *, stream_mode=None, ...) -> Iterator[dict | Any]:
    stream = SyncQueue()  # 内部队列

    config = ensure_config(self.config, config)
    callback_manager = get_callback_manager_for_config(config)
    run_manager = callback_manager.on_chain_start(...)

    # 配置流模式
    stream_modes, output_keys, interrupt_before_, interrupt_after_, checkpointer, store, cache, durability_ = self._defaults(config, ...)

    # 设置消息流
    if "messages" in stream_modes:
        run_manager.inheritable_handlers.append(StreamMessagesHandler(stream.put, ...))

    # 设置自定义流
    if "custom" in stream_modes:
        def stream_writer(c):
            stream.put((namespace, "custom", c))
```

`stream()` 方法的核心是用一个内部 `SyncQueue` 作为事件总线——各组件（节点执行器、消息处理器、自定义写入器）将事件放入队列，外层 `for` 循环从队列中取出事件并 `yield` 给调用者。

**委托给 PregelLoop**：在 stream 方法的深处，真正的执行逻辑委托给 `SyncPregelLoop` 或 `AsyncPregelLoop`：

```python
# (简化)
loop = SyncPregelLoop(
    input=input,
    config=config,
    nodes=self.nodes,
    channels=channels,
    managed=managed,
    checkpointer=checkpointer,
    store=store,
    ...)
```

PregelLoop 是实际执行超级步的类——下一章将详细解析。Pregel 本身更像"调度层"，负责配置、回调管理和流式输出格式化。

### 运行原理

Pregel 的执行流程如下：

1. **配置阶段**：`stream()` / `invoke()` 合并用户配置和默认配置，设置回调和流处理器
2. **委托给 PregelLoop**：创建 `SyncPregelLoop`（同步）或 `AsyncPregelLoop`（异步）
3. **超级步循环**（在 PregelLoop 内）：
   - **Plan**：根据通道更新选择待执行的节点
   - **Execute**：并行执行选中的节点
   - **Update**：将节点的写入应用到通道
4. **流式输出**：每个超级步的结果通过 `SyncQueue` 传回 stream() 方法，yield 给用户
5. **终止**：当没有节点被选中（所有通道无新更新），或达到中断条件

```
用户代码              Pregel               PregelLoop
--------              ------               ----------
invoke(input)  -->    stream(input)  -->   创建loop
                      <-- yield chunk  <--  超级步1结果
                      <-- yield chunk  <--  超级步2结果
                      <-- yield chunk  <--  无更多任务
返回 latest    <--    遍历结束
```

---

## 4.5 动手实验

### 实验 1：用 NodeBuilder 直接构建 Pregel 图

```python
from langgraph.channels import EphemeralValue, LastValue
from langgraph.pregel import Pregel, NodeBuilder

node1 = (
    NodeBuilder()
    .subscribe_only("input")
    .do(lambda x: x + x)
    .write_to("output")
)

node2 = (
    NodeBuilder()
    .subscribe_to("output")
    .do(lambda x: x["output"] + "!")
    .write_to("result")
)

app = Pregel(
    nodes={"doubler": node1, "exclaimer": node2},
    channels={
        "input": EphemeralValue(str),
        "output": LastValue(str),
        "result": EphemeralValue(str),
    },
    input_channels="input",
    output_channels="result",
)

result = app.invoke({"input": "hello"})
print(result)  # {"result": "hellohello!"}
```

### 实验 2：对比 Protocol 的结构化子类型

```python
from langgraph.pregel.protocol import PregelProtocol

# Pregel 满足 PregelProtocol —— 不是因为显式声明 implements
# 而是因为它拥有 PregelProtocol 要求的所有方法
def check_protocol(obj):
    # 运行时 Python 不强制 Protocol 检查
    # 但类型检查器（mypy）会验证
    print("Has invoke:", hasattr(obj, 'invoke'))
    print("Has stream:", hasattr(obj, 'stream'))
    print("Has get_state:", hasattr(obj, 'get_state'))

check_protocol(app)
# True, True, True
```

### 实验 3：观察 invoke 对 stream 的委托

```python
from langgraph.channels import EphemeralValue
from langgraph.pregel import Pregel, NodeBuilder

step_count = 0

def counting_node(x):
    global step_count
    step_count += 1
    return x + 1

node = (
    NodeBuilder()
    .subscribe_only("value")
    .do(counting_node)
    .write_to("value")  # 写回自身，形成循环
)

app = Pregel(
    nodes={"counter": node},
    channels={"value": EphemeralValue(int)},
    input_channels="value",
    output_channels="value",
)

# invoke 内部会遍历 stream 的所有事件
result = app.invoke({"value": 1})
print(f"Steps: {step_count}, Result: {result}")
```

### 实验 4：探索 Pregel 的通道与节点结构

```python
from langgraph.channels import EphemeralValue
from langgraph.pregel import Pregel, NodeBuilder

n1 = NodeBuilder().subscribe_only("a").do(lambda x: x.upper()).write_to("b").build()
n2 = NodeBuilder().subscribe_only("b").do(lambda x: x + "!").write_to("c").build()

app = Pregel(
    nodes={"upper": n1, "exclaim": n2},
    channels={"a": EphemeralValue(str), "b": EphemeralValue(str), "c": EphemeralValue(str)},
    input_channels="a",
    output_channels="c",
)

# 检查节点结构
for name, node in app.nodes.items():
    print(f"Node '{name}':")
    print(f"  channels (subscriptions): {node.channels}")
    print(f"  triggers: {node.triggers}")
    print(f"  writers: {len(node.writers)} writer(s)")

# 检查通道
for name, channel in app.channels.items():
    print(f"Channel '{name}': {type(channel).__name__}")
```

---

## 小结

本章揭示了 Pregel 引擎的核心架构：

1. **PregelProtocol 定义执行契约**：用 Protocol + @abstractmethod 描述所有可执行图必须实现的接口，混合了结构化子类型和名义子类型的语义
2. **Pregel 是调度层**：它持有节点和通道的映射，管理配置和回调，但将实际执行委托给 PregelLoop
3. **NodeBuilder 是 Pregel 级别的节点构建器**：直接操作通道订阅和写入，比 StateGraph 级别的 add_node 更底层
4. **invoke 是 stream 的聚合**：所有执行入口最终都通过 stream 机制，invoke 只是消费流的最终值
5. **关键字参数替代 Builder**：Python 用 `*` 强制关键字参数和默认值，替代了 Java 的 Builder 模式

下一章将深入 PregelLoop，解析超级步的 Plan-Execute-Update 三阶段循环、通道读写机制和检查点恢复逻辑。