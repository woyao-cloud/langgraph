# 第四章 Pregel 引擎——入口与协议

> 源码路径：`langgraph/pregel/protocol.py`、`langgraph/pregel/main.py`

## 1 Python 进阶

### 1.1 Protocol 与抽象基类的融合

`protocol.py` 定义了 `PregelProtocol`，它同时继承了 `Runnable`（LangChain 的 ABC）和使用了 `@abstractmethod`：

```python
class PregelProtocol(Runnable[InputT, Any], Generic[StateT, ContextT, InputT, OutputT]):
    @abstractmethod
    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Self: ...

    @abstractmethod
    def stream(self, input, config=None, *, context=None, stream_mode=None,
               interrupt_before=None, interrupt_after=None, subgraphs=False,
               version: Literal["v1", "v2"] = "v1") -> Iterator[...]: ...
```

这体现了一种**混合策略**：`Runnable` 是 LangChain 通过 ABC 实现的基类，提供 `invoke`、`batch` 等方法的默认实现；`PregelProtocol` 在此基础上用 `@abstractmethod` 声明子类**必须**重写的方法（如 `stream`、`get_state`、`update_state`）。

为什么不用纯 `Protocol`？因为 `Pregel` 需要继承 `Runnable` 的大量具体实现（如 `batch`、`ainvoke` 的默认委托逻辑），而 `Protocol` 不支持继承具体方法。

### 1.2 四参数 Generic 与类型安全

```python
class PregelProtocol(Runnable[InputT, Any], Generic[StateT, ContextT, InputT, OutputT]):
class Pregel(PregelProtocol[StateT, ContextT, InputT, OutputT],
             Generic[StateT, ContextT, InputT, OutputT]):
```

四个类型参数：
- `StateT`：图的完整状态类型（如 `TypedDict` 子类）
- `ContextT`：运行时上下文类型（`Runtime[ContextT]` 中的 `ContextT`）
- `InputT`：图的输入类型
- `OutputT`：图的输出类型

这让 `Pregel` 的类型签名非常精确。例如 `CompiledStateGraph[MyState, None, MyInput, MyOutput]` 明确区分了完整状态、输入和输出。

### 1.3 `@overload` 与抽象方法

`PregelProtocol` 对 `stream`、`astream`、`invoke`、`ainvoke` 各定义了三个重载：

```python
@overload
@abstractmethod
def stream(self, input, config=None, *, ..., version: Literal["v2"]) -> Iterator[StreamPart]: ...

@overload
@abstractmethod
def stream(self, input, config=None, *, ..., version: Literal["v1"] = ...) -> Iterator[dict]: ...

@abstractmethod
def stream(self, input, config=None, *, ..., version: Literal["v1", "v2"] = "v1") -> Iterator: ...
```

这种模式让类型检查器能根据 `version` 参数的值推断返回类型：
- `version="v2"` → `Iterator[StreamPart[StateT, OutputT]]`（结构化流输出）
- `version="v1"` → `Iterator[dict[str, Any] | Any]`（传统字典流输出）
- 默认 → 两者兼容

### 1.4 `Self` 返回类型与流畅接口

```python
@abstractmethod
def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Self: ...
```

`Self`（来自 `typing_extensions`）表示"返回与 `self` 相同类型的实例"。这对继承链至关重要——`Pregel.with_config()` 返回 `Pregel`，`CompiledStateGraph.with_config()` 返回 `CompiledStateGraph`，而非 `Runnable`。

### 1.5 `Unpack` 与 `**kwargs` 类型约束

```python
def __init__(self, ..., **deprecated_kwargs: Unpack[DeprecatedKwargs]):
```

`Unpack` 来自 `typing_extensions`，用于将 `TypedDict` 展开为 `**kwargs` 的类型约束。`DeprecatedKwargs` 定义了弃用参数的键和类型，让 IDE 能在 `**kwargs` 中提供补全和类型检查。

### 1.6 `__slots__` 与 `NodeBuilder`

```python
class NodeBuilder:
    __slots__ = ("_channels", "_triggers", "_tags", "_metadata", "_writes", "_bound",
                 "_retry_policy", "_cache_policy")
```

`__slots__` 阻止 `__dict__` 的创建，减少内存占用约 40-50%，同时防止意外属性赋值。`NodeBuilder` 还使用了类级别的类型注解（`_channels: str | list[str]`），这些注解在 `__slots__` 模式下不会创建实例属性，只作为类型提示。

### 1.7 `@property` 与计算属性

```python
@property
def InputType(self) -> Any:
    if isinstance(self.input_channels, str):
        channel = self.channels[self.input_channels]
        if isinstance(channel, BaseChannel):
            return channel.UpdateType
```

`InputType` 和 `OutputType` 是计算属性，根据 `input_channels`/`output_channels` 是 `str` 还是 `Sequence[str]` 动态返回类型信息。当通道是单个字符串时，返回该通道的更新类型；否则返回 `None`（由 `get_input_schema`/`get_output_schema` 动态构建 Pydantic 模型）。

### 1.8 `__init_subclass__` 与类级别钩子

Pregel 使用了 Pydantic 的 `create_model` 在运行时动态创建输入/输出模型：

```python
def get_input_schema(self, config=None):
    config = merge_configs(self.config, config)
    if isinstance(self.input_channels, str):
        return super().get_input_schema(config)
    else:
        return create_model(
            self.get_name("Input"),
            field_definitions={
                k: (c.UpdateType, None)
                for k in self.input_channels
                if (c := self.channels[k]) and isinstance(c, BaseChannel)
            },
        )
```

`create_model` 是 Pydantic 的 `model_create` 封装，在运行时根据通道定义动态创建 Pydantic 模型类。这避免了为每种图定义手动编写输入/输出模型。

## 2 代码走读

### 2.1 PregelProtocol：抽象接口

`PregelProtocol` 定义了 20+ 个抽象方法，分为几类：

**状态查询方法**：
```python
@abstractmethod
def get_state(self, config, *, subgraphs=False) -> StateSnapshot: ...

@abstractmethod
def get_state_history(self, config, *, filter=None, before=None, limit=None) -> Iterator[StateSnapshot]: ...
```

**状态更新方法**：
```python
@abstractmethod
def update_state(self, config, values, as_node=None) -> RunnableConfig: ...

@abstractmethod
def bulk_update_state(self, config, updates) -> RunnableConfig: ...
```

**执行方法**（带 `@overload`）：
```python
@overload
def stream(self, input, config=None, *, ..., version: Literal["v2"]) -> Iterator[StreamPart]: ...

@overload
def stream(self, input, config=None, *, ..., version: Literal["v1"] = ...) -> Iterator[dict]: ...

@abstractmethod
def stream(self, input, config=None, *, ..., version: Literal["v1", "v2"] = "v1") -> Iterator: ...
```

**图可视化方法**：
```python
@abstractmethod
def get_graph(self, config=None, *, xray=False) -> DrawableGraph: ...
```

注意 `stream` 和 `astream` 各有 2 个 `@overload` + 1 个实现签名，`invoke` 和 `ainvoke` 也是如此。这为版本迁移提供了类型安全的过渡路径。

### 2.2 StreamProtocol：轻量级流协议

```python
class StreamProtocol:
    __slots__ = ("modes", "__call__")
    modes: set[StreamMode]
    __call__: Callable[[Self, StreamChunk], None]

    def __init__(self, __call__, modes):
        self.__call__ = cast(Callable[[Self, StreamChunk], None], __call__)
        self.modes = modes
```

`StreamProtocol` 是一个精巧的**函数对象**：它既是可调用的（通过 `__call__` 属性），又携带了 `modes`（流模式集合）。在 Pregel 循环中，`StreamProtocol` 实例被传递给 PregelLoop，用于将输出推送到队列。

`StreamChunk` 的类型是 `tuple[tuple[str, ...], str, Any]`——三元组：(命名空间, 模式, 数据)。

### 2.3 NodeBuilder：流式构造器

```python
class NodeBuilder:
    __slots__ = ("_channels", "_triggers", "_tags", "_metadata",
                 "_writes", "_bound", "_retry_policy", "_cache_policy")

    def subscribe_only(self, channel: str) -> Self:
        """订阅单个通道——节点输入为该通道的原始值"""
        if not self._channels:
            self._channels = channel  # 注意：str 而非 list
        else:
            raise ValueError("Cannot subscribe to single channels when other channels are already subscribed to")
        self._triggers.append(channel)
        return self

    def subscribe_to(self, *channels: str, read: bool = True) -> Self:
        """订阅多个通道——节点输入为 {channel: value} 字典"""
        if isinstance(self._channels, str):
            raise ValueError("Cannot subscribe to channels when subscribed to a single channel")
        if read:
            if not self._channels:
                self._channels = list(channels)
            else:
                self._channels.extend(channels)
        self._triggers.extend(channels)
        return self

    def do(self, node: RunnableLike) -> Self:
        """设置节点函数"""
        if self._bound is not DEFAULT_BOUND:
            self._bound = RunnableSeq(self._bound, coerce_to_runnable(node, name=None, trace=True))
        else:
            self._bound = coerce_to_runnable(node, name=None, trace=True)
        return self

    def write_to(self, *channels: str | ChannelWriteEntry, **kwargs) -> Self:
        """添加写入目标"""
        self._writes.extend(
            ChannelWriteEntry(c) if isinstance(c, str) else c for c in channels
        )
        self._writes.extend(
            ChannelWriteEntry(k, mapper=v) if callable(v) else ChannelWriteEntry(k, value=v)
            for k, v in kwargs.items()
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

`NodeBuilder` 的核心设计决策：

1. **`_channels` 可以是 `str` 或 `list[str]`**——当 `subscribe_only` 时设为 `str`（单值输入），当 `subscribe_to` 时设为 `list`（字典输入）。PregelNode 根据 `channels` 的类型决定是否将输入包装为字典。

2. **`DEFAULT_BOUND` 哨兵值**——用于区分"未设置节点函数"和"已设置"。`do()` 首次调用时替换哨兵，后续调用时用 `RunnableSeq` 链接。

3. **`write_to` 支持 `ChannelWriteEntry`、字符串、关键字参数**——`ChannelWriteEntry("x")` 是直通写入，`ChannelWriteEntry("x", mapper=fn)` 是映射写入，`ChannelWriteEntry("x", value=5)` 是常量写入。

### 2.4 Pregel 类：构造函数

```python
class Pregel(PregelProtocol[StateT, ContextT, InputT, OutputT], Generic[StateT, ContextT, InputT, OutputT]):
    nodes: dict[str, PregelNode]
    channels: dict[str, BaseChannel | ManagedValueSpec]
    stream_mode: StreamMode = "values"
    stream_eager: bool = False
    output_channels: str | Sequence[str]
    stream_channels: str | Sequence[str] | None = None
    interrupt_after_nodes: All | Sequence[str]
    interrupt_before_nodes: All | Sequence[str]
    input_channels: str | Sequence[str]
    step_timeout: float | None = None
    debug: bool
    checkpointer: Checkpointer = None
    store: BaseStore | None = None
    cache: BaseCache | None = None
    retry_policy: Sequence[RetryPolicy] = ()
    cache_policy: CachePolicy | None = None
    context_schema: type[ContextT] | None = None
    config: RunnableConfig | None = None
    name: str = "LangGraph"
    trigger_to_nodes: Mapping[str, Sequence[str]]
```

构造函数：

```python
def __init__(self, *, nodes, channels=None, auto_validate=True, stream_mode="values",
             stream_eager=False, output_channels, stream_channels=None,
             interrupt_after_nodes=(), interrupt_before_nodes=(),
             input_channels, step_timeout=None, debug=None,
             checkpointer=None, store=None, cache=None,
             retry_policy=(), cache_policy=None, context_schema=None,
             config=None, trigger_to_nodes=None, name="LangGraph",
             **deprecated_kwargs):
```

关键初始化步骤：

```python
# 1. 处理弃用参数
if (config_type := deprecated_kwargs.get("config_type", MISSING)) is not MISSING:
    if context_schema is None:
        context_schema = cast(type[ContextT], config_type)

# 2. 验证 checkpointer
checkpointer = ensure_valid_checkpointer(checkpointer)

# 3. 构建 NodeBuilder
self.nodes = {k: v.build() if isinstance(v, NodeBuilder) else v for k, v in nodes.items()}

# 4. 保留 TASKS 通道
self.channels = channels or {}
if TASKS in self.channels and not isinstance(self.channels[TASKS], Topic):
    raise ValueError(f"Channel '{TASKS}' is reserved.")
self.channels[TASKS] = Topic(Send, accumulate=False)

# 5. 规范化 retry_policy
self.retry_policy = (retry_policy,) if isinstance(retry_policy, RetryPolicy) else retry_policy

# 6. 初始化 serde 白名单
self._serde_allowlist: set[tuple[str, ...]] | None = None

# 7. 自动验证
if auto_validate:
    self.validate()
```

注意步骤 3：`NodeBuilder` 在传入构造函数时才调用 `.build()` 转换为 `PregelNode`。这是一种延迟构建模式——用户可以先组装节点配置，再统一编译。

步骤 4 的 `TASKS` 通道是 `Send` 对象的传输通道，默认使用 `Topic(Send, accumulate=False)`（即不累积，每步清空）。

### 2.5 Pregel.validate()：图验证

```python
def validate(self) -> Self:
    validate_graph(
        self.nodes,
        {k: v for k, v in self.channels.items() if isinstance(v, BaseChannel)},
        {k: v for k, v in self.channels.items() if not isinstance(v, BaseChannel)},
        self.input_channels,
        self.output_channels,
        self.stream_channels,
        self.interrupt_after_nodes,
        self.interrupt_before_nodes,
    )
    self.trigger_to_nodes = _trigger_to_nodes(self.nodes)
    return self
```

`validate_graph` 检查：
- 所有节点引用的通道都存在
- 输入通道是有效通道
- 输出通道是有效通道
- 中断节点名称有效
- 所有触发器对应的通道存在

`_trigger_to_nodes` 建立从通道名到节点名的反向映射，用于确定"当通道 X 更新时，应触发哪些节点"。

### 2.6 Pregel.stream()：BSP 执行循环

`stream` 方法是 Pregel 引擎的核心执行入口。其主循环遵循 **Bulk Synchronous Parallel (BSP)** 模型：

```python
def stream(self, input, config=None, *, context=None, stream_mode=None,
           print_mode=(), output_keys=None, interrupt_before=None,
           interrupt_after=None, durability=None, subgraphs=False,
           debug=None, version="v1", **kwargs):
    # ... 参数处理和默认值设置 ...

    stream = SyncQueue()
    config = ensure_config(self.config, config)

    # 设置回调管理器
    callback_manager = get_callback_manager_for_config(config)
    run_manager = callback_manager.on_chain_start(None, input, ...)

    # 解析默认值
    (stream_modes, output_keys, interrupt_before_, interrupt_after_,
     checkpointer, store, cache, durability_) = self._defaults(config, ...)

    # 设置消息流处理器
    if "messages" in stream_modes:
        run_manager.inheritable_handlers.append(StreamMessagesHandler(...))

    # 设置自定义流处理器
    if "custom" in stream_modes:
        def stream_writer(c): stream.put((..., "custom", c))

    # 创建 PregelLoop
    with SyncPregelLoop(input, stream=StreamProtocol(stream.put, stream_modes),
                        config=config, ...) as loop:
        runner = PregelRunner(...)

        # BSP 主循环
        while loop.tick():
            for task in loop.match_cached_writes():
                loop.output_writes(task.id, task.writes, cached=True)
            for _ in runner.tick(
                [t for t in loop.tasks.values() if not t.writes],
                timeout=self.step_timeout,
                get_waiter=get_waiter,
                schedule_task=loop.accept_push,
            ):
                yield from _output(stream_mode, print_mode, subgraphs,
                                   stream.get, queue.Empty, version, ...)

            loop.after_tick()
            if durability_ == "sync":
                loop._put_checkpoint_fut.result()

        # 最终输出
        yield from _output(stream_mode, print_mode, subgraphs, stream.get, ...)

        # 递归深度检查
        if loop.status == "out_of_steps":
            raise GraphRecursionError(...)
```

每一步 `tick()` 执行：
1. **Plan**：根据上一步的通道更新确定要执行的节点
2. **Execute**：并行执行所有就绪节点（通过 `PregelRunner.tick`）
3. **Update**：`after_tick()` 将节点输出写入通道

### 2.7 Pregel.invoke()：委托到 stream

```python
def invoke(self, input, config=None, *, context=None, stream_mode="values",
           print_mode=(), output_keys=None, interrupt_before=None,
           interrupt_after=None, durability=None, version="v1", **kwargs):
    output_keys = output_keys if output_keys is not None else self.output_channels
    latest = None
    chunks = []
    interrupts = []

    if version == "v2":
        for chunk in self.stream(input, config, context=context, ...):
            if stream_mode == "values":
                latest = chunk["data"]
                if chunk_ints := chunk.get("interrupts", ()):
                    interrupts.extend(chunk_ints)
            else:
                chunks.append(chunk)
    else:
        for chunk in self.stream(input, config, context=context, ...):
            # v1: 从 ["updates", "values"] 双流中提取
            ...
    if stream_mode == "values":
        if version == "v2":
            return GraphOutput(value=latest, interrupts=tuple(interrupts))
        if interrupts:
            return {**latest, INTERRUPT: interrupts}
        return latest
    else:
        return chunks
```

`invoke` 的设计哲学是**单一实现路径**：它不重复 stream 的逻辑，而是通过调用 `self.stream()` 收集所有输出，然后返回最终结果。`version="v2"` 返回结构化的 `GraphOutput`，包含 `value` 和 `interrupts`；`version="v1"` 返回原始字典或值。

### 2.8 copy() 与 with_config()：不可变配置

```python
def copy(self, update=None):
    attrs = {k: v for k, v in self.__dict__.items() if k != "__orig_class__"}
    attrs.update(update or {})
    return self.__class__(**attrs)

def with_config(self, config=None, **kwargs):
    return self.copy({"config": merge_configs(self.config, config, cast(RunnableConfig, kwargs))})
```

`copy()` 创建一个新的 `Pregel` 实例，复制所有属性（排除 `__orig_class__`——它是 `Generic` 的内部属性）。`with_config()` 通过 `copy()` 实现，合并新旧配置。这是一种**不可变模式**——修改配置不会改变原始对象，而是返回新实例。

## 3 运行原理

### 3.1 Pregel 执行流程

```
用户调用 graph.invoke(input)
    │
    ├──> Pregel.invoke()
    │       │
    │       ├──> Pregel.stream() 创建 SyncPregelLoop
    │       │       │
    │       │       ├──> loop.tick() —— BSP 步骤 1：Plan
    │       │       │       ├──> 检查哪些通道有更新
    │       │       │       ├──> 根据 trigger_to_nodes 确定要执行的节点
    │       │       │       └──> 创建 Task 对象
    │       │       │
    │       │       ├──> runner.tick(tasks) —— BSP 步骤 2：Execute
    │       │       │       ├──> 对每个 Task 调用 node.bound.invoke(state)
    │       │       │       ├──> 节点输出写入 stream
    │       │       │       └──> yield 输出事件
    │       │       │
    │       │       └──> loop.after_tick() —— BSP 步骤 3：Update
    │       │               ├──> 将写入应用到通道
    │       │               ├──> 检查点保存（如果有）
    │       │               └──> 准备下一步
    │       │
    │       └──> 循环直到没有更多更新
    │
    └──> 返回最终状态
```

### 3.2 通道更新可见性规则

Pregel 遵循严格的 BSP 语义：**步骤 N 的通道更新只在步骤 N+1 可见**。这意味着：

1. 节点 A 在步骤 1 执行，输出写入通道
2. 步骤 1 结束后，通道更新才对其他节点可见
3. 节点 B 在步骤 2 读取节点 A 的输出

这避免了数据竞争，确保确定性执行。

### 3.3 NodeBuilder 到 PregelNode 的转换

```
NodeBuilder                        PregelNode
─────────────                       ──────────
subscribe_only("input")     →       channels="input" (str，单值)
subscribe_to("a", "b")      →       channels=["a", "b"] (list，字典)
do(fn)                       →       bound=RunnableLambda(fn)
write_to("output")           →       writers=[ChannelWrite([ChannelWriteEntry("output")])]
write_to("x", mapper=fn)    →       writers=[ChannelWrite([ChannelWriteEntry("x", mapper=fn)])])
build()                      →       PregelNode(channels=..., triggers=..., writers=..., bound=...)
```

### 3.4 stream_mode 对输出的影响

| stream_mode | 输出内容 |
|-------------|---------|
| `"values"` | 每步后的完整状态 |
| `"updates"` | 每步中每个节点的输出增量 |
| `"custom"` | 通过 `StreamWriter` 发出的自定义数据 |
| `"messages"` | LLM 消息的 token 级流 |
| `"debug"` | 调试信息 |

可以传入列表如 `["values", "updates"]` 同时获取多种模式，此时输出为 `(namespace, mode, data)` 三元组。

## 4 实现细节

### 4.1 构造参数的默认值与验证

`Pregel.__init__` 的参数几乎全部是关键字参数（`*` 之后的参数），这防止了位置参数混淆：

```python
def __init__(self, *, nodes, channels=None, auto_validate=True, ...):
```

`auto_validate=True` 默认在构造时验证图。`CompiledStateGraph` 传入 `auto_validate=False` 延迟验证，因为图还在构建中。

### 4.2 checkpointer 的验证与处理

```python
checkpointer = ensure_valid_checkpointer(checkpointer)
```

`ensure_valid_checkpointer` 将 `True` 转换为继承父图 checkpointer 的哨兵值，将 `False` 转换为 `None`，确保 `checkpointer` 字段只有 `None` 或 `BaseCheckpointSaver` 实例。

### 4.3 `_serde_allowlist` 机制

```python
self._serde_allowlist: set[tuple[str, ...]] | None = None
```

当启用 strict msgpack 序列化时（`_serde.STRICT_MSGPACK_ENABLED`），Pregel 在编译时构建一个类型白名单，只允许序列化/反序列化白名单中的类型。这防止了任意代码执行漏洞：

```python
if _serde.STRICT_MSGPACK_ENABLED:
    schema_types = [self.state_schema, self.input_schema, self.output_schema]
    if self.context_schema is not None:
        schema_types.append(self.context_schema)
    for node in self.nodes.values():
        schema_types.append(node.input_schema)
    serde_allowlist = _serde.build_serde_allowlist(schemas=schema_types, channels=self.channels)
    checkpointer = _serde.apply_checkpointer_allowlist(checkpointer, serde_allowlist)
```

### 4.4 `_defaults` 方法

`_defaults` 方法解析所有可从配置继承的参数：

```python
def _defaults(self, config, stream_mode=None, print_mode=(), output_keys=None,
              interrupt_before=None, interrupt_after=None, durability=None):
    stream_modes = stream_mode if isinstance(stream_mode, list) else [stream_mode or self.stream_mode]
    output_keys = output_keys if output_keys is not None else self.output_channels
    interrupt_before_ = interrupt_before if interrupt_before is not None else self.interrupt_before_nodes
    interrupt_after_ = interrupt_after if interrupt_after is not None else self.interrupt_after_nodes
    checkpointer = ...
    store = ...
    cache = ...
    durability_ = ...
    return (stream_modes, output_keys, interrupt_before_, interrupt_after_,
            checkpointer, store, cache, durability_)
```

这种设计让 `stream`/`invoke` 的调用参数优先于 `Pregel` 实例的默认值，而实例默认值又优先于 `RunnableConfig` 中的值。

### 4.5 v1 与 v2 流版本

Pregel 的 `stream` 方法支持两个版本的输出格式：

- **v1（默认）**：`yield dict | Any`，传统格式
- **v2**：`yield StreamPart[StateT, OutputT]`，结构化格式，包含 `data`、`interrupts` 等字段

v2 的优势：
1. 类型安全：`StreamPart` 是 `TypedDict`，IDE 可以提示字段
2. 中断信息：`interrupts` 字段直接包含中断信息，无需从 `updates` 中提取
3. 状态映射：v2 会使用 `_state_mapper` 和 `_output_mapper` 将原始字典转换为 Pydantic 模型或 TypedDict

### 4.6 TASKS 通道的特殊性

```python
self.channels[TASKS] = Topic(Send, accumulate=False)
```

`TASKS` 通道使用 `Topic(Send, accumulate=False)`：
- `Topic` 是一种通道类型，支持多次写入、一次性读取
- `accumulate=False` 表示每步清空，不跨步累积
- `Send` 是值类型，用于向其他节点发送数据包

这个通道在 `add_conditional_edges` 的 `_control_branch` 中被写入：当节点返回 `Send` 对象时，写入 `(TASKS, send_obj)` 到此通道。

## 5 动手实验

### 实验 1：手动构建 Pregel 实例

```python
from langgraph.channels import EphemeralValue, LastValue
from langgraph.pregel import Pregel, NodeBuilder

# 构建节点
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
    .write_to("final")
)

# 构建 Pregel
app = Pregel(
    nodes={"node1": node1, "node2": node2},
    channels={
        "input": EphemeralValue(str),
        "output": LastValue(str),  # 注意：非 EphemeralValue，因为 node2 需要读取
        "final": EphemeralValue(str),
    },
    input_channels=["input"],
    output_channels=["final"],
)

result = app.invoke({"input": "hello"})
print(result)
# {'final': 'hellohello!'}
```

### 实验 2：使用 ChannelWriteEntry 控制写入

```python
from langgraph.pregel._write import ChannelWriteEntry

# 带映射函数的写入
node = (
    NodeBuilder()
    .subscribe_only("x")
    .do(lambda x: x)
    .write_to(ChannelWriteEntry("doubled", mapper=lambda x: x * 2))
    .write_to(ChannelWriteEntry("constant", value=42))
)

app = Pregel(
    nodes={"node": node},
    channels={
        "x": EphemeralValue(int),
        "doubled": LastValue(int),
        "constant": LastValue(int),
    },
    input_channels=["x"],
    output_channels=["doubled", "constant"],
)

result = app.invoke({"x": 5})
print(result)
# {'doubled': 10, 'constant': 42}
```

### 实验 3：检查编译后图的内部结构

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

def add(a: list, b: int) -> list:
    return a + [b]

class State(TypedDict):
    counter: Annotated[list, add]
    value: int

builder = StateGraph(State)

def increment(state: State):
    return {"counter": state["value"], "value": state["value"] + 1}

builder.add_node("inc", increment)
builder.add_edge(START, "inc")
builder.add_edge("inc", END)

graph = builder.compile()

# 检查节点
for name, node in graph.nodes.items():
    print(f"Node: {name}")
    print(f"  triggers: {node.triggers}")
    print(f"  channels: {node.channels}")
    print(f"  bound: {type(node.bound).__name__}")
    for w in node.writers:
        print(f"  writer: {type(w).__name__}")
        for entry in w.entries:
            print(f"    entry: {entry}")

# Node: __start__
#   triggers: ['__start__']
#   channels: __start__
#   bound: RunnableLambda
#   writer: ChannelWrite
#     entry: ChannelWriteTupleEntry(mapper=<function attach_node.<locals>._get_updates>)
#     entry: ChannelWriteTupleEntry(mapper=<function _control_branch>)

# Node: inc
#   triggers: ['branch:to:inc']
#   channels: ['counter', 'value']
#   bound: RunnableLambda
#   writer: ChannelWrite
#     entry: ChannelWriteTupleEntry(mapper=<function attach_node.<locals>._get_updates>)
#     entry: ChannelWriteTupleEntry(mapper=<function _control_branch>)

# 检查通道
for name, channel in graph.channels.items():
    if hasattr(channel, 'key'):
        print(f"Channel: {name} = {type(channel).__name__}(key={channel.key})")
    else:
        print(f"Channel: {name} = {type(channel).__name__}")
```

### 实验 4：多步 Pregel 与循环

```python
from langgraph.channels import EphemeralValue
from langgraph.pregel import Pregel, NodeBuilder

# 创建一个计数器：每次调用将值加 1，直到达到 5
step_node = (
    NodeBuilder()
    .subscribe_only("counter")
    .do(lambda x: x + 1 if x < 5 else None)
    .write_to("counter")
)

# 注意：EphemeralValue 不保留上一步的值用于同一节点读取
# 要实现循环，需要使用 LastValue 或 BinaryOperatorAggregate
from langgraph.channels.binop import BinaryOperatorAggregate

counter_node = (
    NodeBuilder()
    .subscribe_only("counter")
    .do(lambda x: None if x >= 5 else x + 1)
    .write_to("counter")
)

app = Pregel(
    nodes={"counter": counter_node},
    channels={
        "counter": EphemeralValue(int),  # EphemeralValue 每步后清空
    },
    input_channels=["counter"],
    output_channels=["counter"],
)

# Pregel 的循环语义：当节点写入自己订阅的通道时，
# 会触发新一轮执行
result = list(app.stream({"counter": 0}, stream_mode="updates"))
for step in result:
    print(step)
```

### 实验 5：StreamProtocol 的使用

```python
from langgraph.pregel.protocol import StreamProtocol

# 创建一个简单的流处理器
output_log = []

def stream_handler(self, chunk):
    namespace, mode, data = chunk
    output_log.append((mode, data))

stream = StreamProtocol(stream_handler, modes={"values", "updates"})

# 模拟推流
stream(("graph:0", "values", {"x": 1}))
stream(("graph:0", "updates", {"inc": {"counter": 1}}))

print(output_log)
# [('values', {'x': 1}), ('updates', {'inc': {'counter': 1}})]
print(stream.modes)
# {'values', 'updates'}
```

### 实验 6：PregelProtocol 的类型安全

```python
from langgraph.pregel.protocol import PregelProtocol
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class MyState(TypedDict):
    value: int

builder = StateGraph(MyState)

def node_a(state: MyState) -> dict:
    return {"value": state["value"] + 1}

builder.add_node("a", node_a)
builder.add_edge(START, "a")
builder.add_edge("a", END)

graph = builder.compile()

# PregelProtocol 的抽象方法
print(type(graph).mro())
# CompiledStateGraph -> Pregel -> PregelProtocol -> Runnable -> ABC -> ...

# 调用 with_config（返回 Self，即 CompiledStateGraph）
configured = graph.with_config({"recursion_limit": 100})
print(type(configured).__name__)  # CompiledStateGraph

# 调用 get_graph（返回 DrawableGraph）
drawn = graph.get_graph()
print(type(drawn).__name__)  # Graph
```