# 第三章 图构建——从声明到编译

> 源码路径：`langgraph/graph/_node.py`、`_branch.py`、`state.py`、`message.py`

## 1 Python 进阶

### 1.1 Protocol 与结构化子类型（Structural Subtyping）

`_node.py` 定义了 9 个 `Protocol` 变体来描述节点的签名：

```python
class _Node(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra) -> Any: ...

class _NodeWithConfig(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, config: RunnableConfig) -> Any: ...

class _NodeWithWriter(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, writer: StreamWriter) -> Any: ...
```

这里 `Protocol` 与 `ABC`（名义子类型）有本质区别：

| 特性 | `Protocol` | `ABC` |
|------|-----------|-------|
| 子类型判定 | 结构化：只要有对应方法签名即可 | 名义化：必须显式继承或注册 |
| 运行时检查 | `isinstance(obj, Protocol)` 可选（需 `@runtime_checkable`） | `isinstance(obj, ABC)` 始终可用 |
| 典型用途 | 描述回调签名、函数接口 | 描述共享行为的基类 |

LangGraph 选择 `Protocol` 的原因很明确：节点可以是普通函数、带 `config` 的函数、带 `writer` 的函数——这些函数**不可能**都继承同一个 ABC，但它们的签名可以**结构化匹配**不同的 Protocol 变体。

### 1.2 TypeAlias 与联合类型

`StateNode` 用 `TypeAlias` 定义了所有合法的节点类型：

```python
StateNode: TypeAlias = (
    _Node[NodeInputT]
    | _NodeWithConfig[NodeInputT]
    | _NodeWithWriter[NodeInputT]
    | ...  # 共 9 个 Protocol 变体
    | Runnable[NodeInputT, Any]
)
```

`TypeAlias` 让类型检查器知道这是一个类型别名，而不是一个变量。`NodeInputT_contra` 是逆变类型变量（`contravariant`），确保类型安全——一个接受更宽泛输入类型的函数可以替代接受更窄输入类型的函数。

### 1.3 `@overload`：类型检查器专用签名

`state.py` 中 `add_node` 有 4 个 `@overload` 装饰的重载签名加 1 个实际实现：

```python
@overload
def add_node(self, node: StateNode[...], *, ..., input_schema: None = None, ...) -> Self: ...
@overload
def add_node(self, node: StateNode[...], *, ..., input_schema: type[NodeInputT], ...) -> Self: ...
@overload
def add_node(self, node: str, action: StateNode[...], *, ..., input_schema: None = None, ...) -> Self: ...
@overload
def add_node(self, node: str | StateNode[...], action: ..., *, ..., input_schema: type[NodeInputT], ...) -> Self: ...
def add_node(self, node: str | StateNode[...], action: ..., *, ..., input_schema: ... | None = None, ...) -> Self:
    # 实际实现
```

`@overload` 签名**只对类型检查器可见**，运行时不执行。它们让 IDE 和 mypy 理解：
- 传入 `input_schema=None` 时返回类型与不传时一致
- 传入 `input_schema=SomeType` 时返回类型参数改变
- 第一个参数为 `str` 还是 `StateNode` 决定了第二个参数的含义

### 1.4 `get_type_hints()` 与运行时反射

`_branch.py` 中的 `from_path` 方法使用 `get_type_hints()` 在运行时解析函数的返回类型：

```python
if rtn_type := get_type_hints(func).get("return"):
    if get_origin(rtn_type) is Literal:
        path_map_ = {name: name for name in get_args(rtn_type)}
```

`get_type_hints()` 解析 `__annotations__` 并解析字符串前向引用，比直接读取 `func.__annotations__` 更可靠。`get_origin()` 和 `get_args()` 则用于解包 `Literal["a", "b"]` 为 `Literal` 和 `("a", "b")`。

### 1.5 `functools.partial`、`defaultdict`、`cast()`

- `message.py` 使用 `functools.partial` 实现偏函数应用：当 `add_messages` 只传入一个参数时，返回一个绑定该参数的新函数。
- `state.py` 中 `self.branches: defaultdict[str, dict[str, BranchSpec]]` 使用 `defaultdict(dict)` 自动初始化嵌套字典。
- `cast()` 在多处用于类型窄化，如 `cast(FunctionType, callable_)` 告诉 mypy "我知道这个值的运行时类型"，但 `cast` 在运行时是空操作。

### 1.6 `inspect.signature` 与 `__name__`/`__qualname__`

`add_node` 中通过 `inspect.signature` 获取函数的第一个参数名：

```python
first_parameter_name = next(
    iter(inspect.signature(cast(FunctionType, action)).parameters.keys())
)
```

`_get_node_name` 函数使用 `__name__` 属性推断节点名：

```python
def _get_node_name(node: StateNode[Any, ContextT]) -> str:
    try:
        return getattr(node, "__name__", node.__class__.__name__)
    except AttributeError:
        raise TypeError(f"Unsupported node type: {type(node)}")
```

## 2 代码走读

### 2.1 `_node.py`：9 种 Protocol 与 StateNodeSpec

`_node.py` 定义了 LangGraph 中"节点"这一概念的全部合法形态。从简单到复杂：

| Protocol | 签名 | 典型用例 |
|---------|------|---------|
| `_Node` | `(state) -> Any` | 最简节点 |
| `_NodeWithConfig` | `(state, config) -> Any` | 需要回调配置 |
| `_NodeWithWriter` | `(state, *, writer) -> Any` | 流式输出 |
| `_NodeWithStore` | `(state, *, store) -> Any` | 需要存储 |
| `_NodeWithWriterStore` | `(state, *, writer, store) -> Any` | 流式+存储 |
| `_NodeWithConfigWriter` | `(state, *, config, writer) -> Any` | 配置+流式 |
| `_NodeWithConfigStore` | `(state, *, config, store) -> Any` | 配置+存储 |
| `_NodeWithConfigWriterStore` | `(state, *, config, writer, store) -> Any` | 全参数 |
| `_NodeWithRuntime` | `(state, *, runtime) -> Any` | 运行时上下文 |

注意 `writer`、`store`、`config` 全部是**关键字参数**（`*` 后面），这保证了用户只需传入关心的参数。

`StateNodeSpec` 用 `@dataclass(slots=True)` 定义节点的完整规格：

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

`slots=True` 使用 `__slots__` 替代 `__dict__`，减少内存占用并略微提升属性访问速度。`ends` 字段在运行时由 `add_node` 从返回类型注解中推断——如果函数声明返回 `Command[Literal["a", "b"]]`，`ends` 会被设置为 `("a", "b")`。

### 2.2 `_branch.py`：条件分支的编译与路由

`BranchSpec` 是一个 `NamedTuple`：

```python
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]
    ends: dict[Hashable, str] | None
    input_schema: type[Any] | None = None
```

`NamedTuple` 相比 `dataclass` 更轻量，且自动不可变，适合作为编译期数据结构。

`from_path` 类方法是分支的"编译"入口：

```python
@classmethod
def from_path(cls, path, path_map, infer_schema=False):
    # 1. 若 path_map 是 dict，直接使用
    # 2. 若 path_map 是 list，转为 {name: name}
    # 3. 若 path_map 为 None，尝试从返回类型的 Literal 推断
    if isinstance(path_map, dict):
        path_map_ = path_map.copy()
    elif isinstance(path_map, list):
        path_map_ = {name: name for name in path_map}
    else:
        # 反射推断
        func = path.func or path.afunc
        if rtn_type := get_type_hints(func).get("return"):
            if get_origin(rtn_type) is Literal:
                path_map_ = {name: name for name in get_args(rtn_type)}
```

这是一个精妙的**类型驱动设计**：如果路由函数声明了返回类型 `Literal["route_a", "route_b"]`，LangGraph 可以在编译时就知道所有可能的目标节点，无需用户手动指定 `path_map`。

`_route`/`_aroute` 是同步/异步的成对方法，模式完全一致：

```python
def _route(self, input, config, *, reader, writer):
    if reader:
        value = reader(config)
        # 字典状态合并：将节点的输出键合并到分支输入
        if isinstance(value, dict) and isinstance(input, dict) and self.input_schema is None:
            value = {**input, **value}
    else:
        value = input
    result = self.path.invoke(value, config)
    return self._finish(writer, input, result, config)
```

注意第 4-5 行的条件合并：只有当输入模式和值都是 `dict`，且分支没有显式的 `input_schema` 时，才将节点输出与状态合并。这避免了不必要的状态泄露。

`_finish` 方法处理路由结果：

```python
def _finish(self, writer, input, result, config):
    if not isinstance(result, (list, tuple)):
        result = [result]
    if self.ends:
        destinations = [r if isinstance(r, Send) else self.ends[r] for r in result]
    else:
        destinations = cast(Sequence[Send | str], result)
    # 验证：不允许路由到 None 或 START
    if any(dest is None or dest == START for dest in destinations):
        raise ValueError("Branch did not return a valid destination")
    entries = writer(destinations, False)
    if not entries:
        return input
    # 检查是否有 PASSTHROUGH 写入
    need_passthrough = any(
        isinstance(e, ChannelWriteEntry) and e.value is PASSTHROUGH
        for e in entries
    )
    if need_passthrough:
        return ChannelWrite(entries)  # 延迟到 Pregel 循环中写入
    else:
        ChannelWrite.do_write(config, entries)  # 立即写入
        return input
```

`PASSTHROUGH` 是一个哨兵值，表示"写入值来自节点输出，需要延迟到 tick 结束时才能确定"。如果写入的目标通道是确定的（不含 PASSTHROUGH），则立即写入，避免额外的数据传递开销。

### 2.3 `state.py`：StateGraph 构建器模式

#### 2.3.1 `__init__` 与 `_add_schema`

```python
class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    def __init__(self, state_schema, context_schema=None, *, input_schema=None, output_schema=None, **kwargs):
        self.nodes = {}
        self.edges = set()
        self.branches = defaultdict(dict)
        self.schemas = {}
        self.channels = {}
        self.managed = {}
        self.compiled = False
        self.waiting_edges = set()

        self.state_schema = state_schema
        self.input_schema = cast(type[InputT], input_schema or state_schema)
        self.output_schema = cast(type[OutputT], output_schema or state_schema)

        self._add_schema(self.state_schema)
        self._add_schema(self.input_schema, allow_managed=False)
        self._add_schema(self.output_schema, allow_managed=False)
```

`_add_schema` 的核心是 `_get_channels` 函数，它根据类型注解推断通道类型：

```python
def _get_channels(schema):
    if not hasattr(schema, "__annotations__"):
        return ({"__root__": _get_channel("__root__", schema, allow_managed=False)}, {}, {})
    type_hints = get_type_hints(schema, include_extras=True)
    all_keys = {name: _get_channel(name, typ) for name, typ in type_hints.items() if name != "__slots__"}
    return (
        {k: v for k, v in all_keys.items() if isinstance(v, BaseChannel)},
        {k: v for k, v in all_keys.items() if is_managed_value(v)},
        type_hints,
    )
```

`include_extras=True` 是关键——它让 `get_type_hints` 保留 `Annotated` 标记，从而能够解析 `Annotated[list, add_messages]` 这样的类型。

`_get_channel` 的类型推断逻辑：

```python
def _get_channel(name, annotation, *, allow_managed=True):
    # 1. 剥离 Required/NotRequired 包装
    if hasattr(annotation, "__origin__") and annotation.__origin__ in (Required, NotRequired):
        annotation = annotation.__args__[0]
    # 2. 检查是否为 ManagedValue
    if manager := _is_field_managed_value(name, annotation):
        return manager if allow_managed else error
    # 3. 检查是否为显式 BaseChannel 注解
    elif channel := _is_field_channel(annotation):
        channel.key = name
        return channel
    # 4. 检查是否为 Annotated + reducer 函数
    elif channel := _is_field_binop(annotation):
        channel.key = name
        return channel
    # 5. 默认：LastValue 通道
    fallback = LastValue(annotation)
    fallback.key = name
    return fallback
```

通道类型推断的三条路径：

1. **`Annotated[int, EphemeralValue]`** → `_is_field_channel` 找到 `BaseChannel` 实例/类，创建 `EphemeralValue(int)`
2. **`Annotated[list, add_messages]`** → `_is_field_binop` 找到末尾的可调用 reducer，创建 `BinaryOperatorAggregate(list, reducer)`
3. **`x: int`**（无 Annotated）→ 回退到 `LastValue(int)`

#### 2.3.2 `add_node` 的四重重载与运行时逻辑

`add_node` 的实际实现（去掉文档字符串后的核心逻辑）：

```python
def add_node(self, node, action=None, *, defer=False, metadata=None,
             input_schema=None, retry_policy=None, cache_policy=None,
             destinations=None, **kwargs):
    # 1. 处理弃用参数
    if (retry := kwargs.get("retry", MISSING)) is not MISSING:
        retry_policy = retry
    # 2. 推断节点名称
    if not isinstance(node, str):
        action = node
        node = getattr(action, "__name__", action.__class__.__name__)
    # 3. 验证：不允许重名节点，不允许 END/START
    if node in self.nodes:
        raise ValueError(f"Node `{node}` already present.")
    if node == END or node == START:
        raise ValueError(f"Node `{node}` is reserved.")
    # 4. 推断 input_schema 和 ends（从返回类型）
    inferred_input_schema = None
    ends = EMPTY_SEQ
    try:
        if (isfunction(action) or ismethod(action) or ...) and (hints := get_type_hints(...)):
            if input_schema is None:
                first_param = next(iter(signature(action).parameters.keys()))
                if input_hint := hints.get(first_param):
                    if isinstance(input_hint, type) and get_type_hints(input_hint):
                        inferred_input_schema = input_hint
            if rtn := hints.get("return"):
                # 处理 Union 中的 Command 类型
                rtn_origin = get_origin(rtn)
                if rtn_origin is Union:
                    for arg in get_args(rtn):
                        if get_origin(arg) is Command:
                            rtn = arg; break
                # 从 Command[Literal[...]] 提取 ends
                if rtn_origin is Command and (rargs := get_args(rtn)) and ...:
                    ends = vals
    except (NameError, TypeError, StopIteration):
        pass
    # 5. 创建 StateNodeSpec
    self.nodes[node] = StateNodeSpec(
        coerce_to_runnable(action, name=node, trace=False),
        metadata, input_schema or inferred_input_schema or self.state_schema,
        retry_policy, cache_policy, ends, defer,
    )
    # 6. 若有显式 input_schema，注册到 schema
    if input_schema is not None:
        self._add_schema(input_schema)
    return self
```

关键步骤解析：
- 步骤 4 的类型推断在 `try/except` 中执行，因为 `get_type_hints()` 可能因前向引用失败
- `Union` 处理支持 `Command | None` 的返回类型，提取其中的 `Command` 部分
- `coerce_to_runnable` 将普通函数包装为 `Runnable`，统一执行接口

#### 2.3.3 `add_edge`：创建 EphemeralValue 通道

```python
def add_edge(self, start_key, end_key):
    if isinstance(start_key, str):
        if start_key == END: raise ValueError("END cannot be a start node")
        if end_key == START: raise ValueError("START cannot be an end node")
        self.edges.add((start_key, end_key))
        return self
    # 多起点边（等待边）
    for start in start_key:
        if start == END: raise ValueError("END cannot be a start node")
        if start not in self.nodes: raise ValueError(f"Need to add_node `{start}` first")
    if end_key != END and end_key not in self.nodes:
        raise ValueError(f"Need to add_node `{end_key}` first")
    self.waiting_edges.add((tuple(start_key), end_key))
    return self
```

注意：`add_edge` 只记录边的拓扑结构，**不立即创建通道**。真正的通道创建发生在 `compile()` 的 `attach_edge` 中。

#### 2.3.4 `add_conditional_edges`：创建 BranchSpec

```python
def add_conditional_edges(self, source, path, path_map=None):
    path = coerce_to_runnable(path, name=None, trace=True)
    name = path.name or "condition"
    if name in self.branches[source]:
        raise ValueError(f"Branch with name `{path.name}` already exists for node `{source}`")
    self.branches[source][name] = BranchSpec.from_path(path, path_map, True)
    if schema := self.branches[source][name].input_schema:
        self._add_schema(schema)
    return self
```

`infer_schema=True` 传给了 `from_path`，表示需要推断分支的输入模式。推断过程调用 `_get_branch_path_input_schema`，提取路径函数第一个参数的类型注解。

#### 2.3.5 `compile()`：从构建器到 CompiledStateGraph

这是最关键的方法，将声明式图定义转换为可执行的 Pregel 实例：

```python
def compile(self, checkpointer=None, *, cache=None, store=None,
            interrupt_before=None, interrupt_after=None, debug=False, name=None):
    checkpointer = ensure_valid_checkpointer(checkpointer)
    # 1. 构建 serde 白名单
    serde_allowlist = _serde.build_serde_allowlist(...) if _serde.STRICT_MSGPACK_ENABLED else None
    # 2. 验证图
    self.validate(interrupt_before + interrupt_after)
    # 3. 确定输出通道
    output_channels = "__root__" if len(self.schemas[self.output_schema]) == 1 and "__root__" in ... \
        else [key for key, val in self.schemas[self.output_schema].items() if not is_managed_value(val)]
    # 4. 创建 CompiledStateGraph 实例
    compiled = CompiledStateGraph(
        builder=self, schema_to_mapper={},
        channels={**self.channels, **self.managed, START: EphemeralValue(self.input_schema)},
        input_channels=START, stream_mode="updates", output_channels=output_channels, ...
    )
    # 5. 附加节点、边、分支
    compiled.attach_node(START, None)
    for key, node in self.nodes.items():
        compiled.attach_node(key, node)
    for start, end in self.edges:
        compiled.attach_edge(start, end)
    for starts, end in self.waiting_edges:
        compiled.attach_edge(starts, end)
    for start, branches in self.branches.items():
        for name, branch in branches.items():
            compiled.attach_branch(start, name, branch)
    return compiled.validate()
```

**`attach_node`** 的核心逻辑（针对非 START 节点）：

```python
def attach_node(self, key, node):
    # 1. 确定输出键
    output_keys = list(self.builder.channels) + [k for k, v in self.builder.managed.items()]
    # 2. 创建写入条目
    write_entries = (
        ChannelWriteTupleEntry(mapper=_get_updates if output_keys != ["__root__"] else _get_root),
        ChannelWriteTupleEntry(mapper=_control_branch, static=_control_static(node.ends) if node.ends else None),
    )
    # 3. 为每个节点创建分支通道
    branch_channel = _CHANNEL_BRANCH_TO.format(key)  # "branch:to:{key}"
    self.channels[branch_channel] = LastValueAfterFinish(Any) if node.defer else EphemeralValue(Any, guard=False)
    # 4. 创建 PregelNode
    self.nodes[key] = PregelNode(
        triggers=[branch_channel],
        channels="__root__" if is_single_input else input_channels,
        mapper=mapper,
        writers=[ChannelWrite(write_entries)],
        metadata=node.metadata, retry_policy=node.retry_policy,
        cache_policy=node.cache_policy, bound=node.runnable,
    )
```

**`attach_edge`** 的核心逻辑：

```python
def attach_edge(self, starts, end):
    if isinstance(starts, str):
        # 简单边：start -> end
        if end != END:
            self.nodes[starts].writers.append(
                ChannelWrite((ChannelWriteEntry(_CHANNEL_BRANCH_TO.format(end), None),))
            )
    else:
        # 等待边：[s1, s2, ...] -> end
        channel_name = f"join:{'+'.join(starts)}:{end}"
        self.channels[channel_name] = NamedBarrierValue(str, set(starts))  # 或 NamedBarrierValueAfterFinish
        self.nodes[end].triggers.append(channel_name)
        for start in starts:
            self.nodes[start].writers.append(ChannelWrite((ChannelWriteEntry(channel_name, start),)))
```

每个简单边 `A -> B` 会向节点 A 的写入列表添加一个 `ChannelWriteEntry("branch:to:B", None)`。当 A 执行完毕，写入 `"branch:to:B"` 通道，触发 B。

### 2.4 `message.py`：add_messages reducer

`add_messages` 是 LangGraph 中最常用的 reducer，使用 `@_add_messages_wrapper` 装饰器实现偏函数模式：

```python
@_add_messages_wrapper
def add_messages(left: Messages, right: Messages, *, format=None):
    # 1. 强制转为列表
    # 2. 强制转为消息对象
    # 3. 为缺失 ID 的消息分配 UUID
    # 4. 处理 REMOVE_ALL_MESSAGES
    # 5. 合并：按 ID 去重更新
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)       # 标记删除
            else:
                ids_to_remove.discard(m.id)    # 取消删除
                merged[existing_idx] = m       # 替换更新
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(f"Cannot delete non-existent message '{m.id}'")
            merged_by_id[m.id] = len(merged)
            merged.append(m)                     # 追加新消息
    merged = [m for m in merged if m.id not in ids_to_remove]
    return merged
```

`_add_messages_wrapper` 实现了偏函数模式：

```python
def _add_messages_wrapper(func):
    def _add_messages(left=None, right=None, **kwargs):
        if left is not None and right is not None:
            return func(left, right, **kwargs)
        elif left is not None or right is not None:
            raise ValueError("Must specify both 'left' and 'right'")
        else:
            return partial(func, **kwargs)  # 返回偏函数
    return cast(Callable[[Messages, Messages], Messages], _add_messages)
```

这允许 `Annotated[list[AnyMessage], add_messages]` 的写法——当 `add_messages` 被用作类型注解的元数据时，它会被无参调用，返回偏函数本身，作为 `BinaryOperatorAggregate` 的 reducer。

`MessagesState` 提供了开箱即用的消息状态：

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

## 3 运行原理

### 3.1 从声明到编译的完整映射

```
StateGraph 声明                    Pregel 运行时
─────────────────                 ──────────────
add_node("A", fn)
    │
    ├──> self.channels[key]       → 通道（如 BinaryOperatorAggregate）
    ├──> self.nodes[key]           → PregelNode(bound=fn, triggers=[...], writers=[...])
    │
add_edge("A", "B")
    │
    ├──> ChannelWriteEntry("branch:to:B", None)
    │        添加到 A 的 writers
    │
add_conditional_edges("A", router)
    │
    ├──> BranchSpec(path=router, ends={...})
    │        添加到 A 的 writers
    │        通过 branch.run() 创建路由逻辑
    │
compile()
    │
    ├──> 创建 EphemeralValue(input_schema) 作为 START 通道
    ├──> 创建 EphemeralValue/LastValueAfterFinish 作为 "branch:to:X" 通道
    ├──> 创建 NamedBarrierValue 作为 "join:A+B:C" 等待通道
    ├──> 构建 PregelNode(triggers, channels, writers, bound)
    └──> 返回 CompiledStateGraph(Pregel 子类)
```

### 3.2 通道类型决策流程

```
字段注解                             通道类型
─────────                            ────────
x: int                               LastValue(int)
x: Annotated[int, reducer]           BinaryOperatorAggregate(int, reducer)
x: Annotated[int, EphemeralValue]    EphemeralValue(int)
x: Annotated[int, Context]           ManagedValueSpec (非通道)

特殊通道：
START                                EphemeralValue(input_schema)
"branch:to:Node"                     EphemeralValue(Any, guard=False) 或 LastValueAfterFinish(Any)
"join:A+B:C"                        NamedBarrierValue(str, {A, B})
```

## 4 实现细节

### 4.1 `_get_channels` 的类型解析

`_get_channels` 通过 `get_type_hints(schema, include_extras=True)` 获取带 `Annotated` 的类型提示。`include_extras=True` 是关键参数——默认情况下 `get_type_hints` 会剥除 `Annotated` 包装，但这里需要保留它以检测 reducer 和通道类型。

### 4.2 `_is_field_binop` 的 reducer 验证

```python
def _is_field_binop(typ):
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1 and callable(meta[-1]):
            sig = signature(meta[-1])
            params = list(sig.parameters.values())
            if sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params) == 2:
                return BinaryOperatorAggregate(typ, meta[-1])
            else:
                raise ValueError(f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}")
```

这里验证了 reducer 必须恰好接受 2 个位置参数。`BinaryOperatorAggregate` 的语义是：每次状态更新时，用 `reducer(old_value, new_value)` 计算新值。

### 4.3 add_conditional_edges 的 Literal 推断

当 `path_map` 为 `None` 且路由函数声明了 `Literal` 返回类型时：

```python
def router(state) -> Literal["a", "b", "__end__"]:
    ...

# 编译时自动推断：
path_map = {"a": "a", "b": "b", "__end__": "__end__"}
```

这通过 `get_type_hints(func).get("return")` → `get_origin()` 检查 `Literal` → `get_args()` 提取值来实现。

### 4.4 `_control_branch` 与 Command 支持

`_control_branch` 处理节点返回 `Command` 的情况：

```python
def _control_branch(value):
    if isinstance(value, Send):
        return ((TASKS, value),)
    commands = []
    if isinstance(value, Command):
        commands.append(value)
    elif isinstance(value, (list, tuple)):
        for cmd in value:
            if isinstance(cmd, Command):
                commands.append(cmd)
    rtn = []
    for command in commands:
        goto_targets = [command.goto] if isinstance(command.goto, (Send, str)) else command.goto
        for go in goto_targets:
            if isinstance(go, Send):
                rtn.append((TASKS, go))
            elif isinstance(go, str) and go != END:
                rtn.append((_CHANNEL_BRANCH_TO.format(go), None))
    return rtn
```

### 4.5 add_messages 的去重逻辑

`add_messages` 的合并逻辑处理了以下边界情况：

- **ID 去重**：相同 ID 的新消息替换旧消息
- **RemoveMessage**：标记 ID 为待删除，最后统一过滤
- **REMOVE_ALL_MESSAGES**：清空所有旧消息，只保留 `RemoveMessage` 之后的消息
- **ID 缺失**：自动分配 UUID

```python
# REMOVE_ALL_MESSAGES 处理
if remove_all_idx is not None:
    return right[remove_all_idx + 1:]
```

## 5 动手实验

### 实验 1：构建 StateGraph 并检查通道

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

def reducer(a: list, b: int) -> list:
    return a + [b]

class State(TypedDict):
    x: Annotated[list, reducer]
    y: int

builder = StateGraph(State)

# 检查 _add_schema 创建的通道
for name, channel in builder.channels.items():
    print(f"{name}: {type(channel).__name__}({channel.update_type})")

# 输出：
# x: BinaryOperatorAggregate(<class 'list'>)
# y: LastValue(<class 'int'>)

def node_a(state: State) -> dict:
    return {"x": 1, "y": state["y"] + 1}

builder.add_node("a", node_a)
builder.add_edge(START, "a")
builder.add_edge("a", END)

graph = builder.compile()

# 检查编译后的通道
for name, channel in graph.channels.items():
    print(f"{name}: {type(channel).__name__}")

# 输出：
# x: BinaryOperatorAggregate
# y: LastValue
# __start__: EphemeralValue
# branch:to:a: EphemeralValue
# branch:to:__end__: EphemeralValue
```

### 实验 2：追踪 compile() 的步骤

```python
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(MyState)

def chatbot(state):
    return {"messages": [("assistant", "Hello!")]}

builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# 检查节点定义
for name, node in graph.nodes.items():
    print(f"Node: {name}")
    print(f"  triggers: {node.triggers}")
    print(f"  channels: {node.channels}")
    print(f"  writers: {[type(w).__name__ for w in node.writers]}")

# Node: __start__
#   triggers: ['__start__']
#   channels: __start__
#   writers: ['ChannelWrite']
# Node: chatbot
#   triggers: ['branch:to:chatbot']
#   channels: ['messages']
#   writers: ['ChannelWrite']
```

### 实验 3：Literal 推断条件边

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    value: int

def router(state: State) -> Literal["positive", "negative", "__end__"]:
    if state["value"] > 0:
        return "positive"
    elif state["value"] < 0:
        return "negative"
    else:
        return "__end__"

builder = StateGraph(State)
builder.add_node("positive", lambda s: {"value": s["value"]})
builder.add_node("negative", lambda s: {"value": s["value"]})
builder.add_conditional_edges(START, router)

# 检查 branches —— path_map 自动推断自 Literal
for source, branches in builder.branches.items():
    for name, spec in branches.items():
        print(f"Branch: {source} -> {name}")
        print(f"  ends: {spec.ends}")
        print(f"  input_schema: {spec.input_schema}")

# Branch: __start__ -> condition
#   ends: {'positive': 'positive', 'negative': 'negative', '__end__': '__end__'}
#   input_schema: State
```

### 实验 4：add_messages 去重行为

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage

# 基本合并
msgs1 = [HumanMessage(content="Hi", id="1")]
msgs2 = [AIMessage(content="Hello!", id="2")]
result = add_messages(msgs1, msgs2)
print(f"合并: {[f'{m.id}:{m.content}' for m in result]}")
# ['1:Hi', '2:Hello!']

# ID 去重替换
msgs3 = [HumanMessage(content="Hi again", id="1")]  # 同 ID，新内容
result2 = add_messages(result, msgs3)
print(f"替换: {[f'{m.id}:{m.content}' for m in result2]}")
# ['1:Hi again', '2:Hello!']

# RemoveMessage
msgs4 = [RemoveMessage(id="2")]
result3 = add_messages(result2, msgs4)
print(f"删除: {[f'{m.id}:{m.content}' for m in result3]}")
# ['1:Hi again']
```