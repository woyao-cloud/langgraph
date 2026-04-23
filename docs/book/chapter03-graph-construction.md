# 第三章 图构建 — graph/ 模块

> 从节点定义到编译输出，理解 LangGraph 的图构建全流程

本章走进 `langgraph/graph/` 目录，解析 LangGraph 如何将用户编写的节点与边声明，逐步组装为一个可执行的 Pregel 计算图。这是从"声明式描述"到"命令式运行"的桥梁。

---

## 3.1 节点类型系统 — `_node.py`

### Java 桥梁

在 Java 中，我们习惯用接口（interface）来定义契约。一个节点函数通常被定义为某个 `@FunctionalInterface`，比如 `Function<State, Result>`。但在 LangGraph 中，节点函数的签名是极其灵活的——它可以只接收 state，也可以额外接收 config、writer、store 等参数。Python 没有方法重载，LangGraph 用 **Protocol（结构化子类型）** 来描述这些不同的签名变体。

### Python 概念速查

| Python 概念 | Java 对应 | 说明 |
|---|---|---|
| `Protocol` | `interface`（但鸭子类型） | Protocol 是结构化子类型，只要类的方法签名匹配就满足，无需 `implements` |
| `TypeAlias` | 无直接对应 | 类型别名，给复杂的联合类型起一个可读的名字 |
| `@overload` | 方法重载 | Python 不支持真正的重载，`@overload` 仅用于类型检查器的签名提示 |

### 代码走读

`_node.py` 文件仅 93 行，却定义了 LangGraph 节点类型系统的核心。

**Protocol 家族**：文件定义了 9 个 Protocol 类，每个对应一种节点函数签名：

```python
class _Node(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra) -> Any: ...

class _NodeWithConfig(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, config: RunnableConfig) -> Any: ...

class _NodeWithWriter(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, writer: StreamWriter) -> Any: ...

class _NodeWithStore(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, store: BaseStore) -> Any: ...
```

这些 Protocol 使用了逆变类型参数 `NodeInputT_contra`（contra = 逆变的），这是类型理论中的概念。Java 开发者可以类比 `? super T` 通配符——节点消费 state，因此对输入类型是逆变的。

其余 Protocol 组合了更多参数：`_NodeWithWriterStore`、`_NodeWithConfigWriter`、`_NodeWithConfigStore`、`_NodeWithConfigWriterStore`，以及支持 `Runtime` 上下文的 `_NodeWithRuntime`。

**StateNode 类型别名**：将所有 Protocol 变体和 `Runnable` 合并为一个联合类型：

```python
StateNode: TypeAlias = (
    _Node[NodeInputT]
    | _NodeWithConfig[NodeInputT]
    | _NodeWithWriter[NodeInputT]
    | _NodeWithStore[NodeInputT]
    | _NodeWithWriterStore[NodeInputT]
    | _NodeWithConfigWriter[NodeInputT]
    | _NodeWithConfigStore[NodeInputT]
    | _NodeWithConfigWriterStore[NodeInputT]
    | _NodeWithRuntime[NodeInputT, ContextT]
    | Runnable[NodeInputT, Any]
)
```

Java 开发者注意：Python 的 `|` 联合类型相当于 Java 的多个 `implements` 接口，但更灵活——不需要实际声明，只要运行时签名匹配即可。`TypeAlias` 不创建新类，仅仅是类型检查阶段的别名。

**StateNodeSpec 数据类**：将一个节点的运行时配置打包：

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

`slots=True` 是 Python 3.10+ dataclass 的优化选项，等价于 Java 中用 `final` 字段声明而非反射式 Map 存储——通过预分配的 slot 访问属性，比默认的 `__dict__` 更快、更省内存。

### 运行原理

当你在 StateGraph 中调用 `add_node("chatbot", my_func)` 时，LangGraph 会根据 `my_func` 的参数签名自动判断它属于哪个 Protocol 变体，然后将它包装为 `StateNodeSpec` 存入 `self.nodes` 字典。`ends` 字段记录该节点可能跳转的目标（从返回类型 `Literal["node_a", "node_b"]` 中推断），`defer` 标记该节点是否延迟到运行结束前执行。

---

## 3.2 条件分支 — `_branch.py`

### Java 桥梁

Java 中的路由通常用 `switch` 语句或策略模式实现。LangGraph 将条件分支抽象为 `BranchSpec`——一个携带路由函数和目标映射的不可变对象。与 Java 的 `enum` + `Map` 不同，它还能通过类型注解自动推断路由目标。

### Python 概念速查

| Python 概念 | Java 对应 | 说明 |
|---|---|---|
| `NamedTuple` + 方法 | 不可变 Java record + 工具方法 | NamedTuple 可以拥有 `@classmethod` 和实例方法 |
| `get_type_hints()` | `Method.getGenericReturnType()` | 运行时获取函数的类型注解 |
| `get_origin()` / `get_args()` | `ParameterizedType.getRawType()` / `getActualTypeArguments()` | 拆解泛型类型，如 `Literal["a","b"]` 的 origin 是 `Literal`，args 是 `("a","b")` |
| `zip_longest` | 无内置对应 | 用 `None` 填充不等长序列的 zip |

### 代码走读

**BranchSpec 定义**：

```python
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]
    ends: dict[Hashable, str] | None
    input_schema: type[Any] | None = None
```

三个字段：`path` 是路由函数（可同步/异步），`ends` 是路由值到目标节点名的映射，`input_schema` 是分支函数的输入类型（用于类型推断）。

**from_path() 类方法**——这是最精巧的部分。它将用户传入的 `path` 函数和 `path_map` 转化为标准化的 `BranchSpec`：

```python
@classmethod
def from_path(cls, path, path_map, infer_schema=False):
    # 1. 如果 path_map 是 list，转为 dict：["a","b"] → {"a":"a", "b":"b"}
    if isinstance(path_map, list):
        path_map_ = {name: name for name in path_map}
    # 2. 如果 path_map 未提供，从返回类型注解推断
    elif func is not None:
        if rtn_type := get_type_hints(func).get("return"):
            if get_origin(rtn_type) is Literal:
                path_map_ = {name: name for name in get_args(rtn_type)}
    # 3. 推断输入 schema
    input_schema = _get_branch_path_input_schema(path) if infer_schema else None
    return cls(path=path, ends=path_map_, input_schema=input_schema)
```

这段代码的精妙之处在于**从类型注解自动推断路由映射**。如果你的路由函数声明为：

```python
def route(state) -> Literal["search", "codegen"]:
    ...
```

`get_type_hints(route)` 返回 `{"return": Literal["search", "codegen"]}`，`get_origin()` 判定它是 `Literal`，`get_args()` 提取 `("search", "codegen")`，于是自动构建 `{"search": "search", "codegen": "codegen"}`。Java 的注解处理器在编译期做类似的事，但 Python 在运行时通过反射完成。

**_route / _aroute 方法**：同步和异步路由逻辑完全对称：

```python
def _route(self, input, config, *, reader, writer):
    if reader:
        value = reader(config)
        # 字典状态下合并额外 key
        if isinstance(value, dict) and isinstance(input, dict) and self.input_schema is None:
            value = {**input, **value}
    else:
        value = input
    result = self.path.invoke(value, config)
    return self._finish(writer, input, result, config)
```

`reader` 从通道读取当前状态，`writer` 将路由结果写入目标通道。`{**input, **value}` 是 Python 的字典合并语法——先展开 input，再用 value 覆盖，等价于 Java 的 `new HashMap<>(input); map.putAll(value)`。

**_finish 方法**中的 `zip_longest` 用于将写入条目与标签配对注册：

```python
list(zip_longest(
    writer([e for e in self.ends.values()], True),
    [str(la) for la, e in self.ends.items()],
)) if self.ends else None
```

`zip_longest` 保证了即使 writer 返回的条目数与 ends 数量不完全匹配（理论上不应该），也不会丢失信息——缺失位置填充 `None`。这比普通 `zip` 更安全。

### 运行原理

当 `add_conditional_edges("router", route_fn, {"search": "search_node", ...})` 被调用时，`from_path()` 创建 `BranchSpec`。在编译阶段，`BranchSpec.run()` 生成一个 `RunnableCallable`，它被注册为源节点的写入器。运行时，路由函数的返回值通过 `ends` 映射转化为通道写入目标。

---

## 3.3 状态图构建器 — `state.py`

### Java 桥梁

如果你用过 Apache Camel 或 Spring Integration 的 `RouteBuilder`，会对 StateGraph 感到亲切——它是一个"构建器"，逐步收集节点和边的声明，最终 `compile()` 生成可运行的图。不同之处在于 StateGraph 不是用方法链式调用，而是返回 `self` 以支持链式调用。

### Python 概念速查

| Python 概念 | Java 对应 | 说明 |
|---|---|---|
| `defaultdict(dict)` | `Map<String, Map<String, BranchSpec>>` + `computeIfAbsent` | 缺省值自动初始化为空字典 |
| `@overload` | 方法重载（overloading） | Python 的 `@overload` 仅用于类型检查，运行时只有一个实现 |
| `Generic[StateT, ContextT, InputT, OutputT]` | `class StateGraph<S, C, I, O>` | Python 支持多个类型参数 |
| `__name__` | `Class.getSimpleName()` / `Method.getName()` | 获取函数/类的名称字符串 |
| `cast()` | 强制类型转换 `(Type) obj` | Python 的 `cast()` 仅影响类型检查器，运行时不执行任何操作 |

### 代码走读

**类声明与字段**：

```python
class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    edges: set[tuple[str, str]]
    nodes: dict[str, StateNodeSpec[Any, ContextT]]
    branches: defaultdict[str, dict[str, BranchSpec]]
    channels: dict[str, BaseChannel]
    managed: dict[str, ManagedValueSpec]
    schemas: dict[type[Any], dict[str, BaseChannel | ManagedValueSpec]]
    waiting_edges: set[tuple[tuple[str, ...], str]]
    compiled: bool
```

`branches` 使用 `defaultdict(dict)`——当访问一个不存在的 key 时，自动创建空字典。Java 中你需要 `branches.computeIfAbsent(k, _ -> new HashMap<>())`。`waiting_edges` 存储扇入边（多个源到一个目标），如 `add_edge(["A","B"], "C")`。

**__init__**：

```python
def __init__(self, state_schema, context_schema=None, *, input_schema=None, output_schema=None):
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

`cast()` 在此的作用值得说明：当 `input_schema` 为 `None` 时回退到 `state_schema`，但 `or` 运算的结果类型是 `type[InputT] | type[StateT]`，类型检查器无法自动收窄。`cast()` 告诉 mypy "相信我，这个值的类型是 `type[InputT]`"，但运行时不会有任何检查——这是 Python 类型系统的"信任但不需要验证"哲学。

**add_node 方法与 @overload**：`add_node` 有 4 个 `@overload` 签名加 1 个实际实现：

1. 传入函数，input_schema 推断为 state schema
2. 传入函数，显式指定 input_schema
3. 传入字符串 + 函数，input_schema 推断
4. 传入字符串 + 函数，显式指定 input_schema

Java 开发者注意：Python 的 `@overload` 是**纯类型提示**，不影响运行时行为。在 Java 中，编译器根据参数类型选择重载版本；在 Python 中，运行时方法体内需要自己判断参数类型。`add_node` 的实现通过 `if not isinstance(node, str):` 来区分调用形式。

**add_node 的节点名推断**：

```python
if not isinstance(node, str):
    action = node
    if isinstance(action, Runnable):
        node = action.get_name()
    else:
        node = getattr(action, "__name__", action.__class__.__name__)
```

`__name__` 是 Python 函数对象的内置属性，等价于 Java 的 `Method.getName()`。`getattr(action, "__name__", action.__class__.__name__)` 先尝试取 `__name__`，取不到则用类名——这是一种防御性写法，类似 Java 的 `Optional.ofNullable(action.getName()).orElse(action.getClass().getSimpleName())`。

**add_edge 方法**：

```python
def add_edge(self, start_key: str | list[str], end_key: str) -> Self:
    if isinstance(start_key, str):
        self.edges.add((start_key, end_key))
    else:
        self.waiting_edges.add((tuple(start_key), end_key))
    return self
```

返回 `Self` 是 Python 3.11+ 引入的类型注解，表示返回当前类的实例——支持方法链。Java 中的 Builder 模式返回 `this` 是同样的思路。

**compile() 方法**：这是 StateGraph 最重要的方法，将构建器产物转化为可执行的 `CompiledStateGraph`（Pregel 子类）：

```python
def compile(self, checkpointer=None, *, store=None, interrupt_before=None, ...):
    compiled = CompiledStateGraph(
        builder=self,
        nodes={},
        channels={**self.channels, **self.managed, START: EphemeralValue(self.input_schema)},
        input_channels=START,
        output_channels=output_channels,
        checkpointer=checkpointer,
        ...
    )
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

`{**self.channels, **self.managed, START: EphemeralValue(...)}` 是 Python 字典解包合并语法——先展开 channels，再展开 managed，最后添加 START 通道。编译后的图是一个全新的对象，原 StateGraph 保持不变——体现了不可变构建模式。

### 运行原理

StateGraph 的生命周期是：构造 → 添加节点/边 → compile() → 运行 CompiledStateGraph。构建器是"一次性的"——`compiled` 标志位阻止在编译后继续修改。`compile()` 将声明式的图描述转化为命令式的 Pregel 执行引擎：每个节点变成 `PregelNode`，每条边变成通道订阅/写入规则。

---

## 3.4 边到通道的映射

这是理解 LangGraph 内部机制的关键。用户写的"边"在底层全部转化为"通道"。

### 单源边：`add_edge("A", "B")`

在 `CompiledStateGraph.attach_edge()` 中：

```python
if isinstance(starts, str):
    if end != END:
        self.nodes[starts].writers.append(
            ChannelWrite((ChannelWriteEntry(_CHANNEL_BRANCH_TO.format(end), None),))
        )
```

其中 `_CHANNEL_BRANCH_TO = "branch:to:{}"` 。所以 `add_edge("A", "B")` 的效果是：
- 创建名为 `"branch:to:B"` 的通道（类型 `EphemeralValue`）
- 在节点 A 的 writers 中追加：执行完后写入 `"branch:to:B"` 通道
- 节点 B 的 triggers 中包含 `"branch:to:B"`——当该通道有更新时触发 B

**EphemeralValue** 是"临时值"通道——每个超级步（superstep）结束后自动清零，只用于信号传递，不持久化状态。

### 扇入边：`add_edge(["A", "B"], "C")`

```python
elif end != END:
    channel_name = f"join:{'+'.join(starts)}:{end}"
    self.channels[channel_name] = NamedBarrierValue(str, set(starts))
    self.nodes[end].triggers.append(channel_name)
    for start in starts:
        self.nodes[start].writers.append(
            ChannelWrite((ChannelWriteEntry(channel_name, start),))
        )
```

通道名为 `"join:A+B:C"`，类型为 **NamedBarrierValue**——这是一个屏障同步通道，它需要收到来自 `set("A", "B")` 中**所有**源节点的写入后才会触发目标节点。每个源节点写入自己的名字作为值。

Java 开发者可以类比 `java.util.concurrent.CyclicBarrier` 或 `CountDownLatch`——所有参与方都到达后才放行。但 NamedBarrierValue 更轻量，它在每个超级步中重置，不需要显式 await。

### 条件边：`add_conditional_edges("A", route_fn, path_map)`

编译后调用 `attach_branch()`，将 `BranchSpec.run()` 生成的 `RunnableCallable` 注册为节点 A 的 writer。路由函数的返回值通过 `path_map` 映射为目标通道名（仍然是 `branch:to:X` 格式）。

---

## 3.5 消息处理 — `message.py`

### Java 桥梁

如果你用过 Akka 的 `Message` 体系或 Spring Messaging，会对 LangGraph 的消息模型感到熟悉。但 LangGraph 的 `add_messages` reducer 解决了一个 Java 开发者常遇到的难题：消息去重与覆盖。

### Python 概念速查

| Python 概念 | Java 对应 | 说明 |
|---|---|---|
| `Annotated[list, add_messages]` | 无直接对应 | `Annotated` 将元数据（reducer 函数）附加到类型声明上 |
| `TypedDict` | `record` / `@Data` | 轻量级的结构化类型，比 Pydantic Model 更轻 |
| `@deprecated` | `@Deprecated` | 标记废弃 API |

### 代码走读

**add_messages reducer**：这是 LangGraph 中最常用的 reducer 函数，用于合并两个消息列表。核心逻辑如下：

```python
@_add_messages_wrapper
def add_messages(left: Messages, right: Messages, *, format=None) -> Messages:
    # 1. 强制转为列表
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    # 2. 强制转为 BaseMessage
    left = [message_chunk_to_message(cast(BaseMessageChunk, m)) for m in convert_to_messages(left)]
    right = [message_chunk_to_message(cast(BaseMessageChunk, m)) for m in convert_to_messages(right)]
    # 3. 为缺少 id 的消息分配 UUID
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    # 4. 合并：同 id 覆盖，RemoveMessage 删除
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
            else:
                ids_to_remove.discard(m.id)
                merged[existing_idx] = m  # 同 id 覆盖
        else:
            merged_by_id[m.id] = len(merged)
            merged.append(m)  # 追加
    merged = [m for m in merged if m.id not in ids_to_remove]
    return merged
```

**去重逻辑**：每条消息有 `id` 属性。合并时：
- 如果 right 中的消息 id 在 left 中已存在，则用 right 的消息**覆盖** left 的
- 如果 right 中是 `RemoveMessage`（且 id 匹配），则从结果中**删除**该消息
- 如果 right 中是特殊的 `REMOVE_ALL_MESSAGES`（id 为 `"__remove_all__"`），则清空 left，只保留该标记之后的消息

这解决了聊天场景中的典型问题：AI 重新生成回复时，新消息应该替换同 id 的旧消息，而非追加。

**`_add_messages_wrapper` 装饰器**：

```python
def _add_messages_wrapper(func):
    def _add_messages(left=None, right=None, **kwargs):
        if left is not None and right is not None:
            return func(left, right, **kwargs)
        elif left is not None or right is not None:
            raise ValueError("Must specify both 'left' and 'right'")
        else:
            return partial(func, **kwargs)
    return cast(Callable, _add_messages)
```

这个装饰器让 `add_messages` 具备两种调用模式：传入两个参数时直接合并；不传参数时返回绑定了 `format` 参数的部分应用函数（`partial`）。Java 中这等价于重载——Python 用 `partial` 实现了类似效果。

**MessagesState 与 MessageGraph**：

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`MessagesState` 是最常用的预定义状态——一个 `messages` 键，reducer 为 `add_messages`。`Annotated[list[AnyMessage], add_messages]` 告诉 StateGraph：当多个节点写入 `messages` 时，用 `add_messages` 函数合并。

`MessageGraph` 已被废弃，推荐用 `StateGraph(MessagesState)` 替代。

### 运行原理

在 StateGraph 中，`Annotated[type, reducer]` 声明了一个带有归约函数的通道。当节点 A 返回 `{"messages": [msg1]}`、节点 B 返回 `{"messages": [msg2]}` 时，`add_messages` 被调用两次，依次合并进通道。最终 `messages` 通道的值是去重后的消息列表。

---

## 3.6 动手实验

### 实验 1：观察边到通道的映射

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int

builder = StateGraph(State)
builder.add_node("A", lambda s: {"count": s["count"] + 1})
builder.add_node("B", lambda s: {"count": s["count"] + 10})
builder.add_node("C", lambda s: {"count": s["count"] + 100})
builder.add_edge(START, "A")
builder.add_edge("A", "B")
builder.add_edge(["A", "B"], "C")  # 扇入边
builder.add_edge("C", END)

graph = builder.compile()

# 检查编译后的通道
print("Channels:", list(graph.channels.keys()))
# 你会看到: 'branch:to:B', 'branch:to:C', 'join:A+B:C' 等通道

# 检查节点 B 的触发器
print("Node B triggers:", graph.nodes["B"].triggers)
# 输出: ['branch:to:B']
```

### 实验 2：消息去重

```python
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import add_messages

msgs1 = [HumanMessage(content="Hello", id="1"), AIMessage(content="Hi", id="2")]
msgs2 = [HumanMessage(content="Hello again", id="1")]  # 同 id 覆盖

result = add_messages(msgs1, msgs2)
print([(m.id, m.content) for m in result])
# [("1", "Hello again"), ("2", "Hi")]  -- id=1 被覆盖

# 删除消息
msgs3 = [RemoveMessage(id="2")]
result2 = add_messages(result, msgs3)
print([(m.id, m.content) for m in result2])
# [("1", "Hello again")]  -- id=2 被删除
```

### 实验 3：Protocol 结构化子类型验证

```python
from langgraph.graph._node import _Node, _NodeWithConfig
from langchain_core.runnables import RunnableConfig

# 普通函数自动满足 _Node Protocol
def my_node(state: dict) -> dict:
    return {"count": state["count"] + 1}

# 带 config 的函数自动满足 _NodeWithConfig Protocol
def my_node_with_config(state: dict, config: RunnableConfig) -> dict:
    return {"count": state["count"] + 1}

# 无需 implements 声明，签名匹配即满足
print(callable(my_node))  # True
```

---

## 小结

本章揭示了 LangGraph 图构建的核心机制：

1. **Protocol 替代 Interface**：节点的多种签名变体用 Protocol 联合类型描述，运行时按鸭子类型匹配
2. **边即通道**：`add_edge("A","B")` 创建 `branch:to:B` 通道，扇入边创建 `NamedBarrierValue` 屏障通道
3. **compile() 是关键转换点**：将声明式的 StateGraph 编译为命令式的 CompiledStateGraph（Pregel 子类）
4. **add_messages 是最常用的 reducer**：通过消息 id 实现去重、覆盖、删除

下一章将深入 Pregel 执行引擎，看看编译后的图是如何运行的。