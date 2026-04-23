# 第六章 读写管线——数据的流入与流出

## Python 进阶

本章源码涵盖了以下 Python 高级特性：

- **`@cached_property` vs `@property`**：`PregelNode` 的 `node`、`flat_writers`、`input_cache_key` 使用 `@cached_property`，首次访问时计算并缓存结果，后续直接返回缓存值。相比 `@property` 每次访问都重新计算，这对组合型属性至关重要。
- **`__slots__`**：`Call` 类使用 `__slots__` 声明属性，阻止动态属性添加，减少内存占用。
- **`NamedTuple` 及其方法**：`ChannelWriteEntry` 和 `ChannelWriteTupleEntry` 继承 `NamedTuple`，既具备元组的不可变性和轻量性，又支持点号访问和默认值。
- **`classmethod` vs `staticmethod`**：`ChannelWrite.do_write()` 和 `ChannelRead.do_read()` 是 `@staticmethod`，不依赖实例状态，可以在任意上下文中调用。`ChannelWrite.register_writer()` 和 `is_writer()` 也是 `@staticmethod`，用于鸭类型标记检测。
- **`__init_subclass__` 钩子**：LangGraph 的 `Runnable` 基类使用此钩子注册子类，实现类型自动发现。
- **海象运算符 `:=`**：`_runner.py` 中大量使用赋值表达式，如 `if cb := self.callback():`、`elif exc := fut.exception():`，简化了条件判断和变量提取。
- **`ParamSpec` 和 `TypeVar`**：`call()` 函数使用 `ParamSpec("P")` 和 `TypeVar("T")` 泛化函数签名，让类型检查器正确推断输入/输出类型。
- **`inspect` 模块**：`identifier()` 使用 `__qualname__`、`__module__` 等内省属性；`_explode_args_trace_inputs()` 使用 `inspect.Signature.bind_partial()` 解构参数。
- **`functools.wraps`**：`get_runnable_for_entrypoint()` 使用 `functools.update_wrapper` 将原函数的元数据复制到偏函数。
- **async/sync 双模式**：`ChannelRead._read()` / `_aread()` 是同一逻辑的同步/异步镜像。
- **`__or__` 运算符重载**：LangChain 的 `Runnable` 类通过 `__or__` 实现 `|` 管道组合，`RunnableSeq` 是组合后的序列执行器。

## 代码走读

### _read.py —— 读取管线

#### ChannelRead

```python
class ChannelRead(RunnableCallable):
    channel: str | list[str]
    fresh: bool = False
    mapper: Callable[[Any], Any] | None = None
```

`ChannelRead` 是一个 `RunnableCallable`，既可作为 Runnable 执行（`invoke`/`ainvoke`），也可作为静态方法直接调用。构造时注册 `self._read` 和 `self._aread` 两个方法：

```python
def __init__(self, channel, *, fresh=False, mapper=None, tags=None):
    super().__init__(
        func=self._read,
        afunc=self._aread,
        tags=tags,
        name=None,
        trace=False,
    )
    self.fresh = fresh
    self.mapper = mapper
    self.channel = channel
```

**同步/异步双入口**：

```python
def _read(self, _: Any, config: RunnableConfig) -> Any:
    return self.do_read(config, select=self.channel, fresh=self.fresh, mapper=self.mapper)

async def _aread(self, _: Any, config: RunnableConfig) -> Any:
    return self.do_read(config, select=self.channel, fresh=self.fresh, mapper=self.mapper)
```

两者都委托给 `do_read()` 静态方法。由于读取操作本身不涉及 I/O，异步版本只是保持接口一致性。

**核心静态方法 do_read()**：

```python
@staticmethod
def do_read(config, *, select, fresh=False, mapper=None):
    try:
        read: READ_TYPE = config[CONF][CONFIG_KEY_READ]
    except KeyError:
        raise RuntimeError("Not configured with a read function"
                         "Make sure to call in the context of a Pregel process")
    if mapper:
        return mapper(read(select, fresh))
    else:
        return read(select, fresh)
```

关键设计：`CONFIG_KEY_READ` 是从 `config` 中注入的读取函数（实际是 `local_read` 的偏函数）。`select` 参数决定读取模式——字符串读取单个 channel，列表读取多个 channel 并返回字典。`mapper` 可选地变换结果。

#### PregelNode —— 节点容器

```python
class PregelNode:
    channels: str | list[str]
    triggers: list[str]
    mapper: Callable[[Any], Any] | None
    writers: list[Runnable]
    bound: Runnable[Any, Any]
    retry_policy: Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    tags: Sequence[str] | None
    metadata: Mapping[str, Any] | None
    subgraphs: Sequence[PregelProtocol]
```

`PregelNode` 不是直接被执行的 Runnable，而是**任务组装器**——它持有构建 `PregelExecutableTask` 所需的全部信息。

**`__init__` 子图自动发现**：

```python
if subgraphs is not None:
    self.subgraphs = subgraphs
elif self.bound is not DEFAULT_BOUND:
    try:
        subgraph = find_subgraph_pregel(self.bound)
    except Exception:
        subgraph = None
    if subgraph:
        self.subgraphs = [subgraph]
    else:
        self.subgraphs = []
```

如果未显式提供子图，构造器尝试通过 `find_subgraph_pregel` 从 `bound` 中自动发现子图。这是一个容错设计——发现失败不影响主流程。

**`flat_writers` cached_property**：

```python
@cached_property
def flat_writers(self) -> list[Runnable]:
    writers = self.writers.copy()
    while (len(writers) > 1
           and isinstance(writers[-1], ChannelWrite)
           and isinstance(writers[-2], ChannelWrite)):
        writers[-2] = ChannelWrite(writes=writers[-2].writes + writers[-1].writes)
        writers.pop()
    return writers
```

这是一个优化：如果相邻的两个写入器都是 `ChannelWrite`，则合并它们的写入条目。这减少了执行时的函数调用开销。使用 `@cached_property` 保证只计算一次。

**`node` cached_property**：

```python
@cached_property
def node(self) -> Runnable[Any, Any] | None:
    writers = self.flat_writers
    if self.bound is DEFAULT_BOUND and not writers:
        return None
    elif self.bound is DEFAULT_BOUND and len(writers) == 1:
        return writers[0]
    elif self.bound is DEFAULT_BOUND:
        return RunnableSeq(*writers)
    elif writers:
        return RunnableSeq(self.bound, *writers)
    else:
        return self.bound
```

`node` 将 `bound`（核心逻辑）和 `writers`（输出写入）组合成单一 Runnable。几种情况：
- 无 `bound` 也无 `writers`：返回 `None`（纯触发节点，不执行逻辑）；
- 只有 `writers`：返回写入器序列；
- 两者都有：`bound | writer1 | writer2`（管道组合）。

**`input_cache_key` cached_property**：

```python
@cached_property
def input_cache_key(self) -> INPUT_CACHE_KEY_TYPE:
    return (self.mapper, tuple(self.channels) if isinstance(self.channels, list) else (self.channels,))
```

这个键用于 `prepare_next_tasks` 中的输入缓存——如果两个节点有相同的 `mapper` 和 `channels`，它们的输入值相同，只需计算一次。

**`copy()` 方法**：

```python
def copy(self, update: dict[str, Any]) -> PregelNode:
    attrs = {**self.__dict__, **update}
    attrs.pop("flat_writers", None)
    attrs.pop("node", None)
    attrs.pop("input_cache_key", None)
    return PregelNode(**attrs)
```

显式移除 `cached_property` 缓存，确保副本重新计算。这是不可变模式——创建新实例而非修改原实例。

### _write.py —— 写入管线

#### PASSTHROUGH 哨兵值

```python
SKIP_WRITE = object()
PASSTHROUGH = object()
```

`PASSTHROUGH` 是写入管线的核心设计模式。它是一个模块级 `object()` 实例，保证全局唯一（通过 `is` 而非 `==` 比较）。含义是"这个写入条目的值在构造时未知，需要从运行时输入中获取"。

#### ChannelWriteEntry

```python
class ChannelWriteEntry(NamedTuple):
    channel: str
    value: Any = PASSTHROUGH
    skip_none: bool = False
    mapper: Callable | None = None
```

继承 `NamedTuple`，获得不可变性和轻量性。`value` 默认为 `PASSTHROUGH`，表示"从上游 Runnable 的输出中取值"。`skip_none` 在条件写入中使用——当值为 `None` 时跳过写入。`mapper` 可选地变换值。

#### ChannelWriteTupleEntry

```python
class ChannelWriteTupleEntry(NamedTuple):
    mapper: Callable[[Any], Sequence[tuple[str, Any]] | None]
    value: Any = PASSTHROUGH
    static: Sequence[tuple[str, Any, str | None]] | None = None
```

用于条件写入——`mapper` 从输入值中提取 `(channel, value)` 元组序列。`static` 字段用于静态分析，声明可能写入的 channel，使图在执行前就能确定边的连接关系。

#### ChannelWrite._write() / _awrite()

```python
def _write(self, input: Any, config: RunnableConfig) -> None:
    writes = [
        ChannelWriteEntry(write.channel, input, write.skip_none, write.mapper)
        if isinstance(write, ChannelWriteEntry) and write.value is PASSTHROUGH
        else ChannelWriteTupleEntry(write.mapper, input)
        if isinstance(write, ChannelWriteTupleEntry) and write.value is PASSTHROUGH
        else write
        for write in self.writes
    ]
    self.do_write(config, writes)
    return input
```

列表推导式将所有 `PASSTHROUGH` 值替换为实际的 `input`。注意条件链：
1. `ChannelWriteEntry` 且 `value is PASSTHROUGH`：创建新条目，`value` 替换为 `input`；
2. `ChannelWriteTupleEntry` 且 `value is PASSTHROUGH`：创建新条目，`value` 替换为 `input`；
3. 其他（包括 `Send` 对象或已有具体值的条目）：保持不变。

函数返回 `input`（而非 `None`），确保 `ChannelWrite` 在管道中不吞没上游数据。

#### ChannelWrite.do_write() —— 核心写入方法

```python
@staticmethod
def do_write(config, writes, allow_passthrough=True):
    for w in writes:
        if isinstance(w, ChannelWriteEntry):
            if w.channel == TASKS:
                raise InvalidUpdateError("Cannot write to the reserved channel TASKS")
            if w.value is PASSTHROUGH and not allow_passthrough:
                raise InvalidUpdateError("PASSTHROUGH value must be replaced")
        if isinstance(w, ChannelWriteTupleEntry):
            if w.value is PASSTHROUGH and not allow_passthrough:
                raise InvalidUpdateError("PASSTHROUGH value must be replaced")
    write: TYPE_SEND = config[CONF][CONFIG_KEY_SEND]
    write(_assemble_writes(writes))
```

两步验证：先检查不允许的写入（直接写 `TASKS`）和未替换的 `PASSTHROUGH`，然后从 config 中获取 `CONFIG_KEY_SEND` 函数（实际是 `deque.extend`）执行写入。`_assemble_writes()` 将各种写入条目统一转换为 `(channel, value)` 元组列表。

#### 鸭类型标记模式：register_writer() / is_writer()

```python
@staticmethod
def is_writer(runnable: Runnable) -> bool:
    return (isinstance(runnable, ChannelWrite)
            or getattr(runnable, "_is_channel_writer", MISSING) is not MISSING)

@staticmethod
def register_writer(runnable: R, static=None) -> R:
    object.__setattr__(runnable, "_is_channel_writer", static)
    return runnable
```

这是一个优雅的鸭类型模式。`ChannelWrite` 的实例自动被识别为写入器。其他类型的 Runnable 可以通过 `register_writer()` 被标记——只需设置 `_is_channel_writer` 属性。`object.__setattr__` 的使用是因为某些类（如 Pydantic 模型和 dataclass）会覆盖 `__setattr__`，需要绕过。

`is_writer()` 在 `PregelNode` 的 `copy()` 方法中被使用——区分写入器和其他 Runnable，确保它们被正确放置在 `writers` 列表中。

#### _assemble_writes() —— 写入组装

```python
def _assemble_writes(writes):
    tuples: list[tuple[str, Any]] = []
    for w in writes:
        if isinstance(w, Send):
            tuples.append((TASKS, w))
        elif isinstance(w, ChannelWriteTupleEntry):
            if ww := w.mapper(w.value):
                tuples.extend(ww)
        elif isinstance(w, ChannelWriteEntry):
            value = w.mapper(w.value) if w.mapper is not None else w.value
            if value is SKIP_WRITE:
                continue
            if w.skip_none and value is None:
                continue
            tuples.append((w.channel, value))
    return tuples
```

`Send` 对象被映射到 `TASKS` channel。`ChannelWriteTupleEntry` 通过 `mapper` 提取元组列表。`ChannelWriteEntry` 直接组装，支持 `SKIP_WRITE` 和 `skip_none` 两种跳过条件。

### _call.py —— 调用管线

#### identifier() —— 函数身份识别

```python
def identifier(obj, name=None) -> str | None:
    if isinstance(obj, PregelNode):
        obj = obj.bound
    if isinstance(obj, RunnableSeq):
        obj = obj.steps[0]
    if isinstance(obj, RunnableCallable):
        obj = obj.func
    if name is None:
        name = getattr(obj, "__qualname__", None)
    if name is None:
        name = getattr(obj, "__name__", None)
    if name is None:
        return None
    module_name = getattr(obj, "__module__", None)
    if module_name is None:
        return None
    return f"{module_name}.{name}"
```

`identifier()` 的设计灵感来自 `cloudpickle`。它逐层解包装：
1. `PregelNode` → 取 `bound`；
2. `RunnableSeq` → 取第一个 `step`；
3. `RunnableCallable` → 取底层 `func`。

然后组合 `__module__` 和 `__qualname__`（优先于 `__name__`），生成唯一标识符如 `mymodule.MyClass.my_method`。这用于缓存键和追踪。

#### _whichmodule() —— 模块归属查找

```python
def _whichmodule(obj, name):
    module_name = getattr(obj, "__module__", None)
    if module_name is not None:
        return module_name
    for module_name, module in sys.modules.copy().items():
        if module_name == "__main__" or module is None or not isinstance(module, types.ModuleType):
            continue
        try:
            if _getattribute(module, name)[0] is obj:
                return module_name
        except Exception:
            pass
    return None
```

当 `__module__` 为 `None` 时，遍历 `sys.modules` 查找对象归属。`sys.modules.copy()` 防止迭代期间模块动态导入导致的无限循环。

#### get_runnable_for_entrypoint() / get_runnable_for_task()

```python
def get_runnable_for_entrypoint(func) -> Runnable:
    key = (func, False)
    if key in CACHE:
        return CACHE[key]
    if is_async_callable(func):
        run = RunnableCallable(None, func, name=func.__name__, trace=False, recurse=False)
    else:
        afunc = functools.update_wrapper(
            functools.partial(run_in_executor, None, func), func
        )
        run = RunnableCallable(func, afunc, name=func.__name__, trace=False, recurse=False)
    if not _lookup_module_and_qualname(func):
        return run
    return CACHE.setdefault(key, run)
```

关键设计：同步函数的异步版本使用 `run_in_executor`（线程池包装）。`CACHE` 字典避免重复包装。`_lookup_module_and_qualname()` 确定函数是否可以被 pickle——不可 pickle 的函数不缓存（每次返回新实例）。

`get_runnable_for_task()` 类似，但额外包装了 `ChannelWrite([ChannelWriteEntry(RETURN)])`，将函数返回值写入 `RETURN` channel：

```python
seq = RunnableSeq(
    run,
    ChannelWrite([ChannelWriteEntry(RETURN)]),
    name=name,
    trace_inputs=functools.partial(_explode_args_trace_inputs, inspect.signature(func)),
)
```

`ChannelWriteEntry(RETURN)` 的 `value` 默认为 `PASSTHROUGH`，即从上游 `run` 的输出中取值。`trace_inputs` 使用 `inspect.signature` 解构参数用于追踪。

#### SyncAsyncFuture —— 双模式 Future

```python
class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T]):
    def __await__(self) -> Generator[T, None, T]:
        yield cast(T, ...)
```

这是本章最精巧的设计。`SyncAsyncFuture` 同时继承 `concurrent.futures.Future` 和实现 `__await__`，这意味着：
- 在同步上下文中，它就是一个标准的 `concurrent.futures.Future`——可以 `result()`、`add_done_callback()`；
- 在异步上下文中，它可以被 `await`——`__await__` 返回一个生成器。

`yield cast(T, ...)` 是一个技巧：`...`（省略号）作为 sent value，生成器立即暂停并返回一个占位值。当 `chain_future` 设置结果后，外部 `await` 解除阻塞。

#### call() 函数

```python
def call(func, *args, retry_policy=None, cache_policy=None, **kwargs) -> SyncAsyncFuture[T]:
    config = get_config()
    impl = config[CONF][CONFIG_KEY_CALL]
    fut = impl(func, (args, kwargs), retry_policy=retry_policy,
               cache_policy=cache_policy, callbacks=config["callbacks"])
    return fut
```

`call()` 是用户侧 API——在节点函数中调用其他函数。它从 `config` 中获取 `CONFIG_KEY_CALL`（在 `PregelRunner` 中注入的 `_call` 或 `_acall` 偏函数），将调用转换为 PUSH 任务调度。返回的 `SyncAsyncFuture` 可以同步 `result()` 或异步 `await`。

## 运行原理

### 完整数据流：从节点函数到 Channel 更新

```
  用户定义函数           包装层              组装层             注入层           执行层          写入层
  ─────────          ─────────          ─────────         ─────────       ─────────      ─────────
                                                                         
  def my_node(state): coerce_to_runnable  PregelNode       prepare_next_   run_with_     ChannelWrite
    result = ...    ──────────────►  ──────────────►   ──────────────►  ──────────►   ──────────►
    return result    RunnableCallable  bound + writers   tasks()注入config  retry()       _write()
                                      node cached_prop   CONFIG_KEY_SEND                do_write()
                                      flat_writers      CONFIG_KEY_READ                _assemble_
                                      input_cache_key   CONFIG_KEY_CALL                writes()
                                                                                      → deque.extend
```

### 三大设计模式

**模式 1：配置注入（send/read/call）**

`prepare_next_tasks` 在构建 `PregelExecutableTask` 时，将 `send`、`read`、`call` 函数注入到 `config` 中：

```python
configurable={
    CONFIG_KEY_SEND: writes.extend,    # deque.extend — 线程安全
    CONFIG_KEY_READ: partial(local_read, scratchpad, channels, managed, ...),
    CONFIG_KEY_CALL: partial(_call, weakref.ref(task), ...),
}
```

节点函数不需要知道这些函数的存在——它们通过 `ChannelRead.do_read(config, ...)`、`ChannelWrite.do_write(config, ...)` 和 `call(func, ...)` 间接访问。

**模式 2：PASSTHROUGH 哨兵延迟写入**

写入条目在图构建时创建（此时值未知），`value` 设为 `PASSTHROUGH`。执行时，`_write()` 将 `PASSTHROUGH` 替换为上游输出。这种延迟绑定允许写入条目在管道中自由组合，不受具体值的约束。

```
构建时：ChannelWriteEntry(channel="result", value=PASSTHROUGH)
执行时：→ ChannelWriteEntry(channel="result", value=42)  # PASSTHROUGH 被 input 替换
写入时：→ ("result", 42)  # _assemble_writes 生成最终元组
```

**模式 3：SyncAsyncFuture 双模式调用**

```
同步上下文：
  call(func, x) → SyncAsyncFuture → fut.result() → 阻塞等待子任务完成

异步上下文：
  result = await call(func, x) → SyncAsyncFuture.__await__() → 挂起协程 → 子任务完成后恢复
```

两种模式共享同一 `Future` 对象，消除了 sync/async 代码路径的分叉。

## 实现细节

### PregelNode.node cached_property 的工作原理

`node` 属性的组合逻辑优先考虑性能：
- 纯写入节点（`DEFAULT_BOUND` + 单 `writer`）：直接返回 `writer`，避免 `RunnableSeq` 包装；
- `DEFAULT_BOUND` + 多 `writers`：`RunnableSeq(*writers)`；
- 有 `bound` + `writers`：`RunnableSeq(bound, *writers)`；
- 只有 `bound`：直接返回 `bound`。

这种分情况优化避免了不必要的中间层。`cached_property` 确保组合只做一次。

### ChannelRead.do_read 的双模式处理

```python
if isinstance(select, str):
    # 单 channel 读取 → 返回原始值
    return read(select, fresh)
else:
    # 多 channel 读取 → 返回 dict
    return read(select, fresh)
```

`local_read()` 内部根据 `select` 类型分别处理：字符串只收集匹配 channel 的写入，列表额外处理 `managed_keys`。返回值类型也不同——单 channel 返回原始值，多 channel 返回字典。

### register_writer / is_writer 鸭类型模式

这个模式的价值在于开放扩展：任何第三方 Runnable 都可以被标记为写入器，无需继承 `ChannelWrite`。`get_static_writes()` 进一步支持静态分析——从标记中提取声明的写入 channel，使图在执行前就能确定数据流。

### SyncAsyncFuture 的双继承设计

`SyncAsyncFuture` 继承 `concurrent.futures.Future` 而非 `asyncio.Future`，因为：
- `concurrent.futures.Future` 在任何线程中都可用；
- `asyncio.Future` 必须与事件循环绑定；
- `__await__` 协议让同一个对象在两种上下文中都可用。

`_acall` 中通过 `chain_future(fut, destination)` 将子任务 Future 链接到目标 Future，确保结果传递。`_call` 中也使用 `chain_future`，保证 `commit()` 回调在返回 Future 解析之前执行，维护流式输出顺序。

### identifier() 的不同可调用类型处理

| 输入类型 | 解包策略 | 标识符示例 |
|---------|---------|----------|
| 普通函数 | 直接取 `__module__.__qualname__` | `mymodule.my_func` |
| 类方法 | `__qualname__` 包含类名 | `mymodule.MyClass.method` |
| Lambda | `__qualname__` 包含 `<lambda>` | `mymodule.MyClass.<lambda>` |
| 偏函数 | 无 `__qualname__`，回退到 `__name__` | 通常为 `None` |
| PregelNode | 解包到 `bound.func` | 底层函数的标识符 |

## 动手实验

### 实验 1：手动构建 PregelNode 并追踪执行

```python
from langgraph.pregel._read import PregelNode, ChannelRead, DEFAULT_BOUND
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langgraph._internal._runnable import RunnableCallable, RunnableSeq

# 创建一个简单的节点：读取 "state" channel，计算后写入 "result"
def transform(state):
    return state.get("state", "").upper()

# 构造写入器
writer = ChannelWrite([
    ChannelWriteEntry("result"),  # PASSTHROUGH — 值来自上游
])

# 构造 PregelNode
node = PregelNode(
    channels=["state"],
    triggers=["state"],
    bound=RunnableCallable(transform, None, name="transform"),
    writers=[writer],
)

# 检查 cached_property
print(f"flat_writers: {node.flat_writers}")  # [ChannelWrite<result>]
print(f"node type: {type(node.node)}")      # RunnableSeq
print(f"node: {node.node}")

# 验证 copy() 重置缓存
copy = node.copy({"triggers": ["state", "input"]})
print(f"Original triggers: {node.triggers}")  # ["state"]
print(f"Copy triggers: {copy.triggers}")       # ["state", "input"]
```

### 实验 2：实验 ChannelWrite 的 PASSTHROUGH 机制

```python
from langgraph.pregel._write import (
    ChannelWrite, ChannelWriteEntry, ChannelWriteTupleEntry,
    PASSTHROUGH, SKIP_WRITE, _assemble_writes
)
from langgraph.types import Send
from langgraph._internal._constants import TASKS

# 场景1：PASSTHROUGH 值在 _write 中被替换
entry1 = ChannelWriteEntry("output")  # value = PASSTHROUGH
print(f"Before: value is PASSTHROUGH? {entry1.value is PASSTHROUGH}")  # True

# 模拟 _write 的替换逻辑
input_value = {"key": "hello"}
replaced = ChannelWriteEntry(entry1.channel, input_value, entry1.skip_none, entry1.mapper)
print(f"After: value = {replaced.value}")  # {"key": "hello"}

# 场景2：_assemble_writes 处理不同类型
send = Send("other_node", "data")
tuples = _assemble_writes([send, ChannelWriteEntry("result", 42)])
print(f"Assembled writes: {tuples}")  # [("__tasks__", Send(...)), ("result", 42)]

# 场景3：SKIP_WRITE 和 skip_none
entry_skip = ChannelWriteEntry("x", SKIP_WRITE)
entry_none = ChannelWriteEntry("y", None, skip_none=True)
entry_pass = ChannelWriteEntry("z", None, skip_none=False)
result = _assemble_writes([entry_skip, entry_none, entry_pass])
print(f"Filtered writes: {result}")  # [("z", None)] — SKIP_WRITE 和 None 被过滤
```

### 实验 3：测试 SyncAsyncFuture 的双模式调用

```python
import concurrent.futures
from langgraph.pregel._call import SyncAsyncFuture

# 同步模式：作为标准 Future 使用
fut = SyncAsyncFuture[int]()
assert isinstance(fut, concurrent.futures.Future)

# 设置结果后获取
fut.set_result(42)
print(f"Sync result: {fut.result()}")  # 42

# 验证 __await__ 协议
print(f"Has __await__: {hasattr(SyncAsyncFuture, '__await__')}")  # True

# 异步模式演示（需要在 async 函数中运行）
async def async_demo():
    fut = SyncAsyncFuture[int]()
    # 在另一个线程中设置结果
    import threading
    def set_result():
        import time
        time.sleep(0.1)
        fut.set_result(99)
    threading.Thread(target=set_result, daemon=True).start()
    result = await fut
    print(f"Async result: {result}")  # 99

# 运行异步演示
import asyncio
asyncio.run(async_demo())
```

### 实验 4：观察 identifier() 对不同可调用类型的处理

```python
from langgraph.pregel._call import identifier

# 普通函数
def my_func(x):
    return x + 1

# 类方法
class MyClass:
    def my_method(self, x):
        return x * 2

# Lambda
my_lambda = lambda x: x - 1

print(f"Function: {identifier(my_func)}")     # __main__.my_func
print(f"Method: {identifier(MyClass.my_method)}")  # __main__.MyClass.my_method
print(f"Lambda: {identifier(my_lambda)}")     # __main__.<lambda>

# 不可识别的内置函数
print(f"Builtin: {identifier(len)}")           # None（builtins.len 但 __main__ 上下文）
```