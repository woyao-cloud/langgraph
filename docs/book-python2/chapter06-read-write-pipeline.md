# 第六章 读写管线——节点如何与引擎交换数据

> 源文件：`pregel/_read.py`、`pregel/_write.py`、`pregel/_call.py`

## 6.1 功能概览

在第五章中，我们看到了 Pregel 引擎如何通过超步循环来调度节点。但一个关键问题还没有回答：**节点函数到底是怎么读到状态的？它返回的字典又是怎么变成 Channel 写入的？** 读写管线就是回答这个问题的三个文件：

- **`_read.py`**——`PregelNode` 是节点的完整描述，包含读取哪些 Channel（`channels`）、被哪些 Channel 触发（`triggers`）、节点逻辑（`bound`）和写入列表（`writers`）。`ChannelRead` 是一个 Runnable，负责从配置中读取 Channel 值并注入到节点函数。
- **`_write.py`**——`ChannelWrite` 是一个 Runnable，负责将节点输出转化为 Channel 写入。它支持三种写入模式：直接写入（`ChannelWriteEntry`）、函数映射写入（`ChannelWriteTupleEntry`）和动态发送（`Send`）。`PASSTHROUGH` 哨兵值允许将节点的输入原样传递到输出。
- **`_call.py`**——`call()` 函数和 `SyncAsyncFuture` 类型是函数式 API 的核心。`call()` 允许节点内部动态调度子任务，返回一个既支持 `.result()` 同步等待又支持 `await` 异步等待的 Future 对象。`identifier()` 函数为节点生成稳定的标识符，用于缓存键和检查点。

如果把 Pregel 引擎比作 CPU，那么读写管线就是总线和寄存器——它们决定了数据如何从内存（Channel）加载到 ALU（节点函数），以及计算结果如何写回内存。

## 6.2 应用场景

### 场景 A：你的节点函数如何接收到状态

当你定义一个节点函数并传入 `StateGraph` 时：

```python
def my_node(state: dict) -> dict:
    # state 是从哪里来的？
    return {"messages": ["AI response"]}
```

答案在 `PregelNode` 和 `ChannelRead` 中。`PregelNode.channels` 指定了要读取的 Channel 列表。在 `prepare_single_task()` 中，引擎会调用 `_proc_input()` 读取 Channel 值：

```python
def _proc_input(proc, managed, channels, *, for_execution, scratchpad, input_cache):
    if isinstance(proc.channels, list):
        val = {}
        for chan in proc.channels:
            if chan in channels:
                if channels[chan].is_available():
                    val[chan] = channels[chan].get()
            else:
                val[chan] = managed[chan].get(scratchpad)
    elif isinstance(proc.channels, str):
        if proc.channels in channels:
            val = channels[proc.channels].get()
    # ...
    if proc.mapper:
        val = proc.mapper(val)
    return val
```

当你定义 `add_node("my_node", my_node_func)` 时，LangGraph 会根据 State 的注解自动设置 `channels` 和 `triggers`。对于 `TypedDict` 状态，`channels` 就是所有键的列表；对于 `Pydantic` 模型，同理。`mapper` 属性允许你在数据传递给节点函数之前进行转换，这在条件边编译时特别有用。

### 场景 B：你的节点返回值如何变成 Channel 写入

当你从节点函数返回一个字典：

```python
def my_node(state):
    return {"messages": ["AI: 你好"], "context": "greeting"}
```

这个字典会被 `ChannelWrite` 处理。在 `PregelNode.node` 的 `cached_property` 中，节点函数和写入器被组合成一个 `RunnableSeq`：

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

当你使用 `add_node("my_node", my_func)` 时，LangGraph 会自动创建一个 `ChannelWrite`，其中每个返回键对应一个 `ChannelWriteEntry`，`value=PASSTHROUGH` 表示"从输入中提取该键的值"。

### 场景 C：StreamWriter 的工作原理

`StreamWriter` 允许节点在执行过程中实时输出中间结果：

```python
def my_node(state, *, writer):
    writer("正在思考...")
    result = llm.invoke(state["messages"])
    writer("思考完成")
    return {"messages": [result]}
```

在内部，`StreamWriter` 通过 `CONFIG_KEY_SEND`（即 `writes.extend`）直接将数据写入 `task.writes` 队列。`task.writes` 是一个 `deque`，其 `extend` 操作是线程安全的，可以在并发环境中安全追加。每次写入都会被 `PregelRunner.commit()` 收集并提交到检查点。

关键细节：`CONFIG_KEY_SEND` 在 `prepare_single_task()` 中被设置为 `writes.extend`：

```python
configurable={
    CONFIG_KEY_SEND: writes.extend,  # deque.extend 是线程安全的
}
```

### 场景 D：@task 函数如何与引擎通信

LangGraph 的函数式 API 允许你使用 `@task` 装饰器定义子任务：

```python
@task
def my_task(x: int) -> int:
    return x * 2

def my_node(state):
    result = my_task(5).result()  # 同步等待
    return {"result": result}
```

`call()` 函数是底层机制。当你调用 `my_task(5)` 时，`call()` 从配置中获取 `CONFIG_KEY_CALL`，这是一个由 `_call()` 或 `_acall()` 提供的回调函数。该回调会调度一个新的 PUSH 任务，并返回一个 `SyncAsyncFuture`，可以同步等待（`.result()`）或异步等待（`await`）。

在 `get_runnable_for_task()` 中，`@task` 函数被包装为 `RunnableSeq(run, ChannelWrite([ChannelWriteEntry(RETURN)]))`——这意味着任务函数的返回值会被写入 `RETURN` 通道，供父任务读取。

### 场景 E：条件边为什么需要同时读写

条件边（conditional edge）本质上是一个读取状态并决定写入哪个 Channel 的函数：

```python
def route(state):
    if state["context"] == "greeting":
        return "chatbot"
    return "tool_node"
```

在内部，条件边被编译为一个 `PregelNode`，其中 `bound` 是路由函数本身，`channels` 指定了它需要读取的状态键，`writers` 包含一个 `ChannelWriteTupleEntry`，其 `mapper` 函数根据返回值决定写入哪个 Channel。

`ChannelWriteTupleEntry` 的 `static` 字段声明了可能的写入路径，用于静态分析（如构建图的拓扑结构），而 `mapper` 在运行时动态选择实际写入的路径。`PregelNode.flat_writers` 的 `cached_property` 会将连续的 `ChannelWrite` 合并为一个，减少 Runnable 调用链的深度。

### 场景 F：SyncAsyncFuture 如何实现同步/异步双模

`SyncAsyncFuture` 是一个同时继承 `concurrent.futures.Future` 和实现 `__await__` 的类型：

```python
class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T]):
    def __await__(self):
        yield  # 将控制权交给事件循环
```

这意味着它可以在同步代码中通过 `.result()` 获取结果，也可以在异步代码中通过 `await` 获取结果。`_call()` 函数在同步上下文中创建 `concurrent.futures.Future`，而 `_acall()` 在异步上下文中创建 `asyncio.Future`，两者都通过 `chain_future` 机制确保结果正确传播。

`_call()` 的核心逻辑中有一个重要的设计：当子任务已经执行过（例如重试场景），`next_task.writes` 不为空，此时直接从写入中提取结果，而不是重新执行：

```python
if next_task.writes:
    # 任务已经执行过，直接返回结果
    fut = concurrent.futures.Future()
    ret = next((v for c, v in next_task.writes if c == RETURN), MISSING)
    if ret is not MISSING:
        fut.set_result(ret)
```

### 场景 G：PASSTHROUGH 模式

`PASSTHROUGH = object()` 是一个哨兵值，表示"使用当前输入作为写入值"。这在条件边的编译中大量使用：

```python
# 节点返回 {"messages": [...], "context": "greeting"}
# ChannelWriteEntry 的 value=PASSTHROUGH 表示"从输入中提取 messages 键的值"
ChannelWriteEntry(channel="messages", value=PASSTHROUGH)
```

在 `ChannelWrite._write()` 中，PASSTHROUGH 被替换为实际值：

```python
writes = [
    ChannelWriteEntry(write.channel, input, write.skip_none, write.mapper)
    if isinstance(write, ChannelWriteEntry) and write.value is PASSTHROUGH
    else ChannelWriteTupleEntry(write.mapper, input)
    if isinstance(write, ChannelWriteTupleEntry) and write.value is PASSTHROUGH
    else write  # Send 对象或固定值，直接透传
    for write in self.writes
]
```

`PASSTHROUGH` 的设计使得同一个 `ChannelWrite` 可以处理不同类型的输入——无论节点返回什么，写入器都能正确提取目标字段的值。而 `SKIP_WRITE = object()` 是另一个哨兵值，当 `mapper` 函数返回 `SKIP_WRITE` 时，该写入条目会被跳过，不写入任何 Channel。

## 6.3 Python 进阶

### @cached_property

`PregelNode` 中有三个 `@cached_property`：

- `flat_writers`：将连续的 `ChannelWrite` 合并为一个，减少 Runnable 调用链的深度。
- `node`：将 `bound` 和 `writers` 组合为 `RunnableSeq`，惰性计算。
- `input_cache_key`：将 `mapper` 和 `channels` 元组化为缓存键，避免重复计算。

`@cached_property` 的特点是：首次访问时计算，之后缓存结果。与 `@property` 不同，它不会每次访问都重新计算。注意 `PregelNode.copy()` 方法会清除这些缓存：

```python
def copy(self, update):
    attrs = {**self.__dict__, **update}
    attrs.pop("flat_writers", None)  # 清除缓存
    attrs.pop("node", None)
    attrs.pop("input_cache_key", None)
    return PregelNode(**attrs)
```

### __slots__

`Call` 类使用 `__slots__` 来限制属性：

```python
class Call:
    __slots__ = ("func", "input", "retry_policy", "cache_policy", "callbacks")
```

这比使用 `__dict__` 更节省内存，且访问更快。对于在每次 `@task` 调用时都会创建的对象来说，这是有意义的优化。`Call` 对象只包含五元组：函数、参数、重试策略、缓存策略和回调。

### NamedTuple 带方法和默认值

`ChannelWriteEntry` 和 `ChannelWriteTupleEntry` 继承自 `NamedTuple`，同时拥有元组的不可变性和命名字段的可读性：

```python
class ChannelWriteEntry(NamedTuple):
    channel: str
    value: Any = PASSTHROUGH    # 默认值
    skip_none: bool = False      # 是否跳过 None
    mapper: Callable | None = None  # 值转换函数
```

`NamedTuple` 的默认值通过 `__new__.__defaults__` 实现，语法和 `dataclass` 不同但效果类似。`ChannelWriteTupleEntry` 用于条件边等需要动态决定写入目标的场景，其 `mapper` 函数将输入值转化为 `(channel, value)` 元组列表。

### classmethod vs staticmethod

`ChannelRead.do_read()` 和 `ChannelWrite.do_write()` 都是 `@staticmethod`，这意味着它们可以在不创建实例的情况下调用。这被 `local_read()` 等函数大量使用——它们需要读取或写入状态，但不需要持有 `ChannelRead` 或 `ChannelWrite` 的实例。

### register_writer 与鸭子类型标记

`ChannelWrite.register_writer()` 使用 `object.__setattr__` 来动态标记一个 Runnable 为写入器：

```python
@staticmethod
def register_writer(runnable, static=None):
    object.__setattr__(runnable, "_is_channel_writer", static)
    return runnable
```

使用 `object.__setattr__` 而不是 `setattr` 是为了绕过 Pydantic 模型和 dataclass 的 `__setattr__` 钩子。`is_writer()` 方法检查两种情况：

```python
@staticmethod
def is_writer(runnable):
    return (
        isinstance(runnable, ChannelWrite)
        or getattr(runnable, "_is_channel_writer", MISSING) is not MISSING
    )
```

`get_static_writes()` 方法从 `ChannelWriteTupleEntry.static` 和 `_is_channel_writer` 中提取静态写入声明，用于图编译时的拓扑分析。

### walrus 运算符

`_write.py` 中大量使用 `:=`（海象运算符）：

```python
if ww := w.mapper(w.value):
    tuples.extend(ww)
```

这使得 `mapper` 的返回值在条件判断和后续使用中只需计算一次。

### ParamSpec

`_call.py` 使用了 `ParamSpec` 来保持 `call()` 函数的类型签名：

```python
P = ParamSpec("P")
T = TypeVar("T")

def call(
    func: Callable[P, Awaitable[T]] | Callable[P, T],
    *args: P.args,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    **kwargs: P.kwargs,
) -> SyncAsyncFuture[T]:
```

这使得 `call(my_func, 1, 2, key="value")` 的类型检查与直接调用 `my_func(1, 2, key="value")` 一致。

### inspect 模块

`identifier()` 函数使用 `__qualname__` 和 `__module__` 来生成稳定的标识符。`_lookup_module_and_qualname()` 使用 `sys.modules` 来验证函数是否可以从模块中导入——如果不能（例如 lambda 函数或局部函数），则返回 `None`，表示该函数是动态的，不适合作为缓存键。

`_explode_args_trace_inputs()` 函数使用 `inspect.Signature` 和 `sig.bind_partial` 来解包函数参数，用于追踪和调试：

```python
def _explode_args_trace_inputs(sig, input):
    args, kwargs = input
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop("self", None)
    arguments.pop("cls", None)
```

### __or__ 运算符重载

虽然源码中没有直接重载 `__or__`，但 `RunnableSeq` 通过 LCEL（LangChain Expression Language）支持 `|` 运算符。`PregelNode.node` 返回的 `RunnableSeq(self.bound, *writers)` 就是这种组合的产物。

## 6.4 代码走读

### PregelNode：节点的完整描述

```python
class PregelNode:
    channels: str | list[str]       # 要读取的 Channel
    triggers: list[str]              # 触发此节点的 Channel
    mapper: Callable | None          # 输入映射函数
    writers: list[Runnable]          # 写入器列表
    bound: Runnable                  # 节点逻辑
    retry_policy: ...                # 重试策略
    cache_policy: ...                # 缓存策略
    tags: ...                        # 追踪标签
    metadata: ...                    # 追踪元数据
    subgraphs: ...                   # 子图列表
```

`PregelNode` 不直接执行——它是 `prepare_single_task()` 用来构建 `PregelExecutableTask` 的模板。关键方法：

- `flat_writers`（cached_property）：合并连续的 `ChannelWrite`。实现逻辑是从后往前扫描 `writers` 列表，将两个连续的 `ChannelWrite` 合并为一个新的 `ChannelWrite(writes=writers[-2].writes + writers[-1].writes)`。
- `node`（cached_property）：将 `bound` 和 `writers` 组合为可执行的 Runnable。如果 `bound` 是 `DEFAULT_BOUND`（即没有自定义逻辑），则只返回写入器。
- `copy(update)`：创建新实例，用于构建图时的不可变更新。它会清除所有 `cached_property` 缓存。

### ChannelRead：状态注入机制

```python
class ChannelRead(RunnableCallable):
    channel: str | list[str]    # 要读取的 Channel
    fresh: bool = False          # 是否读取最新值（包含当前任务的写入）
    mapper: Callable | None = None  # 映射函数

    def _read(self, _, config):
        return self.do_read(config, select=self.channel, fresh=self.fresh, mapper=self.mapper)

    @staticmethod
    def do_read(config, *, select, fresh=False, mapper=None):
        read = config[CONF][CONFIG_KEY_READ]  # 从配置中获取读取函数
        if mapper:
            return mapper(read(select, fresh))
        else:
            return read(select, fresh)
```

`ChannelRead` 不直接访问 Channel——它通过配置中的 `CONFIG_KEY_READ` 获取读取函数。这个读取函数是 `local_read`，在 `prepare_single_task()` 中注入。`fresh=True` 时会包含当前任务的未提交写入，这在条件边中用于读取"自己的输出"。

`get_name()` 方法根据 `channel` 类型生成描述性名称：

```python
def get_name(self, suffix=None, *, name=None):
    if isinstance(self.channel, str):
        name = f"ChannelRead<{self.channel}>"
    else:
        name = f"ChannelRead<{','.join(self.channel)}>"
    return super().get_name(suffix, name=name)
```

### ChannelWrite：写入管线

```python
class ChannelWrite(RunnableCallable):
    writes: list[ChannelWriteEntry | ChannelWriteTupleEntry | Send]

    def _write(self, input, config):
        # 1. 将 PASSTHROUGH 替换为实际输入值
        writes = [
            ChannelWriteEntry(write.channel, input, write.skip_none, write.mapper)
            if isinstance(write, ChannelWriteEntry) and write.value is PASSTHROUGH
            else ChannelWriteTupleEntry(write.mapper, input)
            if isinstance(write, ChannelWriteTupleEntry) and write.value is PASSTHROUGH
            else write  # Send 对象或固定值
            for write in self.writes
        ]
        self.do_write(config, writes)
        return input  # 写入后原样返回输入（管道模式）
```

关键点：

1. **PASSTHROUGH 替换**：`PASSTHROUGH` 哨兵值在运行时被替换为节点的输出值。这使得 `ChannelWriteEntry(channel="messages", value=PASSTHROUGH)` 可以从节点输出的 `{"messages": [...]}` 字典中提取值。

2. **do_write()**：从配置中获取 `CONFIG_KEY_SEND`（即 `deque.extend`），调用 `_assemble_writes()` 将写入条目转化为 `(channel, value)` 元组列表，然后提交。

3. **_assemble_writes()**：核心写入组装逻辑。`Send` 对象被转化为 `(TASKS, send)`，`ChannelWriteTupleEntry` 的 `mapper` 被调用并展开结果，`ChannelWriteEntry` 的 `mapper` 和 `skip_none` 被处理：

```python
def _assemble_writes(writes):
    tuples = []
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

4. **register_writer()**：使用 `object.__setattr__` 标记自定义 Runnable 为写入器。这允许 `is_writer()` 方法识别它们。

5. **do_write()** 中的验证：写入 `TASKS` 保留通道会抛出 `InvalidUpdateError`，`PASSTHROUGH` 值在非允许上下文中会抛出错误。

### SyncAsyncFuture：同步异步双模 Future

```python
class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T]):
    def __await__(self):
        yield  # 将控制权交给事件循环
```

这个类只定义了 `__await__` 方法，使其可以在 `async` 函数中使用。继承 `concurrent.futures.Future` 使其可以在同步代码中使用 `.result()`。`_call()` 函数在同步上下文中创建它，然后通过 `chain_future` 将实际的异步结果链接到它。

### call()：函数式 API 的调度核心

```python
def call(func, *args, retry_policy=None, cache_policy=None, **kwargs):
    config = get_config()
    impl = config[CONF][CONFIG_KEY_CALL]  # 获取调度函数
    fut = impl(
        func,
        (args, kwargs),
        retry_policy=retry_policy,
        cache_policy=cache_policy,
        callbacks=config["callbacks"],
    )
    return fut
```

`call()` 从运行配置中获取 `CONFIG_KEY_CALL`，这是在 `prepare_single_task()` 中注入的。对于同步任务，它是 `_call()`；对于异步任务，它是 `_acall()`。两者都会创建一个新的 PUSH 任务并通过 `schedule_task` 调度。

`_call()` 的核心逻辑：

```python
def _call(task, func, input, *, ...):
    scratchpad = task().config[CONF][CONFIG_KEY_SCRATCHPAD]
    if next_task := schedule_task(task(), scratchpad.call_counter(), Call(...)):
        if next_task.writes:
            # 任务已经执行过，直接返回结果
            fut = concurrent.futures.Future()
            ret = next((v for c, v in next_task.writes if c == RETURN), MISSING)
            if ret is not MISSING:
                fut.set_result(ret)
            elif exc := next((v for c, v in next_task.writes if c == ERROR), None):
                fut.set_exception(exc if isinstance(exc, BaseException) else Exception(exc))
            else:
                fut.set_result(None)
        else:
            # 提交新任务到线程池
            fut = submit()(run_with_retry, next_task, ...)
            SKIP_RERAISE_SET.add(fut)  # 异常由父任务处理
            futures()[fut] = next_task
    return chain_future(fut, concurrent.futures.Future())
```

关键细节：

- **`call_counter()`**：`PregelScratchpad` 的 `call_counter()` 确保同一节点内的多次 `call()` 调用产生不同的任务 ID。这使得子任务可以被唯一标识和缓存。
- **`SKIP_RERAISE_SET`**：子任务的异常应该在父任务中处理，而不是在 `_panic_or_proceed` 中全局抛出。这是一个 `WeakSet`，不会阻止 Future 被垃圾回收。
- **`chain_future`**：确保 `commit()` 回调在返回 Future 之前被调用，保证流式输出的顺序正确。
- **`__next_tick__=True`**：子任务在下一个超步中执行，确保当前超步的更新先提交。

### identifier()：稳定的节点标识

```python
def identifier(obj, name=None):
    if isinstance(obj, PregelNode):
        obj = obj.bound
    if isinstance(obj, RunnableSeq):
        obj = obj.steps[0]
    if isinstance(obj, RunnableCallable):
        obj = obj.func
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    module_name = getattr(obj, "__module__", None)
    if name and module_name:
        return f"{module_name}.{name}"
    return None
```

`identifier()` 逐层解包 `PregelNode -> bound -> RunnableSeq.steps[0] -> RunnableCallable.func`，最终返回 `module.qualname` 格式的标识符。如果函数是动态创建的（如 lambda 或局部函数），`_lookup_module_and_qualname()` 返回 `None`，标识符为 `None`，此时使用 `"__dynamic__"` 作为缓存键的回退。

这个标识符被用于 `CachePolicy` 的缓存键生成：

```python
cache_key = CacheKey(
    (CACHE_NS_WRITES, (identifier(proc) or "__dynamic__"), name),
    xxh3_128_hexdigest(args_key.encode() if isinstance(args_key, str) else args_key),
    cache_policy.ttl,
)
```

### get_runnable_for_task：@task 函数的包装

```python
def get_runnable_for_task(func):
    key = (func, True)
    if key in CACHE:
        return CACHE[key]
    # ...
    seq = RunnableSeq(
        run,
        ChannelWrite([ChannelWriteEntry(RETURN)]),  # 写入 RETURN 通道
        name=name,
        trace_inputs=functools.partial(_explode_args_trace_inputs, inspect.signature(func)),
    )
```

`get_runnable_for_task()` 将 `@task` 函数包装为 `RunnableSeq(run, ChannelWrite([ChannelWriteEntry(RETURN)]))`，其中 `RETURN` 是一个特殊通道，用于将子任务的返回值传递给父任务。`trace_inputs` 参数使用 `functools.partial` 将 `inspect.signature(func)` 绑定到 `_explode_args_trace_inputs`，用于在追踪时记录函数参数。

## 6.5 实现原理

### 配置注入：read、send、call 的完整数据流

```
prepare_single_task()
  |
  +-- CONFIG_KEY_READ = partial(local_read, scratchpad, channels, managed, task_writes)
  |     |
  |     +-- 节点内部通过 config[CONF][CONFIG_KEY_READ] 读取状态
  |         ChannelRead.do_read() 调用此函数
  |
  +-- CONFIG_KEY_SEND = writes.extend (deque 的线程安全方法)
  |     |
  |     +-- 节点内部通过 config[CONF][CONFIG_KEY_SEND] 追加写入
  |         StreamWriter 内部使用此方法
  |
  +-- CONFIG_KEY_CALL = partial(_call, ...)
        |
        +-- 节点内部通过 call() 函数调度子任务
            @task 装饰器返回的 SyncAsyncFuture 内部使用此函数
```

这三个注入点使得节点函数完全不需要知道 Pregel 引擎的存在。它们通过 `RunnableConfig` 传递，由 `patch_config()` 和 `merge_configs()` 注入到节点的执行环境中。`local_read()` 函数特别巧妙——它接收当前任务的写入作为参数，使得 `fresh=True` 模式可以读取"自己的写入"：

```python
def local_read(scratchpad, channels, managed, task, select, fresh=False):
    updated = defaultdict(list)
    if isinstance(select, str):
        for c, v in task.writes:
            if c == select:
                updated[c].append(v)
    else:
        for c, v in task.writes:
            if c in select:
                updated[c].append(v)
    if fresh:
        # 应用写入到副本，返回最新值
        local_channels = {k: channels[k].copy() for k in channels}
        for k in local_channels:
            local_channels[k].update(updated[k])
        values = read_channels(local_channels, select)
    else:
        # 读取原始值
        values = read_channels(channels, select)
    return values
```

### PASSTHROUGH 哨兵值的完整流程

```
节点函数返回 {"messages": ["hello"], "context": "greeting"}
  |
  v
ChannelWrite._write(input={"messages": ["hello"], "context": "greeting"})
  |
  +-- ChannelWriteEntry(channel="messages", value=PASSTHROUGH)
  |     +-- 替换为 ChannelWriteEntry(channel="messages", value=["hello"])
  |
  +-- ChannelWriteEntry(channel="context", value=PASSTHROUGH)
  |     +-- 替换为 ChannelWriteEntry(channel="context", value="greeting")
  |
  +-- ChannelWriteTupleEntry(mapper=route_fn, value=PASSTHROUGH)
        +-- 替换为 ChannelWriteTupleEntry(mapper=route_fn, value={"messages": [...], "context": ...})
```

PASSTHROUGH 的设计使得同一个 `ChannelWrite` 可以处理不同类型的输入——无论节点返回什么，写入器都能正确提取目标字段的值。注意 `ChannelWrite._write()` 在写入后返回 `input`（不是写入结果），这是 Runnable 管道模式的要求——每个步骤传递输入到下一步。

### 同步/异步统一

`PregelNode` 同时提供 `invoke`/`ainvoke` 和 `stream`/`astream` 方法，通过 `RunnableCallable` 的 `func`/`afunc` 双模式实现。`ChannelRead` 和 `ChannelWrite` 也都有同步和异步版本（`_read`/`_aread` 和 `_write`/`_awrite`），但它们的逻辑完全相同，只是异步包装不同。

`_call.py` 中的 `_call()` 和 `_acall()` 是两个并行实现：

- `_call()`：同步版本，使用 `concurrent.futures.Future` 和 `run_with_retry`，通过 `submit()` 提交到线程池
- `_acall()`：异步版本，使用 `asyncio.Future` 和 `arun_with_retry`，通过 `submit()` 提交到事件循环

两者共享 `schedule_task` 和 `futures` 参数，通过 `CONFIG_KEY_CALL` 注入到不同的执行上下文中。

`_acall()` 有一个额外的逻辑——它使用 `run_coroutine_threadsafe()` 来处理在同步上下文中调用异步函数的情况：

```python
async def _acall_impl(destination, task, func, input, *, ...):
    # ...
    if next_task := await schedule_task(task(), scratchpad.call_counter(), Call(...)):
        # 创建 asyncio.Future 并设置结果或异常
        if next_task.writes:
            fut = asyncio.Future(loop=loop)
            ret = next((v for c, v in next_task.writes if c == RETURN), MISSING)
            # ...
        else:
            # 调度新任务
            fut = cast(asyncio.Future, submit()(arun_with_retry, next_task, ...))
    if fut is not None:
        chain_future(fut, destination)
```

### 完整数据流图

```
用户代码: graph.invoke({"messages": ["hello"]})
  |
  v
PregelLoop._first() ─── 映射输入到 Channel
  |
  v
PregelLoop.tick() ─── prepare_next_tasks()
  |                      |
  |                      +-- _proc_input() ─── 读取 Channel 值
  |                      |      |
  |                      |      +-- ChannelRead.do_read() ─── 从 config 获取 read 函数
  |                      |
  |                      +-- 创建 PregelExecutableTask
  |                             |
  |                             +-- config[CONFIG_KEY_READ] = local_read(...)
  |                             +-- config[CONFIG_KEY_SEND] = writes.extend
  |                             +-- config[CONFIG_KEY_CALL] = _call(...)
  |
  v
PregelRunner.tick() ─── 执行 task.node
  |                          |
  |                          +-- bound.invoke(input, config)  ─── 节点函数执行
  |                          |      |
  |                          |      +-- 可能调用 config[CONF][CONFIG_KEY_READ]
  |                          |      +-- 可能调用 config[CONF][CONFIG_KEY_SEND]
  |                          |      +-- 可能调用 call() ─── config[CONF][CONFIG_KEY_CALL]
  |                          |
  |                          +-- writers[i].invoke(output, config)  ─── 写入管线
  |                                   |
  |                                   +-- ChannelWrite.do_write() ─── 写入到 deque
  |
  v
PregelRunner.commit() ─── put_writes() ─── 持久化写入
  |
  v
PregelLoop.after_tick() ─── apply_writes() ─── 应用到 Channel
```

## 6.6 动手实验

### 实验 1：追踪 PregelNode 执行

```python
"""模拟 PregelNode 的 node cached_property 如何组合 bound 和 writers"""

PASSTHROUGH = object()

class SimpleChannelWrite:
    """简化版 ChannelWrite"""
    def __init__(self, writes):
        self.writes = writes  # [(channel, value_or_passthrough)]

    def invoke(self, input_data, config=None):
        results = []
        for channel, value in self.writes:
            if value is PASSTHROUGH:
                # 从输入中提取对应键的值
                if isinstance(input_data, dict) and channel in input_data:
                    actual_value = input_data[channel]
                else:
                    actual_value = input_data
                results.append((channel, actual_value))
            else:
                results.append((channel, value))
        return results

class SimpleRunnable:
    """简化版 bound"""
    def __init__(self, func):
        self.func = func

    def invoke(self, input_data, config=None):
        return self.func(input_data)

# 模拟节点
def my_node(state):
    return {"messages": ["AI: " + state.get("input", "")], "step": state.get("step", 0) + 1}

bound = SimpleRunnable(my_node)
writers = [SimpleChannelWrite([("messages", PASSTHROUGH), ("step", PASSTHROUGH)])]

# 执行链
input_data = {"input": "你好", "step": 0}
bound_result = bound.invoke(input_data)
print(f"bound 输出: {bound_result}")
# {'messages': ['AI: 你好'], 'step': 1}

write_result = writers[0].invoke(bound_result)
print(f"写入结果: {write_result}")
# [('messages', ['AI: 你好']), ('step', 1)]

# 模拟 flat_writers 的合并逻辑
# 如果有两个连续的 ChannelWrite，它们会合并为一个
write_a = SimpleChannelWrite([("messages", PASSTHROUGH)])
write_b = SimpleChannelWrite([("step", PASSTHROUGH)])
# 合并后：SimpleChannelWrite([("messages", PASSTHROUGH), ("step", PASSTHROUGH)])
merged = SimpleChannelWrite(write_a.writes + write_b.writes)
print(f"合并后的写入: {merged.invoke(bound_result)}")
# [('messages', ['AI: 你好']), ('step', 1)]
```

### 实验 2：测试 ChannelWrite 和 PASSTHROUGH

```python
"""测试 ChannelWrite 的 PASSTHROUGH、SKIP_WRITE 和 skip_none 行为"""

PASSTHROUGH = object()
SKIP_WRITE = object()

class ChannelWriteEntry:
    """简化版 ChannelWriteEntry"""
    def __init__(self, channel, value=PASSTHROUGH, skip_none=False, mapper=None):
        self.channel = channel
        self.value = value
        self.skip_none = skip_none
        self.mapper = mapper

class ChannelWriteTupleEntry:
    """简化版 ChannelWriteTupleEntry"""
    def __init__(self, mapper, value=PASSTHROUGH):
        self.mapper = mapper
        self.value = value

def assemble_writes(entries, input_value):
    """模拟 _assemble_writes"""
    # 第一步：将 PASSTHROUGH 替换为实际输入值
    resolved = []
    for entry in entries:
        if isinstance(entry, ChannelWriteEntry) and entry.value is PASSTHROUGH:
            resolved.append(ChannelWriteEntry(entry.channel, input_value,
                                              entry.skip_none, entry.mapper))
        elif isinstance(entry, ChannelWriteTupleEntry) and entry.value is PASSTHROUGH:
            resolved.append(ChannelWriteTupleEntry(entry.mapper, input_value))
        else:
            resolved.append(entry)

    # 第二步：组装写入元组
    tuples = []
    for entry in resolved:
        if isinstance(entry, ChannelWriteTupleEntry):
            if ww := entry.mapper(entry.value):
                tuples.extend(ww)
        elif isinstance(entry, ChannelWriteEntry):
            value = entry.mapper(entry.value) if entry.mapper else entry.value
            if value is SKIP_WRITE:
                continue
            if entry.skip_none and value is None:
                continue
            tuples.append((entry.channel, value))
    return tuples

# 测试 1: PASSTHROUGH 从字典中提取值
result = assemble_writes([
    ChannelWriteEntry("messages", PASSTHROUGH),
    ChannelWriteEntry("context", PASSTHROUGH),
], {"messages": ["hello"], "context": "greeting"})
print(f"测试 1 (PASSTHROUGH): {result}")
# [('messages', ['hello']), ('context', 'greeting')]

# 测试 2: skip_none 过滤 None 值
result = assemble_writes([
    ChannelWriteEntry("messages", PASSTHROUGH),
    ChannelWriteEntry("optional", PASSTHROUGH, skip_none=True),
], {"messages": ["hello"], "optional": None})
print(f"测试 2 (skip_none): {result}")
# [('messages', ['hello'])]

# 测试 3: mapper 转换值
result = assemble_writes([
    ChannelWriteEntry("count", PASSTHROUGH, mapper=lambda x: len(x)),
], {"count": ["a", "b", "c"]})
print(f"测试 3 (mapper): {result}")
# [('count', 3)]

# 测试 4: ChannelWriteTupleEntry 条件写入
def route_mapper(value):
    if value.get("score", 0) > 0.5:
        return [("approve", value)]
    return [("reject", value)]

result = assemble_writes([
    ChannelWriteTupleEntry(route_mapper, PASSTHROUGH),
], {"score": 0.8})
print(f"测试 4 (条件写入-approve): {result}")
# [('approve', {'score': 0.8})]

result = assemble_writes([
    ChannelWriteTupleEntry(route_mapper, PASSTHROUGH),
], {"score": 0.3})
print(f"测试 4 (条件写入-reject): {result}")
# [('reject', {'score': 0.3})]
```

### 实验 3：测试 SyncAsyncFuture 和 chain_future

```python
"""模拟 SyncAsyncFuture 的同步等待和异步等待行为"""
import concurrent.futures

class SyncAsyncFuture(concurrent.futures.Future):
    """简化版 SyncAsyncFuture"""
    def __await__(self):
        result = self.result()
        yield
        return result

# 测试 1: 同步使用
fut = SyncAsyncFuture()
fut.set_result(42)
print(f"同步等待: {fut.result()}")  # 42

# 测试 2: chain_future 模式
def chain_future(source, destination):
    """当 source Future 完成时，将结果传递到 destination Future"""
    def callback(future):
        try:
            result = future.result()
            destination.set_result(result)
        except Exception as e:
            destination.set_exception(e)
    source.add_done_callback(callback)
    return destination

inner = concurrent.futures.Future()
outer = concurrent.futures.Future()
chain_future(inner, outer)
inner.set_result("task completed")
print(f"链式 Future: {outer.result()}")  # task completed

# 测试 3: 异常传播
inner_exc = concurrent.futures.Future()
outer_exc = concurrent.futures.Future()
chain_future(inner_exc, outer_exc)
inner_exc.set_exception(ValueError("test error"))
try:
    outer_exc.result()
except ValueError as e:
    print(f"异常传播: {e}")  # test error

# 测试 4: identifier 模式
def my_function():
    pass

module_name = getattr(my_function, "__module__", None)
qualname = getattr(my_function, "__qualname__", None)
identifier = f"{module_name}.{qualname}" if module_name and qualname else None
print(f"函数标识符: {identifier}")

# lambda 没有稳定标识
lam = lambda x: x + 1
lam_identifier = f"{getattr(lam, '__module__', None)}.{getattr(lam, '__qualname__', None)}"
print(f"Lambda 标识符: {lam_identifier}")
# 注意: lambda 的 __qualname__ 包含 '<locals>'，不适合作为稳定标识
```

这些实验帮助你理解读写管线的核心：节点函数如何通过配置注入获取读取和写入能力，`PASSTHROUGH` 如何将节点输出映射到 Channel 写入，以及 `SyncAsyncFuture` 如何统一同步和异步的任务调度模式。掌握了这些，你就能自如地构建自定义的节点、边和中间件。