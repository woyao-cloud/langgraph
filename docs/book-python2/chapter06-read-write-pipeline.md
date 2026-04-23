# 第六章 读写管线——数据的流入与流出

> 源码文件：`pregel/_read.py`, `pregel/_write.py`, `pregel/_call.py`

## 6.1 功能概览

读写管线是 Pregel 引擎与节点函数之间的"神经末梢"。它解决三个核心问题：

1. **读**：节点函数如何获得当前状态？——通过 `ChannelRead` 从配置注入的读函数中读取
2. **写**：节点函数的返回值如何更新状态？——通过 `ChannelWrite` 将返回值转换为通道写入
3. **调用**：`@task` 函数如何与引擎交互？——通过 `call()` 函数提交任务，返回 `SyncAsyncFuture`

这三个模块共同构成了 LangGraph 的"数据管线"——从节点函数的输入到输出，每一步都有明确的机制和类型保障。

## 6.2 应用场景

### A. 理解你的节点函数如何接收状态

当你写 `def my_node(state: State) -> dict:` 时，`state` 参数不是凭空出现的。`ChannelRead` 负责从配置注入的读函数中提取当前通道值，组装成你定义的 `State` TypedDict：

```python
# 你写的代码
def analyze(state: State) -> dict:
    data = state["raw_data"]   # ChannelRead 从 "raw_data" 通道读取
    return {"result": process(data)}

# 内部发生的事（简化）：
# 1. ChannelRead.do_read(config, select=["raw_data", "result"])
# 2. 调用 config[CONF][CONFIG_KEY_READ](["raw_data", "result"])
# 3. 返回 {"raw_data": ..., "result": ...}
# 4. 你的函数接收到这个字典作为 state
```

### B. 节点返回值如何变成通道写入

当你 `return {"key": value}` 时，`ChannelWrite` 负责将这个字典转化为通道写入：

```python
# 你写的代码
def node_a(state: State) -> dict:
    return {"messages": ["新消息"], "count": state["count"] + 1}

# 内部发生的事：
# 1. _get_updates() 将返回值转为 [(channel, value)] 元组列表
# 2. ChannelWriteEntry("messages", value=["新消息"]) 
# 3. ChannelWriteEntry("count", value=新值)
# 4. ChannelWrite.do_write() 调用 config[CONF][CONFIG_KEY_SEND]()
# 5. 写入被收集，在超步结束后统一 apply_writes()
```

### C. StreamWriter 的工作原理

`stream_mode="custom"` 时，你用 `writer: StreamWriter` 写入自定义数据。`StreamWriter` 本质上是通过 `CONFIG_KEY_SEND` 直接写入 `__custom__` 通道：

```python
def my_node(state: State, *, writer: StreamWriter) -> dict:
    for chunk in expensive_computation():
        writer(chunk)  # 内部调用 config[CONF][CONFIG_KEY_SEND]([("__custom__", chunk)])
    return {"result": "done"}

# 流式输出时，客户端收到：
# {"type": "custom", "data": chunk1}
# {"type": "custom", "data": chunk2}
```

### D. @task 函数的通信机制

`@task` 函数通过 `CONFIG_KEY_CALL` 与引擎通信：

```python
@task
def fetch_url(url: str) -> str:
    # 调用 call() → config[CONF][CONFIG_KEY_CALL](func, args)
    # 引擎创建一个 PUSH 任务，异步执行
    return requests.get(url).text

@entrypoint()
def workflow(urls: list[str]) -> list[str]:
    futures = [fetch_url(url) for url in urls]  # 每个 call 返回 SyncAsyncFuture
    results = [f.result() for f in futures]     # 等待结果（并行执行）
    return results
```

### E. 条件边为什么需要同时读和写

`add_conditional_edges` 的路由函数需要读取当前状态（读），然后写入到目标分支通道（写）。这就是 `BranchSpec.run()` 同时需要 `reader` 和 `writer` 参数的原因：

```python
# 条件边内部流程
def route(state):
    if state["score"] > 0.5:
        return "approve"
    return "reject"

# 内部：ChannelRead 读取 state → route() 计算 → ChannelWrite 写入 "branch:to:approve"
```

### F. SyncAsyncFuture 实现并行执行

`SyncAsyncFuture` 同时继承 `concurrent.futures.Future` 和实现 `__await__`，使同一个对象既支持同步 `.result()` 又支持异步 `await`：

```python
# 同步使用
future = fetch_url("https://example.com")
result = future.result()  # 阻塞等待

# 异步使用
async def async_workflow():
    future = fetch_url("https://example.com")
    result = await future  # 非阻塞等待
```

### G. PASSTHROUGH——延迟写入模式

有些写入不能立即执行（比如条件边的写入需要先确定目标），必须通过 Runnable 管线传递。`PASSTHROUGH` 哨兵值标记这些"需要从上游输入中取值"的写入：

```python
# 当 value=PASSTHROUGH 时，写入不在 ChannelWrite 执行时立即完成
# 而是将上游输入作为值传递给通道
ChannelWriteEntry("branch:to:B", value=PASSTHROUGH)
# 执行时：input 被写入 "branch:to:B" 通道
```

## 6.3 Python 进阶

### @cached_property —— 延迟计算的属性

`functools.cached_property` 在首次访问时计算值并缓存，后续访问直接返回缓存。与 `@property` 不同，`@property` 每次访问都重新计算：

```python
from functools import cached_property

class PregelNode:
    @cached_property
    def node(self) -> Runnable:
        # 首次访问时组合 bound + writers → RunnableSeq
        # 后续访问直接返回缓存，不重新组合
        if self.writers:
            return RunnableSeq(self.bound, *self.flat_writers)
        return self.bound
```

`@cached_property` 要求类有 `__dict__`（不能与 `__slots__` 同时使用，除非显式包含 `__dict__`）。

### NamedTuple 带方法

`ChannelWriteEntry` 和 `ChannelWriteTupleEntry` 是 `NamedTuple` 子类，但也可以有默认值和方法：

```python
class ChannelWriteEntry(NamedTuple):
    channel: str
    value: Any = PASSTHROUGH      # 默认值
    skip_none: bool = False
    mapper: Callable | None = None
```

`NamedTuple` 的默认值通过 `__new__.__defaults__` 实现，语法和 `dataclass` 不同但效果类似。

### __init_subclass__ —— 子类注册钩子

`ChannelWrite` 使用 `__init_subclass__` + 标记属性实现鸭子类型的 writer 检测：

```python
class ChannelWrite(RunnableCallable):
    def __init_subclass__(cls, **kwargs):
        # 子类化时自动注册为 writer
        super().__init_subclass__(**kwargs)
        cls._is_writer = True

def is_writer(runnable):
    return getattr(runnable, '_is_writer', False)
```

这是一种无需继承特定基类就能标记"我是 writer"的技巧——比 `isinstance` 检查更灵活。

### walrus 运算符 (:=)

海象运算符同时赋值和判断，在条件表达式中特别有用：

```python
if (result := self.mapper(value)) is not None:
    channel, value = result
```

### ParamSpec —— 泛型函数签名

`ParamSpec` 捕获函数的参数签名（位置参数 + 关键字参数），用于泛型装饰器：

```python
P = ParamSpec("P")
T = TypeVar("T")

def task(func: Callable[P, T]) -> Callable[P, SyncAsyncFuture[T]]:
    # P 保留了原函数的参数签名，返回类型从 T 变为 SyncAsyncFuture[T]
    ...
```

### __or__ 运算符重载

`RunnableSeq` 通过 `__or__` 实现管道组合：

```python
runnable_a | runnable_b  # 等价于 RunnableSeq(runnable_a, runnable_b)
```

### inspect.signature —— 函数参数内省

`inspect.signature()` 返回函数的参数签名，`bind_partial()` 将参数绑定到签名：

```python
sig = inspect.signature(func)
bound = sig.bind_partial(*args, **kwargs)
bound.apply_defaults()
arguments = dict(bound.arguments)
```

## 6.4 代码走读

### PregelNode —— 执行单元

```python
class PregelNode:
    # 读配置
    channels: str | list[str]           # 订阅的通道名
    triggers: list[str]                 # 触发此节点的通道名
    mapper: Callable | None             # 输入转换函数
    fresh: bool = False                 # 是否读取最新值（含待写入）

    # 执行逻辑
    bound: Runnable                      # 核心执行逻辑
    writers: Sequence[Runnable] = ()     # 执行后的写入器

    # 策略
    retry_policy: Sequence[RetryPolicy] = ()
    cache_policy: CachePolicy | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @cached_property
    def node(self) -> Runnable:
        """组合 bound + writers 为单个 Runnable。"""
        if self.writers:
            return RunnableSeq(self.bound, *self.flat_writers)
        return self.bound

    @cached_property
    def flat_writers(self) -> list[Runnable]:
        """展平嵌套的 writers。"""
        ...
```

**核心设计**：`PregelNode` 将"读什么"、"触发什么"、"执行什么"、"写什么"封装为一个完整单元。`node` 属性用 `@cached_property` 延迟组合 `bound + writers`，避免每次执行都重新组合。

### ChannelRead —— 读取通道值

```python
class ChannelRead(RunnableCallable):
    channel: str | list[str]
    fresh: bool = False
    mapper: Callable | None = None

    def _read(self, _: Any, config: RunnableConfig) -> Any:
        return self.do_read(config, select=self.channel, fresh=self.fresh, mapper=self.mapper)

    @staticmethod
    def do_read(config, *, select, fresh=False, mapper=None) -> Any:
        read: READ_TYPE = config[CONF][CONFIG_KEY_READ]  # 从配置中提取读函数
        if isinstance(select, list):
            # 多通道：返回字典 {channel_name: value}
            values = read(select, fresh)
            if mapper:
                return mapper(values)
            return values
        else:
            # 单通道：直接返回值
            values = read([select], fresh)
            if mapper:
                return mapper(values[0])
            return values[0]
```

**双入口设计**：`select` 可以是 `str`（返回单值）或 `list[str]`（返回字典），让同一个 `ChannelRead` 适配两种使用模式。

### ChannelWrite —— 写入通道值

```python
PASSTHROUGH = object()  # 哨兵值：值来自上游输入

class ChannelWriteEntry(NamedTuple):
    channel: str
    value: Any = PASSTHROUGH    # PASSTHROUGH 表示值从输入中取
    skip_none: bool = False
    mapper: Callable | None = None

class ChannelWrite(RunnableCallable):
    writes: list[ChannelWriteEntry | ChannelWriteTupleEntry | Send]

    def _write(self, input: Any, config: RunnableConfig) -> None:
        writes = []
        for write in self.writes:
            if isinstance(write, ChannelWriteEntry) and write.value is PASSTHROUGH:
                # PASSTHROUGH：用上游输入替换值
                writes.append(ChannelWriteEntry(write.channel, input, write.skip_none, write.mapper))
            elif isinstance(write, ChannelWriteTupleEntry) and write.value is PASSTHROUGH:
                writes.append(ChannelWriteTupleEntry(write.mapper, input))
            else:
                writes.append(write)  # Send 或固定值，直接透传
        ChannelWrite.do_write(config, writes)  # 调用配置中的发送函数

    @staticmethod
    def do_write(config: RunnableConfig, writes: Sequence) -> None:
        send: TYPE_SEND = config[CONF][CONFIG_KEY_SEND]
        entries = ChannelWrite._assemble_writes(writes)
        if entries:
            send(entries)  # 批量发送所有写入
```

**PASSTHROUGH 的意义**：`ChannelWriteEntry("branch:to:B", PASSTHROUGH)` 在定义时不知道要写什么值——值由上游 Runnable 的输出决定。这在条件边场景中至关重要：路由结果决定写哪个通道，但写入本身需要通过 Runnable 管线传递。

### SyncAsyncFuture —— 同步/异步双模式 Future

```python
class SyncAsyncFuture(concurrent.futures.Future, Generic[T]):
    """同时支持 .result()（同步）和 await（异步）的 Future。"""

    def __init__(self, task_id: str, *, loop=None):
        super().__init__()
        self.task_id = task_id
        self._loop = loop

    def __await__(self, _=None):
        """支持 await future 语法。"""
        if not self.done():
            # 在事件循环中等待完成
            ...
        yield
        return self.result()
```

**双继承设计**：继承 `concurrent.futures.Future` 获得 `.result()` / `.set_result()` 等同步方法，实现 `__await__` 获得异步支持。这样 `@task` 函数返回的 Future 在同步和异步上下文中都能使用。

### identifier() —— 函数身份标识

```python
def identifier(obj: Any, name: str | None = None) -> str | None:
    """返回对象的模块.限定名，如 'mymodule.my_func'。"""
    # 层层解包：PregelNode → RunnableSeq → RunnableCallable → 原始函数
    if isinstance(obj, PregelNode):
        obj = obj.bound
    if isinstance(obj, RunnableSeq):
        obj = obj.steps[0]
    if isinstance(obj, RunnableCallable):
        obj = obj.func

    name = getattr(obj, "__qualname__", None)
    module_name = getattr(obj, "__module__", None)
    if name and module_name:
        return f"{module_name}.{name}"
    return None
```

**用途**：`identifier()` 为函数生成稳定的全局身份标识，用于缓存键（`CachePolicy`）和序列化引用（`serde`）。

### call() —— 提交任务执行

```python
def call(func, *args, retry_policy=None, cache_policy=None, **kwargs) -> SyncAsyncFuture:
    """通过 CONFIG_KEY_CALL 提交任务，返回 SyncAsyncFuture。"""
    config = get_config()
    call_fn = config[CONF][CONFIG_KEY_CALL]  # 从配置中提取调用函数
    return call_fn(func, (args, kwargs), retry_policy=retry_policy, cache_policy=cache_policy)
```

**配置注入模式**：`call()` 不直接执行函数，而是通过配置注入的 `call_fn` 提交给引擎。引擎决定何时、如何执行——可能是同步执行、可能是异步执行、可能加入重试和缓存。

## 6.5 实现原理

### 完整数据流

```
用户定义节点函数
       │
       ▼
coerce_to_runnable() ─── 包装为 RunnableCallable
       │
       ▼
PregelNode 组装 ─── bound=RunnableCallable, writers=[ChannelWrite, ...]
       │
       ▼
PregelNode.node (cached_property) ─── 组合为 RunnableSeq(bound, writer1, writer2, ...)
       │
       ▼
PregelLoop.tick() 执行 ─── 注入 config（send, read, call）
       │
       ├── ChannelRead.do_read(config) ─── 读取当前通道值 → 组装 state
       │
       ├── bound.invoke(state, config) ─── 执行用户函数 → 返回 dict
       │
       └── ChannelWrite.do_write(config) ─── 将返回值转为通道写入 → 提交
              │
              ▼
       apply_writes() ─── 在超步结束后统一应用所有写入
```

### 三个关键设计模式

1. **配置注入**：`send`/`read`/`call` 三个函数通过 `RunnableConfig` 注入，节点函数不需要知道引擎的存在。这是依赖倒置原则（DIP）的典型应用。

2. **PASSTHROUGH 哨兵**：用 `object()` 单例标记"值来自上游输入"，避免在定义时就绑定值。这使 `ChannelWrite` 可以作为 Runnable 管线的一部分。

3. **同步/异步统一**：`SyncAsyncFuture` 通过双继承实现同一对象的同步/异步使用，消除了 `@task` 装饰器需要分别处理 sync/async 函数的复杂性。

## 6.6 动手实验

```python
# 实验 1：观察 cached_property 行为
from functools import cached_property

class MyNode:
    call_count = 0

    @cached_property
    def computed(self):
        self.call_count += 1
        return f"result-{self.call_count}"

node = MyNode()
print(node.computed)  # "result-1" — 首次计算
print(node.computed)  # "result-1" — 缓存命中，不再计算
print(node.call_count)  # 1 — 只计算了一次

# 实验 2：PASSTHROUGH 哨兵模式
PASSTHROUGH = object()

class WriteEntry:
    def __init__(self, channel, value=PASSTHROUGH):
        self.channel = channel
        self.value = value

# 定义时不知道值
entry = WriteEntry("output", PASSTHROUGH)
assert entry.value is PASSTHROUGH

# 运行时用上游输入替换
def process_write(entry, upstream_input):
    if entry.value is PASSTHROUGH:
        return (entry.channel, upstream_input)
    return (entry.channel, entry.value)

print(process_write(entry, "hello"))  # ("output", "hello")
print(process_write(WriteEntry("fixed", "constant"), "ignored"))  # ("fixed", "constant")

# 实验 3：register_writer 鸭子类型标记
class Base:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._is_writer = True

class MyWriter(Base):
    pass

class NotWriter:
    pass

def is_writer(obj):
    return getattr(obj, '_is_writer', False)

print(is_writer(MyWriter()))    # True
print(is_writer(NotWriter()))   # False

# 实验 4：SyncAsyncFuture 双模式
import concurrent.futures

class SimpleSyncAsyncFuture(concurrent.futures.Future):
    def __await__(self, _=None):
        if not self.done():
            # 简化：同步等待
            pass
        yield
        return self.result()

fut = SimpleSyncAsyncFuture()
fut.set_result("hello")
print(fut.result())  # "hello" — 同步模式

# 实验 5：identifier 模式
def my_function():
    pass

print(f"{my_function.__module__}.{my_function.__qualname__}")
# "__main__.my_function" — 模块.限定名格式
```