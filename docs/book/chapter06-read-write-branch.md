# 第六章 读写与分支 — _read.py, _write.py, _call.py

Pregel 模型中，节点之间的通信全靠"读通道、写通道"来完成。本章走读三个核心文件：`_read.py` 定义了"谁来读、读什么"，`_write.py` 定义了"写到哪里、写什么"，`_call.py` 则把用户函数包装成可在 Pregel 内部调度的可运行对象。三者共同构成了图执行的数据流骨架。

---

## 6.1 Java 桥梁

| Java 概念 | Python 对应 | 说明 |
|-----------|------------|------|
| `@Override getter` + 缓存字段 | `functools.cached_property` | Java 中手动实现"首次访问计算、后续返回缓存"；Python 用装饰器一行搞定 |
| `static method` | `@staticmethod` | 两者语义相同：不接收隐式 self/this 参数 |
| `class method` | `@classmethod` | Python 类方法接收 `cls` 作为第一个参数；Java 无直接等价物，最接近的是工厂方法 |
| `@FunctionalInterface` | `ParamSpec` | Java 用函数式接口标注 Lambda 类型签名；Python `ParamSpec` 捕获可调用对象的参数规格 |
| `java.lang.reflect.Method` | `inspect` 模块 | Java 反射获取方法签名；Python `inspect.signature()` 可获取函数的完整参数信息 |
| `Supplier<T>` / memoization | `cached_property` / `property` | Java 需手写懒加载；Python `property` 每次调用都执行，`cached_property` 只计算一次 |
| `try-with-resources` | `__init_subclass__` | 无直接对应，但 `__init_subclass__` 是 Python 3.6+ 引入的类初始化钩子，在子类创建时自动调用 |

---

## 6.2 Python 概念速查

### cached_property vs property

```python
class Example:
    @property
    def computed(self):
        """每次访问都重新计算"""
        return expensive_calc()

    @cached_property
    def cached(self):
        """首次访问计算，后续返回缓存值"""
        return expensive_calc()
```

Java 等价写法需要手动实现：

```java
private volatile Object cached;
public Object getCached() {
    if (cached == null) {
        cached = expensiveCalc();
    }
    return cached;
}
```

Python 的 `cached_property` 是非线程安全的（与 Java 的双重检查锁不同），但在 Pregel 的单线程超步内使用是安全的。`PregelNode` 使用 `@cached_property` 来缓存 `node`、`flat_writers` 和 `input_cache_key`，因为它们在初始化后不会改变。

注意：`cached_property` 在 Python 3.8 之前是 `functools.cached_property`，3.8+ 已内置。LangGraph 使用 `from functools import cached_property`。

### classmethod vs staticmethod

```python
class ChannelWrite:
    @staticmethod
    def do_write(config, writes):
        """静态方法：不接收 self 或 cls，仅操作传入的参数"""

    @classmethod
    def register_writer(cls, runnable, static=None):
        """类方法：接收 cls，可用于返回同一类型的实例"""
        return runnable
```

Java 中 `static method` 大致对应 `@staticmethod`，但 Python 的 `@classmethod` 在 Java 中没有直接等价物——它接收的是类本身而非实例，常用于替代构造函数（工厂模式）。`ChannelWrite.is_writer()` 和 `register_writer()` 用 `@staticmethod` 因为它们不需要类引用，而 `register_writer()` 虽然是静态方法但巧妙地使用 `object.__setattr__` 绕过 Pydantic 的 `__setattr__` 限制。

### __init_subclass__

```python
class Base:
    def __init_subclass__(cls, **kwargs):
        """子类创建时自动调用，类似 Java 的注解处理器"""
        super().__init_subclass__(**kwargs)
        cls._registered_writers = []
```

`__init_subclass__` 在 Python 3.6 引入，允许基类在子类定义时自动执行逻辑。LangGraph 未直接使用此特性，但 `ChannelWrite.register_writer()` 使用 `object.__setattr__` 实现了类似的效果——给任意对象动态添加标记属性。

### 海象运算符（Walrus Operator）

```python
# Python 3.8+ 的赋值表达式
if (result := mapper(value)) is not None:
    process(result)
```

Java 没有直接等价物，需要先赋值再判断：

```java
Object result = mapper.apply(value);
if (result != null) {
    process(result);
}
```

`_write.py` 中大量使用海象运算符来简化"计算-判断-使用"模式。

### ParamSpec

```python
from typing_extensions import ParamSpec
P = ParamSpec("P")

def call(func: Callable[P, Awaitable[T]] | Callable[P, T],
         *args: P.args, **kwargs: P.kwargs) -> SyncAsyncFuture[T]:
```

`ParamSpec` 捕获可调用对象的完整参数签名（位置参数 + 关键字参数），在类型层面保证传入的 `args`/`kwargs` 与 `func` 的签名匹配。Java 没有等价物——函数式接口只能描述固定签名，而 `ParamSpec` 描述的是"与某个可调用对象相同的签名"。

### inspect 模块

```python
import inspect
sig = inspect.signature(func)
bound = sig.bind_partial(*args, **kwargs)
bound.apply_defaults()
```

Java 通过 `java.lang.reflect.Method` 获取参数信息；Python 的 `inspect` 模块更轻量，可以直接获取函数签名、绑定参数、检查是否为协程函数等。`_call.py` 使用 `inspect.signature()` 来提取任务函数的参数名，用于追踪（trace）输出。

---

## 6.3 代码走读

### 6.3.1 `_read.py` — PregelNode 与 ChannelRead

#### PregelNode 类

`PregelNode` 是图中"节点"的完整描述——不是可执行单元，而是**任务模板**，包含执行所需的一切信息：

```python
class PregelNode:
    channels: str | list[str]      # 从哪些 channel 读取输入
    triggers: list[str]            # 哪些 channel 更新会触发此节点
    bound: Runnable[Any, Any]      # 节点的核心逻辑（用户函数）
    writers: list[Runnable]        # 执行后写入 channel 的 Runnable 列表
    mapper: Callable | None        # 输入变换函数
    retry_policy: ...              # 重试策略
    cache_policy: ...              # 缓存策略
    subgraphs: Sequence[PregelProtocol]  # 子图引用
```

**Java 类比**：`PregelNode` 类似 Spring 的 `BeanDefinition`——它不直接执行，而是描述如何创建和配置一个执行单元。

#### cached_property: node

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

`node` 属性将 `bound`（核心逻辑）和 `writers`（写入链）组合为一条 `RunnableSeq` 执行链。逻辑分支处理了多种边界情况：

- 无逻辑无写入 → 不产生可运行对象
- 只有写入 → 直接返回写入器
- 有逻辑有写入 → 串联为序列

使用 `@cached_property` 而非 `@property` 是因为组合逻辑只需计算一次，重复执行 `RunnableSeq` 构造是浪费。

#### cached_property: flat_writers

```python
@cached_property
def flat_writers(self) -> list[Runnable]:
    writers = self.writers.copy()
    while (
        len(writers) > 1
        and isinstance(writers[-1], ChannelWrite)
        and isinstance(writers[-2], ChannelWrite)
    ):
        writers[-2] = ChannelWrite(
            writes=writers[-2].writes + writers[-1].writes,
        )
        writers.pop()
    return writers
```

优化逻辑：如果尾部有连续的 `ChannelWrite`，合并它们为一个——减少执行时的函数调用层级。注意 `writers.copy()` 保持不可变性：不修改原始列表，而是创建副本后操作。

#### ChannelRead

```python
class ChannelRead(RunnableCallable):
    channel: str | list[str]
    fresh: bool = False
    mapper: Callable | None = None

    def _read(self, _: Any, config: RunnableConfig) -> Any:
        return self.do_read(config, select=self.channel, ...)

    @staticmethod
    def do_read(config, *, select, fresh=False, mapper=None) -> Any:
        read: READ_TYPE = config[CONF][CONFIG_KEY_READ]
        return mapper(read(select, fresh)) if mapper else read(select, fresh)
```

`ChannelRead` 是一个 Runnable，通过配置注入的 `CONFIG_KEY_READ` 函数读取 channel 值。`do_read()` 是静态方法，既可以在 Runnable 模式下使用，也可以在命令式代码中直接调用——这种"双重入口"设计在 LangGraph 中反复出现，提供了声明式和命令式两种使用方式。

`fresh=True` 时会读取包含当前任务写入的最新值（通过 `_algo.local_read()`），否则读取已提交的稳定值。

### 6.3.2 `_write.py` — ChannelWrite 与写入条目

#### 数据结构

```python
class ChannelWriteEntry(NamedTuple):
    channel: str            # 目标 channel 名称
    value: Any = PASSTHROUGH  # 写入值，PASSTHROUGH 表示使用输入
    skip_none: bool = False   # 值为 None 时跳过
    mapper: Callable | None = None  # 值变换函数

class ChannelWriteTupleEntry(NamedTuple):
    mapper: Callable[[Any], Sequence[tuple[str, Any]] | None]  # 从输出提取写入
    value: Any = PASSTHROUGH
    static: ... | None       # 静态分析用的声明式写入描述
```

`ChannelWriteEntry` 描述单个 channel 的直接写入，`ChannelWriteTupleEntry` 描述通过 mapper 函数从输出动态计算多个写入。两者通过 `NamedTuple` 实现——不可变且内存高效，类似 Java 的 `record`（Java 16+）。

`PASSTHROUGH` 是一个哨兵对象（`object()`），表示"值从输入透传"。这比 Java 中常用的 `null` 哨兵更安全——`None` 可能是合法的业务值。

#### ChannelWrite

```python
class ChannelWrite(RunnableCallable):
    writes: list[ChannelWriteEntry | ChannelWriteTupleEntry | Send]

    def _write(self, input: Any, config: RunnableConfig) -> None:
        writes = [
            ChannelWriteEntry(w.channel, input, w.skip_none, w.mapper)
            if isinstance(w, ChannelWriteEntry) and w.value is PASSTHROUGH
            else ChannelWriteTupleEntry(w.mapper, input)
            if isinstance(w, ChannelWriteTupleEntry) and w.value is PASSTHROUGH
            else w
            for w in self.writes
        ]
        self.do_write(config, writes)
        return input  # 透传输入，支持链式调用
```

执行时，`PASSTHROUGH` 值被替换为实际输入，然后调用 `do_write()`。注意 `_write()` 返回 `input` 而非 `None`——这使得 `ChannelWrite` 可以嵌入 `RunnableSeq` 链中而不打断数据流。

#### do_write() 与 _assemble_writes()

```python
@staticmethod
def do_write(config, writes, allow_passthrough=True) -> None:
    # 校验
    for w in writes:
        if isinstance(w, ChannelWriteEntry):
            if w.channel == TASKS:
                raise InvalidUpdateError("Cannot write to the reserved channel TASKS")
    # 执行
    write: TYPE_SEND = config[CONF][CONFIG_KEY_SEND]
    write(_assemble_writes(writes))
```

`do_write()` 校验后调用配置注入的 `send` 函数。`_assemble_writes()` 将各类写入条目展平为 `list[tuple[str, Any]]`：

```python
def _assemble_writes(writes) -> list[tuple[str, Any]]:
    tuples = []
    for w in writes:
        if isinstance(w, Send):
            tuples.append((TASKS, w))           # Send → TASKS channel
        elif isinstance(w, ChannelWriteTupleEntry):
            if (ww := w.mapper(w.value)):        # 海象运算符
                tuples.extend(ww)
        elif isinstance(w, ChannelWriteEntry):
            value = w.mapper(w.value) if w.mapper else w.value
            if value is SKIP_WRITE: continue
            if w.skip_none and value is None: continue
            tuples.append((w.channel, value))
    return tuples
```

海象运算符 `ww := w.mapper(w.value)` 在计算的同时赋值，避免了重复调用 mapper。

#### is_writer() 与 register_writer()

```python
@staticmethod
def is_writer(runnable: Runnable) -> bool:
    return (
        isinstance(runnable, ChannelWrite)
        or getattr(runnable, "_is_channel_writer", MISSING) is not MISSING
    )

@staticmethod
def register_writer(runnable, static=None) -> R:
    object.__setattr__(runnable, "_is_channel_writer", static)
    return runnable
```

`is_writer()` 使用双重检测：先检查类型，再检查标记属性。`register_writer()` 用 `object.__setattr__` 绕过 Pydantic/dataclass 的 `__setattr__` 限制，给任意 Runnable 添加 `_is_channel_writer` 标记。这是一种**鸭子类型增强**——不改变类继承关系，但通过标记属性实现类型识别。

**Java 对比**：类似 Java 的 `Marker Interface`（如 `Serializable`），但不需要修改类定义，可以在运行时动态标记。

### 6.3.3 `_call.py` — 函数包装与任务调度桥接

#### SyncAsyncFuture

```python
class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T]):
    def __await__(self) -> Generator[T, None, T]:
        yield cast(T, ...)

def call(func, *args, retry_policy=None, cache_policy=None, **kwargs) -> SyncAsyncFuture[T]:
    config = get_config()
    impl = config[CONF][CONFIG_KEY_CALL]
    fut = impl(func, (args, kwargs), retry_policy=retry_policy, ...)
    return fut
```

`SyncAsyncFuture` 继承 `concurrent.futures.Future` 并实现 `__await__`，使同一个 Future 对象既可以同步调用 `.result()`，也可以 `await` 异步等待。这是 LangGraph 精心设计的**同步/异步统一接口**。

**Java 对比**：Java 中 `CompletableFuture` 天然支持 `join()`（同步）和 `thenApply`（异步），但无法在同一个方法中同时支持两种调用方式。Python 通过 `__await__` 协议让同一个对象在两种上下文中表现不同。

#### identifier() — 函数标识

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
    module_name = getattr(obj, "__module__", None)
    return f"{module_name}.{name}"
```

`identifier()` 逐层剥开包装（`PregelNode` → `RunnableSeq` → `RunnableCallable` → 原始函数），最终返回 `"module.function_name"` 格式的标识符。这类似 Java 的 `Class.getName()`，但需要手动解包，因为 Python 的 Runnable 包装层次比 Java 的代理模式更深。

#### get_runnable_for_entrypoint()

```python
def get_runnable_for_entrypoint(func) -> Runnable:
    if is_async_callable(func):
        run = RunnableCallable(None, func, name=func.__name__, ...)
    else:
        afunc = functools.update_wrapper(
            functools.partial(run_in_executor, None, func), func
        )
        run = RunnableCallable(func, afunc, name=func.__name__, ...)
    ...
```

将用户函数包装为 `RunnableCallable`。关键设计：同步函数会被自动包装一个异步版本（通过 `run_in_executor` 在线程池中执行同步代码），使 Pregel 引擎始终可以用 `await` 调用——无论用户代码是同步还是异步。

**Java 对比**：类似 `ExecutorService.submit(callable)` 将同步代码转为 `Future`，但 Python 的 `run_in_executor` 更透明——它将同步函数调度到线程池，返回可 await 的 Future。

#### get_runnable_for_task()

```python
def get_runnable_for_task(func) -> Runnable:
    ...
    seq = RunnableSeq(
        run,
        ChannelWrite([ChannelWriteEntry(RETURN)]),
        name=name,
        trace_inputs=functools.partial(
            _explode_args_trace_inputs, inspect.signature(func)
        ),
    )
    ...
```

与 `get_runnable_for_entrypoint` 不同，任务型 Runnable 额外串联了一个 `ChannelWrite([ChannelWriteEntry(RETURN)])`——自动将函数返回值写入 `RETURN` channel。同时，`trace_inputs` 使用 `inspect.signature()` 提取参数名，用于追踪输出——让调试信息显示 "调用 `agent(state, config)` " 而非 "调用 `func(*args)`"。

缓存机制：两个函数都使用模块级 `CACHE` 字典缓存已转换的 Runnable，避免重复包装。这类似 Java 中的 `ConcurrentHashMap.computeIfAbsent` 模式，但 Python 版本不涉及并发安全（模块初始化阶段是单线程的）。

---

## 6.4 运行原理

三个模块协作的数据流如下：

```
用户定义: def my_node(state): return {"messages": [...]}

   ┌─── get_runnable_for_task() ───┐
   │  包装为 RunnableSeq:           │
   │  [my_node, ChannelWrite(RETURN)]│
   └────────────────────────────────┘
              ↓
   ┌─── PregelNode 初始化 ──────────┐
   │  bound = my_node包装            │
   │  writers = [ChannelWrite(...)]  │
   │  node = cached_property:       │
   │    RunnableSeq(bound, *writers) │
   └────────────────────────────────┘
              ↓
   ┌─── 执行（由 PregelRunner 调度）───┐
   │  1. ChannelRead → 读取 channel   │
   │  2. bound → 执行用户函数         │
   │  3. ChannelWrite → 写入 channel  │
   │     └─ do_write() → send()      │
   │        └─ put_writes() → 持久化  │
   └─────────────────────────────────┘
```

关键设计模式总结：

1. **配置注入**：`ChannelRead` 通过 `CONFIG_KEY_READ` 读取状态，`ChannelWrite` 通过 `CONFIG_KEY_SEND` 写入状态，`call()` 通过 `CONFIG_KEY_CALL` 调度函数。三者都不是直接操作全局状态，而是通过配置注入的回调函数——这使得同一套组件可以在同步/异步、顶层/子图等不同上下文中工作。

2. **透传设计**：`ChannelWrite._write()` 返回 `input` 而非 `None`，使得写入器可以串联在 `RunnableSeq` 中而不阻断数据流。类似 Java 中的 `InterceptingFilter` 模式——过滤器执行副作用后传递原始请求。

3. **同步/异步统一**：`SyncAsyncFuture`、`RunnableCallable(同步func, 异步afunc)`、`run_in_executor` 等机制，使同一套逻辑在同步和异步上下文中均可用，且不牺牲性能。

---

## 6.5 动手实验

### 实验 1：cached_property vs property 行为差异

```python
from functools import cached_property

class Demo:
    call_count = 0

    @property
    def regular(self):
        self.call_count += 1
        return self.call_count

    @cached_property
    def cached(self):
        self.call_count += 1
        return self.call_count

d = Demo()
print(d.regular)  # 1
print(d.regular)  # 2 (每次重新计算)
print(d.cached)   # 3 (首次计算)
print(d.cached)   # 3 (返回缓存值)

# 观察 PregelNode.node 的行为：多次访问只计算一次
```

### 实验 2：PASSTHROUGH 哨兵模式

```python
from langgraph.pregel._write import PASSTHROUGH, SKIP_WRITE, ChannelWrite, ChannelWriteEntry

# PASSTHROUGH 让 ChannelWriteEntry 在执行时才绑定实际输入
entry = ChannelWriteEntry(channel="messages", value=PASSTHROUGH)
# 执行时: ChannelWriteEntry(channel="messages", value=actual_input)

# 对比 Java 中用 null 作哨兵的风险：null 可能是合法值
```

### 实验 3：register_writer() 标记模式

```python
from langgraph.pregel._write import ChannelWrite
from langchain_core.runnables import RunnableLambda

# 创建自定义 Runnable 并标记为写入器
my_writer = RunnableLambda(lambda x: x)
ChannelWrite.register_writer(my_writer, static=[(ChannelWriteEntry("out"), None)])

# 检测
print(ChannelWrite.is_writer(my_writer))  # True
print(ChannelWrite.is_writer(RunnableLambda(lambda x: x)))  # False

# 对比 Java: 需要实现标记接口 class MyWriter implements ChannelWriter
```

### 实验 4：SyncAsyncFuture 双模调用

```python
import asyncio
import concurrent.futures
from langgraph.pregel._call import SyncAsyncFuture

fut = SyncAsyncFuture()
fut.set_result("hello")

# 同步模式
print(fut.result())  # "hello"

# 异步模式
async def async_call():
    result = await fut
    print(result)

asyncio.run(async_call())  # "hello"

# 对比 Java: CompletableFuture.join() vs thenApply() 的分离模式
```