# 第八章：函数式 API — func/ 模块

## 8.1 概述

LangGraph 提供了两种构建图的 API：面向对象的 StateGraph API（显式定义节点和边）和函数式 API（用 `@task` 和 `@entrypoint` 装饰器声明工作流）。本章走读 `func/__init__.py` 中的函数式 API 实现，以及 `pregel/_call.py` 中支撑其运行的 `SyncAsyncFuture` 和 `get_runnable_for_entrypoint()`。函数式 API 的本质是：**将普通 Python 函数转换为一个单节点的 Pregel 图**。

## 8.2 Java 桥梁

如果你熟悉 Spring 的注解驱动开发，可以这样类比：

- `@task` 类似 `@Service` 或 `@Component`——标注一个函数为可管理的任务单元，支持重试策略和缓存策略。
- `@entrypoint` 类似 `@SpringBootApplication`——将一个函数变成整个工作流的入口，底层自动构建出一个 Pregel 实例。
- `SyncAsyncFuture` 类似 `CompletableFuture<T>`——调用 `@task` 得到一个 Future，可以同步 `.result()` 或异步 `await` 获取结果。

但有一个根本区别：Spring 的注解在编译期或类加载时由容器处理，而 Python 的装饰器在模块加载时立即执行，直接修改或替换被装饰的对象。

## 8.3 Python 概念速查

| 概念 | Python 用法 | Java 对应 |
|------|-----------|----------|
| 装饰器模式 | `@decorator` 语法，本质是 `func = decorator(func)` | 注解处理器（Annotation Processor），编译期生成代码 |
| `functools.update_wrapper` | 将原始函数的 `__name__`、`__doc__` 等属性复制到包装器 | Java 无直接对应；代理模式中手动委托方法签名 |
| `ParamSpec` 和 `TypeVar` | `P = ParamSpec("P")`、`T = TypeVar("T")` 用于泛型装饰器的类型签名 | Java 的 `<P, T>` 泛型类型参数 |
| `@overload` | 为同一函数提供多个类型签名（仅用于类型检查） | Java 的方法重载（真正的多方法） |
| 装饰器类 vs 装饰器函数 | 类实现 `__call__` 即可做装饰器 | Java 注解可标注在类上，但行为由处理器决定 |
| `__call__` 方法 | 使实例可调用：`obj(*args)` | 实现 `Callable<V>` 接口的 `call()` 方法 |
| `inspect.signature` | 运行时获取函数的参数签名 | Java 的 `Method.getParameterTypes()` 反射 |
| `dataclass` | `@dataclass` 自动生成 `__init__`、`__repr__` 等 | Lombok `@Data` 或 Java `record` |

**重点：装饰器是运行时变换，不是编译时标注**

Java 的 `@Override` 或 `@Transactional` 是元数据标注，实际行为由框架在运行时或编译期通过代理/AOP 注入。Python 的 `@task` 则在模块加载时立即执行：

```python
@task
def my_func(x: int) -> int:
    return x + 1

# 等效于：
def my_func(x: int) -> int:
    return x + 1
my_func = task(my_func)  # my_func 现在是 _TaskFunction 实例，不再是原函数
```

调用 `my_func(1)` 实际调用的是 `_TaskFunction.__call__()`，返回的是 `SyncAsyncFuture[int]`，而不是 `int`。

## 8.4 代码走读

### 8.4.1 @task 装饰器 — _TaskFunction 类

`_TaskFunction` 是 `@task` 装饰器返回的核心类：

```python
class _TaskFunction(Generic[P, T]):
    def __init__(
        self,
        func: Callable[P, Awaitable[T]] | Callable[P, T],
        *,
        retry_policy: Sequence[RetryPolicy],
        cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
        name: str | None = None,
    ) -> None:
        if name is not None:
            if hasattr(func, "__func__"):
                # 处理类方法：创建绑定的 partial
                instance_method = functools.partial(func.__func__, func.__self__)
                instance_method.__name__ = name
                func = instance_method
            else:
                func.__name__ = name
        self.func = func
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        functools.update_wrapper(self, func)  # 复制函数元数据
```

**`functools.update_wrapper(self, func)`** 的作用：将 `func` 的 `__name__`、`__module__`、`__doc__`、`__annotations__` 等属性复制到 `self`。Java 中没有直接对应——Java 的代理类无法自动继承被代理方法的 Javadoc 或参数名。Python 的这一机制确保了装饰后的函数在调试和文档工具中看起来和原函数一样。

**`__call__` 方法**——使 `_TaskFunction` 实例可以像函数一样被调用：

```python
def __call__(self, *args: P.args, **kwargs: P.kwargs) -> SyncAsyncFuture[T]:
    return call(
        self.func,
        retry_policy=self.retry_policy,
        cache_policy=self.cache_policy,
        *args,
        **kwargs,
    )
```

调用 `@task` 装饰的函数时，不会立即执行函数体，而是返回一个 `SyncAsyncFuture`。这个 Future 的实际执行由 PregelLoop 调度——通过 `config[CONF][CONFIG_KEY_CALL]` 获取执行器来提交任务。

**Java 对比**：这类似于提交任务到 `ExecutorService`：

```java
// Java 等效
ExecutorService executor = ...;
Future<Integer> future = executor.submit(() -> myFunc(x));
// future.get() 获取结果
```

Python 中则更简洁：

```python
future = my_task(x)      # 返回 SyncAsyncFuture
result = future.result() # 同步获取，或 await future 异步获取
```

### 8.4.2 @task 装饰器函数 — 三重重载

`task()` 函数使用 `@overload` 提供了三种类型签名：

```python
@overload
def task(__func_or_none__: None = None, *, name=..., retry_policy=..., cache_policy=...)
    -> Callable[[Callable[P, Awaitable[T]] | Callable[P, T]], _TaskFunction[P, T]]: ...

@overload
def task(__func_or_none__: Callable[P, Awaitable[T]]) -> _TaskFunction[P, T]: ...

@overload
def task(__func_or_none__: Callable[P, T]) -> _TaskFunction[P, T]: ...

def task(__func_or_none__=None, *, name=None, retry_policy=None, cache_policy=None, **kwargs):
    # 实际实现
```

**Python `@overload` vs Java 方法重载**：这是 Java 开发者最容易混淆的概念。Java 的重载是真正的多方法——编译器根据参数类型选择调用哪个方法。Python 的 `@overload` 只是类型提示，对运行时行为毫无影响，仅服务于 mypy/pyright 等类型检查器。运行时只有最后一个 `def task(...)` 会被执行。

这种模式用于处理装饰器的两种用法：

```python
# 用法1：无参数装饰器
@task
def my_func(x): ...

# 用法2：带参数装饰器
@task(retry_policy=RetryPolicy(max_attempts=5))
def my_func(x): ...
```

**`__func_or_none__` 参数名**：双下划线前缀是 Python 的约定，表示"这是内部实现细节"。当 `@task` 不带括号使用时，Python 自动将被装饰的函数作为第一个参数传入；当带括号使用时，第一次调用返回装饰器函数，第二次调用才传入被装饰函数。

### 8.4.3 SyncAsyncFuture — 同步异步两用 Future

```python
class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T]):
    def __await__(self) -> Generator[T, None, T]:
        yield cast(T, ...)
```

`SyncAsyncFuture` 同时继承 `concurrent.futures.Future` 和实现 `__await__`，使得同一个 Future 对象既可以 `.result()` 同步获取结果，也可以 `await future` 异步获取。

**`__await__` 的实现**：`yield cast(T, ...)` 看起来很奇怪。这个 yield 是一个占位符——实际的值由 PregelLoop 的执行器设置到 Future 中。`__await__` 使对象成为可等待对象（awaitable），配合 `await` 关键字使用。

**Java 对比**：Java 的 `CompletableFuture<T>` 天然支持 `future.get()`（同步）和 `future.thenApply()`（异步回调），但没有 Python 这种统一的 `await` 语法。Python 的 `await` 是语言级关键字，编译器会自动将协程函数转换为状态机。

### 8.4.4 call() 函数 — 任务提交

```python
def call(
    func: Callable[P, Awaitable[T]] | Callable[P, T],
    *args: Any,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    **kwargs: Any,
) -> SyncAsyncFuture[T]:
    config = get_config()
    impl = config[CONF][CONFIG_KEY_CALL]
    fut = impl(
        func, (args, kwargs),
        retry_policy=retry_policy,
        cache_policy=cache_policy,
        callbacks=config["callbacks"],
    )
    return fut
```

`call()` 从当前运行配置中获取 `CONFIG_KEY_CALL` 指向的执行器，将函数、参数和策略打包提交。这个执行器由 PregelLoop 注入——在循环外调用 `@task` 函数会因找不到执行器而报错。

**Python ContextVar 的隐式传播**：`get_config()` 从当前上下文获取配置，这个配置是通过 `ContextVar` 在 PregelLoop 启动时设置的，会自动传播到所有子任务和 `await` 点。Java 中要实现类似效果，需要显式传递 `ExecutionContext` 对象或使用 `ThreadLocal`。

### 8.4.5 @entrypoint 装饰器 — entrypoint 类

`entrypoint` 不是一个函数，而是一个**类**：

```python
class entrypoint(Generic[ContextT]):
    def __init__(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        cache: BaseCache | None = None,
        context_schema: type[ContextT] | None = None,
        cache_policy: CachePolicy | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> None:
```

**装饰器类 vs 装饰器函数**：`@entrypoint(checkpointer=InMemorySaver())` 的执行分为两步：

1. `entrypoint(checkpointer=InMemorySaver())` ——调用 `__init__`，构造一个 `entrypoint` 实例。
2. 将该实例作为装饰器应用到函数上 ——调用 `__call__`。

```python
@entrypoint(checkpointer=InMemorySaver())
def my_workflow(x: int) -> int:
    return x + 1

# 等效于：
decorator_instance = entrypoint(checkpointer=InMemorySaver())
my_workflow = decorator_instance(my_workflow)  # 调用 __call__
# my_workflow 现在是 Pregel 实例
```

**Java 对比**：Java 的注解只能携带常量属性，不能携带运行时对象（如 `InMemorySaver()` 实例）。Python 装饰器的参数可以是任意表达式，这给予了极大的灵活性。

### 8.4.6 entrypoint.__call__() — 构建单节点 Pregel 图

这是函数式 API 的核心——将一个函数转换为一个完整的 Pregel 图：

```python
def __call__(self, func: Callable[..., Any]) -> Pregel:
    bound = get_runnable_for_entrypoint(func)
    stream_mode: StreamMode = "updates"

    sig = inspect.signature(func)
    first_parameter_name = next(iter(sig.parameters.keys()), None)
    if not first_parameter_name:
        raise ValueError("Entrypoint function must have at least one parameter")
    input_type = (
        sig.parameters[first_parameter_name].annotation
        if sig.parameters[first_parameter_name].annotation is not inspect.Signature.empty
        else Any
    )
```

`inspect.signature(func)` 在运行时解析函数签名，获取参数名和类型注解。Java 中等效的是 `Method.getParameterTypes()` 和 `Method.getGenericParameterTypes()`，但 Java 的类型擦除使得运行时获取泛型类型更加困难。

**Channel 架构**：

```python
graph: Pregel[Any, ContextT, Any, Any] = Pregel(
    nodes={
        func.__name__: PregelNode(
            bound=bound,
            triggers=[START],
            channels=START,
            writers=[
                ChannelWrite([
                    ChannelWriteEntry(END, mapper=_pluck_return_value),
                    ChannelWriteEntry(PREVIOUS, mapper=_pluck_save_value),
                ])
            ],
        )
    },
    channels={
        START: EphemeralValue(input_type),
        END: LastValue(output_type, END),
        PREVIOUS: LastValue(save_type, PREVIOUS),
    },
    input_channels=START,
    output_channels=END,
    stream_channels=END,
    ...
)
```

入口函数被包装成一个单节点 Pregel 图，使用三个 Channel：

- **START**（`EphemeralValue`）：输入通道。`EphemeralValue` 表示"一次性"的值——被消费后不再保留，不会出现在检查点中。这确保了每次调用使用新的输入，不会与上次的输入混淆。
- **END**（`LastValue`）：输出通道。`LastValue` 只保留最新值。
- **PREVIOUS**（`LastValue`）：保存值通道。用于 `entrypoint.final` 场景——允许返回值和持久化值不同。

### 8.4.7 entrypoint.final — 解耦返回值与保存值

```python
@dataclass(**_DC_KWARGS)
class final(Generic[R, S]):
    value: R    # 返回给调用者的值
    save: S     # 保存到检查点的值
```

`entrypoint.final` 是一个嵌套在 `entrypoint` 类中的 dataclass。它解决了"函数的返回值和需要持久化的状态不一致"的问题。

典型场景：工作流返回用户友好的结果，但检查点保存内部状态供下次调用使用。

```python
@entrypoint(checkpointer=InMemorySaver())
def my_workflow(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    return entrypoint.final(value=previous, save=2 * number)

# 第一次调用：number=3, previous=None → 返回 0, 保存 6
# 第二次调用：number=1, previous=6 → 返回 6, 保存 2
```

**`_pluck_return_value` 和 `_pluck_save_value`** 这两个内部函数在写入 Channel 时根据是否为 `entrypoint.final` 实例来分流：

```python
def _pluck_return_value(value: Any) -> Any:
    return value.value if isinstance(value, entrypoint.final) else value

def _pluck_save_value(value: Any) -> Any:
    return value.save if isinstance(value, entrypoint.final) else value
```

**Python `isinstance` vs Java `instanceof`**：两者语义相同，但 Python 的 `isinstance` 支持元组类型检查：`isinstance(obj, (ClassA, ClassB))`。

### 8.4.8 get_runnable_for_entrypoint() — 包装函数为 Runnable

```python
def get_runnable_for_entrypoint(func: Callable[..., Any]) -> Runnable:
    key = (func, False)
    if key in CACHE:
        return CACHE[key]
    else:
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

关键细节：

1. **同步函数转为异步兼容**：对于同步函数 `func`，创建 `afunc = functools.partial(run_in_executor, None, func)`，将其包装为在线程池中执行的异步版本。这样 Pregel 的异步执行循环可以统一处理。
2. **`functools.update_wrapper(afunc, func)`**：确保异步包装器保留了原函数的名称和文档。
3. **缓存**：`CACHE.setdefault(key, run)` 避免重复创建 Runnable——对同一个函数多次调用 `get_runnable_for_entrypoint()` 只会创建一次。
4. **`_lookup_module_and_qualname`**：检查函数是否可以被模块名+限定名定位（而非动态创建的 lambda），决定是否缓存。

## 8.5 运行原理：函数式 API vs StateGraph API

### 对比表

| 维度 | 函数式 API (`@entrypoint`) | StateGraph API |
|------|--------------------------|----------------|
| 图结构 | 单节点 Pregel 图 | 多节点 Pregel 图 |
| 边的定义 | 无显式边（START → 唯一节点 → END） | 显式 `add_edge()`、`add_conditional_edges()` |
| 并行性 | 通过 `@task` 的 Future 实现并行 | 通过 `Send` 在条件边中实现并行 |
| 状态管理 | 函数参数 + `previous` 关键字参数 | TypedDict/Pydantic 的 Channel 系统 |
| 返回值 | 直接 return，或 `entrypoint.final` | 写入 Channel，由 `output_channels` 决定输出 |
| 底层产物 | `Pregel` 实例 | `Pregel` 实例 |

**鸭式类型（Duck Typing）**：两种 API 最终都产出 `Pregel` 实例。调用方不需要关心图是如何构建的——只要它支持 `.invoke()`、`.stream()`、`.ainvoke()` 等方法即可。Python 的鸭式类型使得这种多态无需共同接口：如果一个对象走起来像鸭子、叫起来像鸭子，那它就是鸭子。

Java 中要实现相同效果，需要定义 `interface Graph { ... }` 并让两种构建方式都实现它。Python 省去了这个步骤——`Pregel` 本身就是共同的类型。

### 函数式 API 的并行执行

```python
@task
def slow_computation(x: int) -> int:
    import time; time.sleep(1)
    return x * 2

@entrypoint()
def my_workflow(numbers: list[int]) -> list[int]:
    # 三个 task 并行执行，总耗时约 1 秒而非 3 秒
    futures = [slow_computation(n) for n in numbers]
    results = [f.result() for f in futures]
    return results
```

`@task` 的调用返回 `SyncAsyncFuture`，PregelLoop 会将这些任务提交到执行器并行运行。`f.result()` 会阻塞直到结果就绪。这与 Java 中 `CompletableFuture.allOf(futures).join()` 的模式类似，但 Python 的 `@task` 由 PregelLoop 统一管理生命周期、重试和缓存。

## 8.6 动手实验

### 实验 1：@task 基本用法

```python
import time
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

@task
def add_one(x: int) -> int:
    print(f"  执行 add_one({x})")
    return x + 1

@entrypoint(checkpointer=InMemorySaver())
def add_numbers(numbers: list[int]) -> list[int]:
    futures = [add_one(n) for n in numbers]
    results = [f.result() for f in futures]
    return results

config = {"configurable": {"thread_id": "test-1"}}
result = add_numbers.invoke([1, 2, 3], config)
print("结果:", result)  # [2, 3, 4]
```

### 实验 2：entrypoint.final — 解耦返回值和保存值

```python
@entrypoint(checkpointer=InMemorySaver())
def counter(n: int, *, previous: int = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    print(f"  输入={n}, 上次保存={previous}")
    return entrypoint.final(value=previous, save=previous + n)

config = {"configurable": {"thread_id": "counter-1"}}
print(counter.invoke(10, config))   # 返回 0 (previous=None), 保存 10
print(counter.invoke(5, config))    # 返回 10 (previous=10), 保存 15
print(counter.invoke(3, config))    # 返回 15 (previous=15), 保存 18
```

### 实验 3：函数式 API 与 interrupt 配合

```python
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

@task
def generate_draft(topic: str) -> str:
    return f"关于{topic}的草稿内容"

@entrypoint(checkpointer=InMemorySaver())
def review_workflow(topic: str) -> dict:
    draft_future = generate_draft(topic)
    draft = draft_future.result()
    feedback = interrupt({"草稿": draft, "问题": "请审核"})
    return {"草稿": draft, "反馈": feedback}

config = {"configurable": {"thread_id": "review-1"}}

# 第一次执行——被中断
for event in review_workflow.stream("人工智能", config):
    print("事件:", event)

# 恢复执行——传入人工审核
for event in review_workflow.stream(Command(resume="内容不错，通过"), config):
    print("结果:", event)
```

注意：恢复时 `generate_draft` 不会重新执行——因为 `@task` 的结果被检查点缓存。这就是函数式 API 与 interrupt 配合的强大之处：节点整体重执行，但 `@task` 的结果从缓存读取。

### 实验 4：对比 StateGraph 和函数式 API 的产物类型

```python
from langgraph.graph import StateGraph, START, END
from langgraph.func import entrypoint
from langgraph.pregel import Pregel

# StateGraph 方式
builder = StateGraph(dict)
builder.add_node("node", lambda x: {"result": x.get("input", 0) + 1})
builder.add_edge(START, "node")
builder.add_edge("node", END)
sg_graph = builder.compile()

# 函数式 API 方式
@entrypoint()
def my_entry(x: int) -> int:
    return x + 1

# 两者都是 Pregel 实例
print(type(sg_graph))   # <class 'langgraph.pregel.Pregel'>
print(type(my_entry))   # <class 'langgraph.pregel.Pregel'>
print(isinstance(sg_graph, Pregel))  # True
print(isinstance(my_entry, Pregel))  # True
```

### 思考题

1. `@task` 装饰器返回的 `_TaskFunction` 实例不再是原函数——如果第三方代码通过 `isinstance(my_task, FunctionType)` 检查它是否为函数，会发生什么？`functools.update_wrapper` 能否解决这个问题？
2. `entrypoint.__call__` 使用 `EphemeralValue` 作为 START Channel——如果改用 `LastValue`，在启用检查点后连续两次调用入口函数时，第二次的输入会是什么？
3. `SyncAsyncFuture` 同时支持 `.result()` 和 `await`——如果在同步上下文中调用异步入口函数的 `@task`，会发生什么？