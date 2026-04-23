# 第八章 函数式 API — func/ 模块

## 1. 功能概览

LangGraph 的函数式 API（`func/` 模块）提供了一种无需构建图拓扑即可定义工作流的方式。它围绕两个核心装饰器构建：

- **`@entrypoint`**：定义工作流的入口函数，自动将其包装为 Pregel 图。支持检查点、状态管理、中断恢复等全部 Pregel 特性
- **`@task`**：定义可并行执行的任务单元，调用时返回 `SyncAsyncFuture`，支持重试策略和缓存策略

函数式 API 的设计哲学是"能简单则简单"——当你不需要复杂的有向图拓扑时，用 `@entrypoint` 直接定义入口函数即可。它与 `StateGraph` 共享底层 Pregel 引擎，但省去了手动添加节点和边的样板代码。

核心类型关系：
- `@entrypoint` 将函数转为 `Pregel` 实例（单节点图）
- `@task` 将函数转为 `_TaskFunction` 实例（调用时提交到 Pregel 执行队列）
- `entrypoint.final` 允许返回值与保存值分离

## 2. 应用场景

### 场景 A：简单管道——无需图拓扑

当你只需要"输入 -> 处理 -> 输出"的线性流程时，`@entrypoint` 远比 `StateGraph` 简洁：

```python
from langgraph.func import entrypoint

@entrypoint()
def translate(text: str) -> str:
    """简单的翻译管道，无需构建 StateGraph"""
    # 预处理
    cleaned = text.strip().lower()
    # 翻译（简化示意）
    result = f"[翻译结果] {cleaned}"
    return result

# 直接调用——返回的是 Pregel 图
translate.invoke("Hello World")
# 输出: '[翻译结果] hello world'
```

### 场景 B：并行任务执行

`@task` 的核心价值——轻松实现并行：

```python
from langgraph.func import entrypoint, task
import time

@task
def fetch_user(user_id: int) -> dict:
    time.sleep(1)  # 模拟 API 调用
    return {"id": user_id, "name": f"user_{user_id}"}

@task
def fetch_orders(user_id: int) -> list:
    time.sleep(1)  # 模拟 API 调用
    return [{"order_id": i, "user_id": user_id} for i in range(3)]

@entrypoint()
def user_dashboard(user_id: int) -> dict:
    # 两个 task 并行执行，总耗时约1秒而非2秒
    user_future = fetch_user(user_id)
    orders_future = fetch_orders(user_id)
    return {
        "user": user_future.result(),
        "orders": orders_future.result(),
    }

user_dashboard.invoke(42)
```

异步版本同样简洁：

```python
@task
async def async_fetch(url: str) -> dict:
    # 异步 HTTP 请求
    return {"url": url, "data": "..."}

@entrypoint()
async def async_pipeline(urls: list[str]) -> list[dict]:
    futures = [async_fetch(url) for url in urls]
    # 所有请求并行发出
    import asyncio
    return await asyncio.gather(*futures)

await async_pipeline.ainvoke(["https://a.com", "https://b.com"])
```

### 场景 C：结合 @task 与 interrupt 实现异步人工审核

```python
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

@task
def generate_report(data: str) -> str:
    return f"基于 {data} 生成的报告"

@entrypoint(checkpointer=InMemorySaver())
def review_pipeline(data: str) -> dict:
    report_future = generate_report(data)
    report = report_future.result()
    # 中断等待人工审核
    review = interrupt({"report": report, "question": "请审核报告"})
    return {"report": report, "review": review}

config = {"configurable": {"thread_id": "review-1"}}
# 第一次执行——生成报告后中断
review_pipeline.invoke("销售数据", config)

# 人工审核后恢复——generate_report 不会重新执行（结果已缓存）
review_pipeline.invoke(Command(resume="报告内容准确，批准发布"), config)
```

### 场景 D：entrypoint.final 分离返回值与保存值

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

@entrypoint(checkpointer=InMemorySaver())
def accumulator(number: int, *, previous: int = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    # 返回给调用者的是 previous（历史累计值）
    # 保存到检查点的是 previous + number（新累计值）
    return entrypoint.final(value=previous, save=previous + number)

config = {"configurable": {"thread_id": "acc-1"}}
print(accumulator.invoke(3, config))   # 0 （首次 previous 为 None）
print(accumulator.invoke(5, config))   # 3 （上次保存了 0+3=3）
print(accumulator.invoke(2, config))   # 8 （上次保存了 3+5=8）
```

典型用途：返回摘要给用户，但保存完整数据到检查点：

```python
@entrypoint(checkpointer=InMemorySaver())
def process_document(doc: str, *, previous: dict = None) -> entrypoint.final[str, dict]:
    full_result = {"doc": doc, "analysis": "...", "metadata": {...}}
    summary = f"处理完成：{doc}"
    # 用户看到摘要，检查点保存完整数据
    return entrypoint.final(value=summary, save=full_result)
```

### 场景 E：@entrypoint 与 StateGraph 的选择

| 特性 | `@entrypoint` | `StateGraph` |
|------|--------------|-------------|
| 定义方式 | 单函数 + 装饰器 | 添加节点 + 边 |
| 适用场景 | 线性/简单并行流 | 复杂有向图拓扑 |
| 状态管理 | `previous` 参数 | 多 channel + reducer |
| 并行任务 | `@task` 返回 Future | `Send()` 消息 |
| 中断支持 | 完整支持 | 完整支持 |
| 代码量 | 少 | 多 |
| 灵活性 | 有限 | 高 |

选择原则：能用 `@entrypoint` 解决的问题不要用 `StateGraph`，需要复杂拓扑时再用 `StateGraph`。

### 场景 F：@task 配合 retry_policy 处理不可靠操作

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

@task(retry_policy=RetryPolicy(
    max_attempts=3,           # 最多重试3次
    initial_interval=1.0,    # 首次重试间隔1秒
    backoff_factor=2.0,      # 间隔倍增
    max_interval=10.0,       # 最大间隔10秒
))
def call_external_api(endpoint: str) -> dict:
    # 可能因网络问题失败的操作
    import requests
    resp = requests.get(endpoint, timeout=5)
    resp.raise_for_status()
    return resp.json()

@entrypoint()
def fetch_data(endpoints: list[str]) -> list[dict]:
    futures = [call_external_api(ep) for ep in endpoints]
    return [f.result() for f in futures]
```

### 场景 G：@task 配合 cache_policy 缓存计算结果

```python
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

@task(cache_policy=CachePolicy(ttl=300))  # 缓存5分钟
def expensive_computation(params: str) -> str:
    # 耗时计算
    import hashlib
    return hashlib.sha256(params.encode()).hexdigest()

@entrypoint(cache=InMemoryCache())
def compute(inputs: list[str]) -> list[str]:
    futures = [expensive_computation(p) for p in inputs]
    return [f.result() for f in futures]

# 相同参数的第二次调用会命中缓存
compute.invoke(["data1", "data2"])
compute.invoke(["data1", "data3"])  # "data1" 命中缓存
```

### 场景 H：previous 参数实现有状态的多轮工作流

```python
from typing import Optional
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

@entrypoint(checkpointer=InMemorySaver())
def chatbot(message: str, *, previous: Optional[list] = None) -> list:
    history = previous or []
    history.append({"role": "user", "content": message})
    # 模拟 AI 回复
    history.append({"role": "assistant", "content": f"收到：{message}"})
    return history

config = {"configurable": {"thread_id": "chat-1"}}
chatbot.invoke("你好", config)       # [用户:你好, AI:收到：你好]
chatbot.invoke("今天天气如何", config)  # [用户:你好, AI:收到：你好, 用户:今天天气如何, AI:收到：今天天气如何]
```

## 3. Python 进阶

### 装饰器类 vs 装饰器函数

`@task` 是装饰器函数——既可以带参数调用 `@task(retry_policy=...)`，也可以直接装饰 `@task`：

```python
# 带参数：task(...) 返回 decorator，decorator(func) 返回 _TaskFunction
@task(retry_policy=RetryPolicy(max_attempts=3))
def my_task(x): ...

# 不带参数：task(func) 直接返回 _TaskFunction
@task
def my_task(x): ...
```

`@entrypoint` 是装饰器类——它是一个实现了 `__call__` 的类：

```python
class entrypoint(Generic[ContextT]):
    def __init__(self, checkpointer=None, store=None, ...): ...
    def __call__(self, func): ...  # 将函数转为 Pregel
```

这种设计让 `entrypoint.final` 可以作为类属性存在，IDE 自动补全和类型检查器都能正确识别。

### functools.update_wrapper

`_TaskFunction.__init__` 中调用了 `functools.update_wrapper(self, func)`：

```python
class _TaskFunction(Generic[P, T]):
    def __init__(self, func, *, retry_policy, cache_policy, name=None):
        ...
        functools.update_wrapper(self, func)
```

`update_wrapper` 将原函数的 `__name__`、`__doc__`、`__module__` 等属性复制到 `_TaskFunction` 实例上，使得装饰后的对象在文档和调试信息中看起来像原函数。

### ParamSpec 与 TypeVar 构建泛型装饰器

```python
P = ParamSpec("P")   # 参数规格——捕获 *args, **kwargs 的类型
T = TypeVar("T")     # 返回值类型

class _TaskFunction(Generic[P, T]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> SyncAsyncFuture[T]:
        ...
```

`ParamSpec("P")` 让装饰器能精确保留被装饰函数的参数签名类型。`P.args` 和 `P.kwargs` 在调用时分别展开为位置参数和关键字参数的类型。这使得 `@task` 装饰器在类型检查器中完全透明。

### @overload 三重重载

`task()` 函数有三个 `@overload` 签名：

```python
@overload
def task(__func_or_none__: None = None, *, name=..., retry_policy=..., cache_policy=..., **kwargs) -> Callable[..., _TaskFunction[P, T]]: ...

@overload
def task(__func_or_none__: Callable[P, Awaitable[T]]) -> _TaskFunction[P, T]: ...

@overload
def task(__func_or_none__: Callable[P, T]) -> _TaskFunction[P, T]: ...

def task(__func_or_none__=None, *, name=None, retry_policy=None, cache_policy=None, **kwargs):
    ...
```

三重重载覆盖了：1) `@task(retry_policy=...)` 带参数调用，2) `@task` 装饰异步函数，3) `@task` 装饰同步函数。类型检查器根据第一个参数的类型选择对应的重载签名。

### inspect.signature

`entrypoint.__call__` 使用 `inspect.signature(func)` 分析函数签名：

```python
sig = inspect.signature(func)
first_parameter_name = next(iter(sig.parameters.keys()), None)
input_type = sig.parameters[first_parameter_name].annotation
```

它提取第一个参数作为图的输入类型，检查 `previous` 参数是否存在以决定是否注入状态，解析返回类型中的 `entrypoint.final[R, S]` 以分离 value 和 save 的类型。

### get_origin / get_args

当返回类型是 `entrypoint.final[int, str]` 时：

```python
origin = get_origin(sig.return_annotation)  # entrypoint.final 类本身
type_annotations = get_args(sig.return_annotation)  # (int, str)
output_type, save_type = type_annotations  # int, str
```

`get_origin` 提取泛型的原始类，`get_args` 提取类型参数。这允许 `@entrypoint` 在运行时解析类型注解，为 Pregel 图提供正确的输入输出类型信息。

### 嵌套 dataclass：entrypoint.final

`entrypoint.final` 是定义在 `entrypoint` 类内部的 dataclass：

```python
class entrypoint(Generic[ContextT]):
    @dataclass(**_DC_KWARGS)
    class final(Generic[R, S]):
        value: R
        save: S
```

嵌套类的好处：`entrypoint.final` 作为命名空间明确表达了它属于 entrypoint 体系。`_DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}` 保证了构造时必须用关键字参数、实例不可变、内存高效。

### __slots__

`LazyAtomicCounter` 和 `entrypoint.final` 都使用了 `__slots__`：

```python
class LazyAtomicCounter:
    __slots__ = ("_counter",)
```

`__slots__` 阻止了 `__dict__` 的创建，使每个实例只占用固定的内存空间。对于频繁创建的小对象（如 scratchpad 中的计数器），这能显著降低内存开销。

## 4. 代码走读

### _TaskFunction

```python
class _TaskFunction(Generic[P, T]):
    def __init__(self, func, *, retry_policy, cache_policy, name=None):
        if name is not None:
            if hasattr(func, "__func__"):
                # 类方法：创建绑定到实例的 partial
                instance_method = functools.partial(func.__func__, func.__self__)
                instance_method.__name__ = name
                func = instance_method
            else:
                func.__name__ = name
        self.func = func
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        functools.update_wrapper(self, func)
```

`name` 参数处理了两种情况：普通函数直接修改 `__name__`，类方法则需要创建 `functools.partial` 以避免修改原始类方法（因为类方法可能被多个 task 共享）。

`__call__` 将调用委托给 `call()` 函数：

```python
def __call__(self, *args: P.args, **kwargs: P.kwargs) -> SyncAsyncFuture[T]:
    return call(self.func, *args, retry_policy=self.retry_policy,
                cache_policy=self.cache_policy, **kwargs)
```

### @task 装饰器

`task()` 函数的实现遵循装饰器工厂模式：

```python
def task(__func_or_none__=None, *, name=None, retry_policy=None, cache_policy=None, **kwargs):
    retry_policies = (() if retry_policy is None
                      else (retry_policy,) if isinstance(retry_policy, RetryPolicy)
                      else retry_policy)

    def decorator(func):
        return _TaskFunction(func, retry_policy=retry_policies,
                             cache_policy=cache_policy, name=name)

    if __func_or_none__ is not None:
        return decorator(__func_or_none__)  # @task 无参数形式
    return decorator                          # @task(...) 有参数形式
```

`retry_policy` 参数接受三种形式：`None`（无重试）、单个 `RetryPolicy` 实例、`RetryPolicy` 列表——内部统一转换为元组。

### entrypoint 类

**构造函数** 保存配置参数：

```python
class entrypoint(Generic[ContextT]):
    def __init__(self, checkpointer=None, store=None, cache=None,
                 context_schema=None, cache_policy=None, retry_policy=None, **kwargs):
        # 处理废弃参数 config_schema -> context_schema
        # 处理废弃参数 retry -> retry_policy
        self.checkpointer = checkpointer
        self.store = store
        ...
```

**__call__ 方法** 将函数转为 Pregel 图：

```python
def __call__(self, func: Callable[..., Any]) -> Pregel:
    bound = get_runnable_for_entrypoint(func)
    sig = inspect.signature(func)
    # ... 类型提取逻辑 ...
    graph = Pregel(
        nodes={
            func.__name__: PregelNode(
                bound=bound,
                triggers=[START],
                channels=START,
                writers=[ChannelWrite([
                    ChannelWriteEntry(END, mapper=_pluck_return_value),
                    ChannelWriteEntry(PREVIOUS, mapper=_pluck_save_value),
                ])],
            )
        },
        channels={
            START: EphemeralValue(input_type),
            END: LastValue(output_type, END),
            PREVIOUS: LastValue(save_type, PREVIOUS),
        },
        input_channels=START,
        output_channels=END,
        ...
    )
    return graph
```

关键设计决策：
- **单节点图**：只有一个节点（函数名），触发条件是 `START`
- **三个 channel**：`START`（EphemeralValue，输入即消费）、`END`（LastValue，输出值）、`PREVIOUS`（LastValue，保存值）
- **Writer 分流**：`_pluck_return_value` 提取 `entrypoint.final.value` 写入 END，`_pluck_save_value` 提取 `entrypoint.final.save` 写入 PREVIOUS

### entrypoint.final

```python
@dataclass(**_DC_KWARGS)
class final(Generic[R, S]):
    value: R   # 返回给调用者的值
    save: S    # 保存到检查点的值
```

`_pluck_return_value` 和 `_pluck_save_value` 在 `__call__` 中定义，通过 `isinstance(value, entrypoint.final)` 判断返回值是否为 `final` 实例：

```python
def _pluck_return_value(value):
    return value.value if isinstance(value, entrypoint.final) else value

def _pluck_save_value(value):
    return value.save if isinstance(value, entrypoint.final) else value
```

如果不是 `final` 实例，value 和 save 相同——即普通返回值既返回又保存。

### SyncAsyncFuture

```python
class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T]):
    def __await__(self) -> Generator[T, None, T]:
        yield cast(T, ...)
```

`SyncAsyncFuture` 继承自 `concurrent.futures.Future`，同时实现了 `__await__` 协议。这意味着它既可以在同步代码中用 `.result()` 获取值，也可以在异步代码中用 `await` 获取值。

`__await__` 中的 `yield cast(T, ...)` 是一个关键实现：`...` 是 Python 的 Ellipsis 单例，`cast(T, ...)` 仅作类型标注用途。实际上 `__await__` 不会被执行——当 Pregel 执行引擎调度 task 时，会直接将结果设置到 Future 对象上，`await` 或 `.result()` 只是从中读取已完成的值。

### call() 函数

```python
def call(func, *args, retry_policy=None, cache_policy=None, **kwargs) -> SyncAsyncFuture[T]:
    config = get_config()
    impl = config[CONF][CONFIG_KEY_CALL]
    fut = impl(func, (args, kwargs), retry_policy=retry_policy,
               cache_policy=cache_policy, callbacks=config["callbacks"])
    return fut
```

`call()` 从运行时配置中获取 `CONFIG_KEY_CALL` 回调——这是 Pregel 注入的任务调度器。它将函数、参数、策略和回调打包后提交执行，返回一个 Future。这解释了为什么 `@task` 只能在 `@entrypoint` 或 `StateGraph` 内部调用——脱离 Pregel 上下文时，`CONFIG_KEY_CALL` 不存在。

### get_runnable_for_entrypoint

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

同步函数被包装为同时拥有同步和异步入口的 `RunnableCallable`——异步入口使用 `run_in_executor` 在线程池中执行。`CACHE` 字典避免重复包装同一函数。`_lookup_module_and_qualname` 检查函数是否能通过模块路径定位——动态创建的函数（如 lambda）不可缓存。

## 5. 实现原理

### @entrypoint 如何构建单节点 Pregel

`@entrypoint` 本质上是一个语法糖，它将函数转化为一个三通道单节点的 Pregel 图：

```
输入 → [START channel] → [entrypoint 节点] → [END channel] → 输出
                                    ↓
                            [PREVIOUS channel] → 下次调用的 previous 参数
```

**START 通道** 使用 `EphemeralValue`——这是一种"读即焚"的 channel，值被读取后自动清除，防止重复消费。

**END 通道** 使用 `LastValue`——保存最新的输出值，`output_channels=END` 指定从该通道读取最终结果。

**PREVIOUS 通道** 使用 `LastValue`——保存最新状态值。下次调用时，Pregel 的 `channels_from_checkpoint()` 恢复此 channel 的值，`previous` 参数通过检查点机制注入。

Writer 中的 `ChannelWriteEntry(END, mapper=_pluck_return_value)` 和 `ChannelWriteEntry(PREVIOUS, mapper=_pluck_save_value)` 分别将输出写入两个通道，mapper 函数负责从 `entrypoint.final` 中提取不同的值。

### @task 如何实现并行执行

`@task` 的并行机制分三层：

1. **提交层**：`call()` 通过 `CONFIG_KEY_CALL` 将任务提交到 Pregel 的任务队列，立即返回 `SyncAsyncFuture`
2. **调度层**：Pregel 引擎在当前超步骤中调度所有已提交的任务，同步任务分配到线程池、异步任务直接在事件循环中执行
3. **收集层**：`.result()` 阻塞等待 Future 完成；`await` 异步等待 Future 完成

```python
# 这两行提交的任务会被 Pregel 并行调度
future_a = task_a(x)  # 立即返回 Future
future_b = task_b(y)  # 立即返回 Future
# 此时两个任务正在并行执行
result_a = future_a.result()  # 等待完成
result_b = future_b.result()  # 等待完成
```

`get_runnable_for_task` 在包装函数后追加了 `ChannelWrite([ChannelWriteEntry(RETURN)])`，确保 task 的返回值通过 RETURN 通道写回 Pregel 的状态管理中，这是检查点保存 task 结果的机制。

### entrypoint.final 的值/保存分离

当 `@entrypoint` 函数返回 `entrypoint.final(value=X, save=Y)` 时：

1. 节点的 writer 被触发，接收返回值
2. `_pluck_return_value` 提取 `X`，写入 END 通道——调用者获得 `X`
3. `_pluck_save_value` 提取 `Y`，写入 PREVIOUS 通道——检查点保存 `Y`
4. 下次调用时，`previous` 参数从 PREVIOUS 通道恢复的值就是 `Y`

如果不使用 `entrypoint.final`，则 `value` 和 `save` 是同一个值，两个通道写入相同内容。

## 6. 动手实验

### 实验 1：创建 @entrypoint 工作流

```python
from langgraph.func import entrypoint

@entrypoint()
def simple_pipeline(text: str) -> str:
    """最简单的 entrypoint 工作流"""
    return f"处理结果：{text.upper()}"

# 入口函数实际上是一个 Pregel 实例
print(type(simple_pipeline))  # <class 'langgraph.pregel.Pregel'>

# 直接调用
result = simple_pipeline.invoke("hello")
print(result)  # 处理结果：HELLO

# 流式调用
for chunk in simple_pipeline.stream("world"):
    print(chunk)  # {'simple_pipeline': '处理结果：WORLD'}
```

### 实验 2：使用 @task 实现并行

```python
import time
from langgraph.func import entrypoint, task

@task
def slow_add(x: int, y: int) -> int:
    time.sleep(1)
    return x + y

@task
def slow_multiply(x: int, y: int) -> int:
    time.sleep(1)
    return x * y

@entrypoint()
def compute(input_val: int) -> dict:
    start = time.time()
    # 两个 task 并行执行
    add_future = slow_add(input_val, 10)
    mul_future = slow_multiply(input_val, 10)
    result = {
        "add": add_future.result(),
        "multiply": mul_future.result(),
    }
    elapsed = time.time() - start
    print(f"耗时：{elapsed:.1f}秒（并行，应约1秒而非2秒）")
    return result

compute.invoke(5)  # 耗时约1秒
```

### 实验 3：测试 entrypoint.final

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

@entrypoint(checkpointer=InMemorySaver())
def counter(n: int, *, previous: int = None) -> entrypoint.final[int, int]:
    prev = previous or 0
    new_total = prev + n
    # 调用者看到 prev（旧累计），检查点保存 new_total（新累计）
    return entrypoint.final(value=prev, save=new_total)

config = {"configurable": {"thread_id": "counter-test"}}

r1 = counter.invoke(10, config)
print(f"第1次：返回={r1}")  # 0 (previous 为 None，prev=0)

r2 = counter.invoke(20, config)
print(f"第2次：返回={r2}")  # 10 (上次的 save=10)

r3 = counter.invoke(5, config)
print(f"第3次：返回={r3}")  # 30 (上次的 save=10+20=30)
```

### 实验 4：验证 @task 只能在入口内调用

```python
from langgraph.func import task

@task
def standalone_task(x: int) -> int:
    return x + 1

try:
    standalone_task(5)  # 在 Pregel 上下文外调用
except Exception as e:
    print(f"预期错误：{type(e).__name__}")  # KeyError 或类似错误
```

这验证了 `call()` 依赖 `CONFIG_KEY_CALL` 的存在——脱离 Pregel 执行环境，task 无法调度。

### 实验 5：@entrypoint 与 interrupt 的组合

```python
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

@task
def analyze(text: str) -> str:
    return f"分析结果：{text[:50]}..."

@entrypoint(checkpointer=InMemorySaver())
def review_flow(doc: str) -> dict:
    analysis = analyze(doc).result()
    approval = interrupt({"analysis": analysis, "prompt": "是否通过？"})
    return {"analysis": analysis, "approval": approval}

config = {"configurable": {"thread_id": "review-exp"}}

# 首次执行——中断
review_flow.invoke("这是一份很长的文档内容...", config)

# 恢复
result = review_flow.invoke(Command(resume="通过"), config)
print(result)  # {'analysis': '分析结果：这是一份很长的文档内容...', 'approval': '通过'}
```