# 第八章 函数式 API — func/ 模块

## Python 进阶

### 装饰器类 vs 装饰器函数

LangGraph 的 `entrypoint` 和 `task` 分别采用了类装饰器和函数装饰器,体现了两种截然不同的设计思路。

**函数装饰器** (`@task`):
```python
def task(func_or_none=None, *, name=None, ...):
    if func_or_none is not None:
        return decorator(func_or_none)  # @task 直接调用
    return decorator                      # @task() 返回装饰器
```

**类装饰器** (`entrypoint`):
```python
class entrypoint(Generic[ContextT]):
    def __init__(self, checkpointer=None, store=None, ...):
        self.checkpointer = checkpointer
        ...

    def __call__(self, func):
        # 将函数转换为 Pregel 实例
        return Pregel(...)
```

类装饰器的优势在于:
1. 可以持有状态(`self.checkpointer`, `self.store`)
2. 可以暴露嵌套类型(`entrypoint.final`)
3. 支持 `@entrypoint(checkpointer=...)` 和 `@entrypoint` 两种语法

`__call__` 方法使类实例既是装饰器工厂(带参数时),也是装饰器(无参数时)。

### functools.update_wrapper 保留元数据

```python
class _TaskFunction(Generic[P, T]):
    def __init__(self, func, *, retry_policy, cache_policy, name=None):
        ...
        functools.update_wrapper(self, func)
```

`update_wrapper` 将原函数的 `__name__`, `__doc__`, `__module__`, `__qualname__`, `__dict__`, `__wrapped__` 等属性复制到包装对象上。这对于调试、序列化和 introspection 都至关重要 —— 没有 `update_wrapper`,`_TaskFunction` 实例的 `__name__` 会是 `"__call__"`,而非原始函数名。

### ParamSpec 与 TypeVar 泛型装饰器

```python
P = ParamSpec("P")  # 参数规格 —— 捕获 *args, **kwargs 的类型
T = TypeVar("T")    # 返回类型

class _TaskFunction(Generic[P, T]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> SyncAsyncFuture[T]:
```

`ParamSpec` 是 Python 3.10 引入的类型特性,用于保留被装饰函数的参数签名类型。这让 `@task` 装饰器不丢失原函数的类型信息:

```python
@task
def add(a: int, b: int) -> int:
    return a + b

# add 的类型是 _TaskFunction[[int, int], SyncAsyncFuture[int]]
# 调用 add(1, 2) 返回 SyncAsyncFuture[int],而非 Any
```

### @overload 多类型签名

`task` 函数有三种 `@overload` 签名:

```python
@overload
def task(__func_or_none__: None = None, *, ...) -> Callable[..., _TaskFunction[P, T]]: ...

@overload
def task(__func_or_none__: Callable[P, Awaitable[T]]) -> _TaskFunction[P, T]: ...

@overload
def task(__func_or_none__: Callable[P, T]) -> _TaskFunction[P, T]: ...
```

第一种用于 `@task(name="x")` (带参数的装饰器调用),后两种用于 `@task` (直接装饰同步或异步函数)。这让类型检查器能根据调用方式推断正确的返回类型。

### inspect.signature 参数自省

`entrypoint.__call__` 使用 `inspect.signature` 分析被装饰函数:

```python
sig = inspect.signature(func)
first_parameter_name = next(iter(sig.parameters.keys()), None)
```

通过 `sig.parameters` 可以获取:
- 参数名
- 参数类型注解(`.annotation`)
- 参数默认值(`.default`)
- 参数类别(`POSITIONAL_ONLY`, `VAR_KEYWORD` 等)

这对函数式 API 至关重要,因为需要区分普通输入参数和可注入参数(`config`, `previous`, `runtime`)。

### get_origin / get_args 返回类型分析

```python
origin = get_origin(sig.return_annotation)
if origin is entrypoint.final:
    type_annotations = get_args(sig.return_annotation)
    output_type, save_type = type_annotations
```

当用户标注 `-> entrypoint.final[int, str]` 时:
- `get_origin()` 返回 `entrypoint.final` 类本身
- `get_args()` 返回 `(int, str)`

这让运行时能从类型注解中提取返回值类型和保存值类型,自动构建 Pregel 的 channel 类型。

### dataclass 作为嵌套类 — entrypoint.final

```python
class entrypoint(Generic[ContextT]):
    @dataclass(**_DC_KWARGS)
    class final(Generic[R, S]):
        value: R
        save: S
```

`entrypoint.final` 是一个嵌套在 `entrypoint` 类中的 dataclass。这种设计:
1. **命名空间**: `entrypoint.final` 而非 `Final` 或 `EntrypointFinal`,与 `entrypoint` 装饰器紧密关联
2. **泛型**: `Generic[R, S]` 使类型检查器能区分 `entrypoint.final[int, str]` 和 `entrypoint.final[str, dict]`
3. **不可变**: `_DC_KWARGS` 包含 `frozen=True`,确保一旦创建就不可修改

`_DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}` 使:
- 构造必须用 `entrypoint.final(value=..., save=...)`
- 实例有 `__slots__`,内存紧凑
- 实例不可变,防止意外修改

### `__slots__` 优化

`LazyAtomicCounter` 和 `entrypoint.final` 都使用 `__slots__`:

```python
class LazyAtomicCounter:
    __slots__ = ("_counter",)
```

`__slots__` 的好处:
1. 省去每个实例约 56 字节的 `__dict__` 开销
2. 属性访问通过描述符而非 `__dict__` 查找,更快
3. 防止意外属性赋值 —— `counter.foo = 1` 会抛出 `AttributeError`

---

## 代码走读

### `_TaskFunction` 类 — 任务函数包装器

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
```

构造参数:
- `func`: 被包装的原始函数,可能是同步或异步
- `retry_policy`: 重试策略序列(注意是 `Sequence`,支持多个策略)
- `cache_policy`: 缓存策略,可选
- `name`: 可选的自定义名称

名称处理逻辑:

```python
if name is not None:
    if hasattr(func, "__func__"):
        # 处理类方法 —— 绑定方法
        instance_method = functools.partial(func.__func__, func.__self__)
        instance_method.__name__ = name
        func = instance_method
    else:
        # 处理普通函数/偏函数/可调用类
        func.__name__ = name
self.func = func
self.retry_policy = retry_policy
self.cache_policy = cache_policy
functools.update_wrapper(self, func)
```

关键细节:
1. **绑定方法处理**: 当 `func` 是绑定方法时(如 `obj.method`),`func.__func__` 是原始函数,`func.__self__` 是实例。创建 `functools.partial` 绑定两者,然后重命名 —— 这避免了修改原始类方法(可能被多个任务共享)。
2. **`func.__name__ = name`**: 直接修改函数对象。对于普通函数这是安全的,因为 `_TaskFunction` 已经通过 `update_wrapper` 复制了所有属性。
3. **`functools.update_wrapper(self, func)`**: 确保包装器保留原始函数的元数据。

`__call__` 方法:

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

调用 `_TaskFunction` 实例时,委托给 `call()` 函数,传入 `retry_policy` 和 `cache_policy`。返回值是 `SyncAsyncFuture[T]` —— 一个同时支持同步和异步等待的 Future。

缓存清理方法:

```python
def clear_cache(self, cache: BaseCache) -> None:
    if self.cache_policy is not None:
        cache.clear(((CACHE_NS_WRITES, identifier(self.func) or "__dynamic__"),))

async def aclear_cache(self, cache: BaseCache) -> None:
    if self.cache_policy is not None:
        await cache.aclear(
            ((CACHE_NS_WRITES, identifier(self.func) or "__dynamic__"),)
        )
```

两种清除方式(同步/异步),都使用 `identifier(self.func)` 作为缓存命名空间。`identifier()` 从函数对象提取模块限定名(如 `"mymodule.my_func"`)。如果函数无法识别(动态创建),则回退到 `"__dynamic__"`。

### `task` 装饰器 — 三重重载实现

```python
@overload
def task(__func_or_none__: None = None, *, ...) -> Callable[..., _TaskFunction[P, T]]: ...

@overload
def task(__func_or_none__: Callable[P, Awaitable[T]]) -> _TaskFunction[P, T]: ...

@overload
def task(__func_or_none__: Callable[P, T]) -> _TaskFunction[P, T]: ...

def task(__func_or_none__=None, *, name=None, retry_policy=None, cache_policy=None, **kwargs):
```

实现逻辑:

```python
retry_policies: Sequence[RetryPolicy] = (
    ()
    if retry_policy is None
    else (retry_policy,)
    if isinstance(retry_policy, RetryPolicy)
    else retry_policy  # 已经是 Sequence
)

def decorator(func):
    return _TaskFunction(func, retry_policy=retry_policies, cache_policy=cache_policy, name=name)

if __func_or_none__ is not None:
    return decorator(__func_or_none__)  # @task (无括号)
return decorator  # @task() (有括号)
```

**标准化处理**: `retry_policy` 可以是 `None`(无重试)、单个 `RetryPolicy`(包装为单元素元组)或 `RetryPolicy` 序列(直接使用)。这种三路分支确保内部逻辑总是处理 `Sequence[RetryPolicy]`。

**双重调用模式**: `__func_or_none__` 参数名暗示它可以是函数或 `None`:
- `@task` → `task(add_one)` → `__func_or_none__=add_one` → 直接装饰
- `@task(name="custom")` → `task(name="custom")` → `__func_or_none__=None` → 返回装饰器

**弃用参数处理**:

```python
if (retry := kwargs.get("retry", MISSING)) is not MISSING:
    warnings.warn(
        "`retry` is deprecated. Please use `retry_policy` instead.",
        category=LangGraphDeprecatedSinceV05,
        stacklevel=2,
    )
    if retry_policy is None:
        retry_policy = retry
```

使用 `MISSING` 哨兵值而非 `None`,因为 `None` 可能是合法的用户输入。`MISSING` 保证只在没有提供该参数时触发。

### `entrypoint` 类 — 函数到图的转换器

#### 构造函数

```python
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

每个参数对应 Pregel 的一个构造参数。`**kwargs` 使用 `Unpack[DeprecatedKwargs]` 处理已弃用的 `config_schema` 和 `retry` 参数,既保留向后兼容性,又通过类型系统提示弃用。

弃用逻辑:

```python
if (config_schema := kwargs.get("config_schema", MISSING)) is not MISSING:
    warnings.warn(
        "`config_schema` is deprecated. Please use `context_schema` instead.",
        category=LangGraphDeprecatedSinceV10,
        stacklevel=2,
    )
    if context_schema is None:
        context_schema = cast(type[ContextT], config_schema)
```

`LangGraphDeprecatedSinceV10` 表示在 v1.0 中移除,`LangGraphDeprecatedSinceV05` 表示在 v0.5 中弃用。

#### `entrypoint.final` 嵌套 dataclass

```python
@dataclass(**_DC_KWARGS)
class final(Generic[R, S]):
    value: R
    save: S
```

`final` 是 `entrypoint` 的嵌套类,用于解耦返回值和保存值:

```python
@entrypoint(checkpointer=InMemorySaver())
def my_workflow(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    return entrypoint.final(value=previous, save=2 * number)
```

- `value`: 返回给调用者的值
- `save`: 保存到检查点的值,下次调用通过 `previous` 参数传入

`Generic[R, S]` 让类型检查器知道 `value` 的类型是 `R`, `save` 的类型是 `S`,与函数签名的返回类型注解 `entrypoint.final[int, int]` 对应。

#### `entrypoint.__call__` — Pregel 构造器

这是整个函数式 API 的核心 —— 将一个普通函数转换为完整的 Pregel 图:

```python
def __call__(self, func: Callable[..., Any]) -> Pregel:
```

**步骤 1: 生成器检查**

```python
if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
    raise NotImplementedError("Generators are not supported in the Functional API.")
```

使用 `inspect` 模块检测生成器函数。函数式 API 不支持生成器,因为生成器的惰性求值与 Pregel 的步进执行模型冲突。

**步骤 2: 包装为 Runnable**

```python
bound = get_runnable_for_entrypoint(func)
```

`get_runnable_for_entrypoint` 检查函数是同步还是异步:
- 异步函数: 包装为 `RunnableCallable(None, func)` —— 只有异步执行路径
- 同步函数: 创建异步版本 `run_in_executor(None, func)`,并保留同步版本

**步骤 3: 类型推断**

```python
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

从函数签名提取第一个参数的类型注解作为输入类型。如果没有注解,默认为 `Any`。

**步骤 4: 返回类型分析**

```python
output_type, save_type = Any, Any
if sig.return_annotation is not inspect.Signature.empty:
    if sig.return_annotation is entrypoint.final:
        output_type = save_type = Any
    else:
        origin = get_origin(sig.return_annotation)
        if origin is entrypoint.final:
            type_annotations = get_args(sig.return_annotation)
            if len(type_annotations) != 2:
                raise TypeError(
                    "Please an annotation for both the return_ and the save values."
                )
            output_type, save_type = type_annotations
        else:
            output_type = save_type = sig.return_annotation
```

四路分支:
1. 无返回注解 → `Any, Any`
2. `-> entrypoint.final` (无泛型参数) → `Any, Any`
3. `-> entrypoint.final[int, str]` → `(int, str)`
4. `-> int` → `(int, int)` —— 返回值即保存值

**步骤 5: 提取器函数**

```python
def _pluck_return_value(value: Any) -> Any:
    return value.value if isinstance(value, entrypoint.final) else value

def _pluck_save_value(value: Any) -> Any:
    return value.save if isinstance(value, entrypoint.final) else value
```

两个闭包函数:
- `_pluck_return_value`: 如果返回 `entrypoint.final`,取 `.value`;否则直接返回
- `_pluck_save_value`: 如果返回 `entrypoint.final`,取 `.save`;否则直接返回

这实现了返回值与保存值的解耦。

**步骤 6: Pregel 构造**

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
    stream_mode="updates",
    stream_eager=True,
    checkpointer=self.checkpointer,
    store=self.store,
    cache=self.cache,
    cache_policy=self.cache_policy,
    retry_policy=self.retry_policy or (),
    context_schema=self.context_schema,
)
```

逐字段解析:

- **nodes**: 单节点图,节点名为函数名。`triggers=[START]` 表示输入到来时触发。`channels=START` 表示读取 START channel 的值作为输入。`writers` 包含两个写操作:
  - 写入 `END` channel(值经 `_pluck_return_value` 提取)—— 最终返回给用户
  - 写入 `PREVIOUS` channel(值经 `_pluck_save_value` 提取)—— 保存到检查点

- **channels**: 三个 channel:
  - `START: EphemeralValue(input_type)` —— 临时值,消费后消失
  - `END: LastValue(output_type)` —— 保留最终值
  - `PREVIOUS: LastValue(save_type)` —— 保留前次保存值

- **EphemeralValue vs LastValue**:
  - `EphemeralValue` 在被读取后自动清除,适合一次性输入
  - `LastValue` 保留最后一次写入的值,适合持久化状态

- **stream_mode="updates"**: 只流式传输更新,而非完整状态
- **stream_eager=True**: 尽快输出流式结果,不等整个步骤完成

**步骤 7: serde allowlist**

```python
if _serde.STRICT_MSGPACK_ENABLED:
    serde_allowlist = _serde.build_serde_allowlist(
        schemas=[input_type, output_type, save_type]
        + ([self.context_schema] if self.context_schema is not None else []),
        channels=graph.channels,
    )
    graph._serde_allowlist = serde_allowlist
    graph.checkpointer = _serde.apply_checkpointer_allowlist(
        graph.checkpointer, serde_allowlist
    )
```

当启用严格 msgpack 序列化时,构建一个允许列表,只允许特定类型通过序列化。这是一种安全措施,防止任意对象的反序列化漏洞。`build_serde_allowlist` 从类型注解和 channel 规格中收集允许的类型。

### `SyncAsyncFuture` — 双模态 Future

```python
class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T]):
    def __await__(self) -> Generator[T, None, T]:
        yield cast(T, ...)
```

`SyncAsyncFuture` 继承自 `concurrent.futures.Future`,同时实现了 `__await__`:
- **同步调用**: `future.result()` —— 阻塞等待结果
- **异步调用**: `await future` —— 通过 `__await__` 挂起协程

`__await__` 中 `yield cast(T, ...)` 使用 `...`(Ellipsis)作为占位值,这是一个精巧的 hack —— `__await__` 必须是生成器,但实际值的解析由 Pregel 运行时处理。

### `call()` 函数 — 任务调用入口

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
        func,
        (args, kwargs),
        retry_policy=retry_policy,
        cache_policy=cache_policy,
        callbacks=config["callbacks"],
    )
    return fut
```

关键设计:
1. `get_config()` 获取当前运行时配置,其中包含 `CONFIG_KEY_CALL` —— 这是 Pregel 注入的回调函数
2. `impl()` 是 Pregel 运行时的任务提交函数,它创建并调度 `SyncAsyncFuture`
3. 这实现了**依赖注入**: `call()` 不直接知道如何调度任务,而是从配置中获取实现

---

## 运行原理

### `@entrypoint` 如何将函数转换为 Pregel 实例

```
用户代码:
    @entrypoint(checkpointer=InMemorySaver())
    def my_func(input: int) -> str:
        return str(input)

转换过程:
    1. entrypoint.__init__(checkpointer=InMemorySaver())
       → 创建 entrypoint 实例,存储 checkpointer

    2. entrypoint.__call__(my_func)
       → get_runnable_for_entrypoint(my_func)
       → RunnableCallable(my_func, my_func_async, name="my_func")

    3. 类型推断:
       input_type = int (第一个参数的注解)
       output_type = str (返回值注解)
       save_type = str (未使用 final,与 output_type 相同)

    4. Pregel 构造:
       ┌─────────────────────────────────────────────┐
       │  Pregel                                     │
       │                                             │
       │  channels:                                  │
       │    START ── EphemeralValue(int)             │
       │    END   ── LastValue(str, "END")            │
       │    PREVIOUS ── LastValue(str, "PREVIOUS")   │
       │                                             │
       │  nodes:                                     │
       │    my_func ── PregelNode                    │
       │      triggers: [START]                      │
       │      channels: START                        │
       │      writers: ChannelWrite([               │
       │        END ── mapper=_pluck_return_value    │
       │        PREVIOUS ── mapper=_pluck_save_value │
       │      ])                                     │
       │                                             │
       │  input: START                               │
       │  output: END                                │
       └─────────────────────────────────────────────┘

数据流:
    my_func.invoke(42)
         ↓
    START channel ← 42
         ↓
    my_func node 读取 START → 输入 42
         ↓
    my_func(42) → "42"
         ↓
    _pluck_return_value("42") → "42" → END channel → 返回用户
    _pluck_save_value("42") → "42" → PREVIOUS channel → 保存到检查点
```

### `entrypoint.final` 的 value/save 分离

```
用户代码:
    @entrypoint(checkpointer=InMemorySaver())
    def counter(n: int, *, previous: Any = None) -> entrypoint.final[int, int]:
        prev = previous or 0
        return entrypoint.final(value=prev, save=prev + n)

第一次调用: counter.invoke(5)
    previous = None → prev = 0
    return entrypoint.final(value=0, save=5)
    ↓
    _pluck_return_value → 0 (返回给用户)
    _pluck_save_value → 5 (保存到 PREVIOUS channel)

第二次调用: counter.invoke(3)
    previous = 5 → prev = 5
    return entrypoint.final(value=5, save=8)
    ↓
    _pluck_return_value → 5 (返回给用户)
    _pluck_save_value → 8 (保存到 PREVIOUS channel)
```

### `@task` 创建 SyncAsyncFuture 的并行执行

```
用户代码:
    @task
    def add_one(x: int) -> int:
        return x + 1

    @entrypoint()
    def main(nums: list[int]) -> list[int]:
        futures = [add_one(n) for n in nums]  # 创建 3 个 SyncAsyncFuture
        return [f.result() for f in futures]    # 等待所有结果

执行流程:
    main([1, 2, 3])
    ↓
    add_one(1) → call() → impl(func, (1,), ...) → SyncAsyncFuture
    add_one(2) → call() → impl(func, (2,), ...) → SyncAsyncFuture
    add_one(3) → call() → impl(func, (3,), ...) → SyncAsyncFuture
    ↓
    Pregel 调度器并行执行三个任务
    ↓
    futures[0].result() → 2
    futures[1].result() → 3
    futures[2].result() → 4
```

### `get_runnable_for_entrypoint` 的缓存机制

```python
CACHE: dict[tuple[Callable[..., Any], bool], Runnable] = {}

def get_runnable_for_entrypoint(func: Callable[..., Any]) -> Runnable:
    key = (func, False)
    if key in CACHE:
        return CACHE[key]
    # ... 创建 runnable ...
    if not _lookup_module_and_qualname(func):
        return run  # 动态函数,不缓存
    return CACHE.setdefault(key, run)
```

缓存键是 `(func, False)`,其中 `False` 表示入口函数(非任务函数)。只有能被 `_lookup_module_and_qualname` 识别的函数才会缓存 —— 动态创建的函数(如 lambdas)不会被缓存,因为它们的模块和限定名无法被解析。

---

## 实现细节

### `entrypoint.__call__` 中 Pregel 的每个 channel 和 node 详解

**START channel (`EphemeralValue`)**:
- 每次 `invoke`/`stream` 调用时写入输入值
- 被节点读取后自动清除
- 类型由第一个参数的注解决定
- 临时性质保证每次调用独立,不会残留上次数据

**END channel (`LastValue`)**:
- 保留最终返回值
- `output_type` 决定序列化行为
- 通过 `_pluck_return_value` 写入,剥离 `entrypoint.final` 包装

**PREVIOUS channel (`LastValue`)**:
- 保存 `entrypoint.final.save` 值
- 下次调用时通过 `previous` 参数注入
- 仅当 checkpointer 存在时有意义

**单节点 PregelNode**:
- `triggers=[START]`: 输入到来时触发
- `channels=START`: 读取 START channel 的值作为输入
- `writers`: 两个写入操作,分别输出到 END 和 PREVIOUS

### `_pluck_return_value` 和 `_pluck_save_value` 的工作机制

这两个函数被注册为 `ChannelWriteEntry` 的 `mapper`:

```python
ChannelWriteEntry(END, mapper=_pluck_return_value),
ChannelWriteEntry(PREVIOUS, mapper=_pluck_save_value),
```

在 Pregel 写入 channel 时,值会先经过 `mapper` 函数转换:
1. 节点返回 `"hello"` → `_pluck_return_value("hello")` → `"hello"` → 写入 END
2. 节点返回 `entrypoint.final(value=1, save=2)` → `_pluck_return_value(...)` → `1` → 写入 END
3. 同上 → `_pluck_save_value(...)` → `2` → 写入 PREVIOUS

`isinstance(value, entrypoint.final)` 的检查确保了兼容性 —— 普通返回值和 `final` 包装值都能正确处理。

### serde allowlist 机制

```python
if _serde.STRICT_MSGPACK_ENABLED:
    serde_allowlist = _serde.build_serde_allowlist(
        schemas=[input_type, output_type, save_type]
        + ([self.context_schema] if self.context_schema is not None else []),
        channels=graph.channels,
    )
    graph._serde_allowlist = serde_allowlist
    graph.checkpointer = _serde.apply_checkpointer_allowlist(
        graph.checkpointer, serde_allowlist
    )
```

工作原理:
1. `STRICT_MSGPACK_ENABLED`: 仅当安装了 `ormsgpack` 时为 `True`
2. `build_serde_allowlist`: 从类型注解和 channel 规格收集允许的类型
3. `apply_checkpointer_allowlist`: 使用 `checkpointer.with_allowlist(allowlist)` 创建新的 checkpointer 实例,限制序列化范围

这是一种纵深防御策略:即使攻击者能控制检查点数据,也无法反序列化不在白名单中的类型。

### `_lookup_module_and_qualname` — 函数来源追踪

```python
def _lookup_module_and_qualname(obj, name=None):
    if name is None:
        name = getattr(obj, "__qualname__", None)
    # ...
    module_name = _whichmodule(obj, name)
    if module_name == "__main__":
        return None  # 不缓存主模块中的函数
    module = sys.modules.get(module_name, None)
    # ...
    obj2, parent = _getattribute(module, name)
    if obj2 is not obj:
        return None  # 函数已被替换,不缓存
    return module, name
```

这段代码借自 `cloudpickle`,确保只缓存「可重新定位」的函数:
- 主模块中的函数不缓存(进程重启后可能不存在)
- 动态创建的函数不缓存(lambda, `eval` 创建的函数)
- 已被替换的函数不缓存(防止缓存过期引用)

### `identifier()` — 函数全局标识

```python
def identifier(obj, name=None):
    if isinstance(obj, PregelNode):
        obj = obj.bound
    if isinstance(obj, RunnableSeq):
        obj = obj.steps[0]
    if isinstance(obj, RunnableCallable):
        obj = obj.func
    # ...
    return f"{module_name}.{name}"
```

`identifier` 从各种包装类型中提取原始函数,然后返回 `module.qualname` 格式的标识符(如 `"mymodule.my_func"`)。这用于缓存键的命名空间,确保不同模块的同名函数不会冲突。

---

## 动手实验

### 实验 1: 创建 @entrypoint 并检查 Pregel 结构

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

@entrypoint(checkpointer=InMemorySaver())
def my_workflow(x: int) -> int:
    return x * 2

# 检查生成的 Pregel 实例
graph = my_workflow  # entrypoint.__call__ 返回的就是 Pregel

print(f"节点: {list(graph.nodes.keys())}")  # ['my_workflow']
print(f"输入 channel: {graph.input_channels}")  # 'START'
print(f"输出 channel: {graph.output_channels}")  # 'END'

# 检查 channel 类型
for name, channel in graph.channels.items():
    print(f"  {name}: {type(channel).__name__}")
# START: EphemeralValue
# END: LastValue
# PREVIOUS: LastValue

# 检查节点的 writer
node = graph.nodes["my_workflow"]
for writer in node.flat_writers:
    if hasattr(writer, "writes"):
        for entry in writer.writes:
            print(f"  写入 {entry.channel}, mapper={entry.mapper.__name__}")
# 写入 END, mapper=_pluck_return_value
# 写入 PREVIOUS, mapper=_pluck_save_value
```

### 实验 2: 使用 @task 进行并行执行

```python
import time
from langgraph.func import entrypoint, task

@task
def slow_add(x: int) -> int:
    time.sleep(1.0)  # 模拟慢速操作
    return x + 1

@entrypoint()
def parallel_main(nums: list[int]) -> list[int]:
    # 创建多个 Future,它们会被 Pregel 并行调度
    futures = [slow_add(n) for n in nums]
    # 等待所有结果
    results = [f.result() for f in futures]
    return results

# 调用
result = parallel_main.invoke([1, 2, 3])
print(result)  # [2, 3, 4]  —— 总时间约 1 秒而非 3 秒
```

### 实验 3: entrypoint.final 的 value/save 分离

```python
from typing import Any
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

@entrypoint(checkpointer=InMemorySaver())
def accumulator(n: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    prev = previous or 0
    # value: 返回给调用者的是累计值
    # save: 保存到检查点的是新累计值
    return entrypoint.final(value=prev, save=prev + n)

config = {"configurable": {"thread_id": "test"}}

# 第一次: previous=None, value=0, save=5
result1 = accumulator.invoke(5, config)
print(f"结果: {result1}, 返回值=0(之前没有累计)")

# 第二次: previous=5, value=5, save=8
result2 = accumulator.invoke(3, config)
print(f"结果: {result2}, 返回值=5(之前累计5), 新保存值=8")

# 第三次: previous=8, value=8, save=10
result3 = accumulator.invoke(2, config)
print(f"结果: {result3}, 返回值=8(之前累计8), 新保存值=10")
```

### 实验 4: 检查 _TaskFunction 的元数据保留

```python
from langgraph.func import task

@task(name="custom_task_name")
def my_function(x: int) -> int:
    """This is my function."""
    return x + 1

# update_wrapper 保留了原始函数的元数据
print(my_function.__name__)  # "custom_task_name" (被 name 参数覆盖)
print(my_function.__doc__)   # "This is my function."

# 类型信息通过 Generic 保留
from typing import get_type_hints
# get_type_hints 可以看到原始函数的签名

# 调用返回 SyncAsyncFuture
future = my_function(1)
print(type(future))  # <class 'SyncAsyncFuture'>
```

### 实验 5: 验证 EphemeralValue 和 LastValue 的行为差异

```python
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue

# EphemeralValue: 读取后消失
eph = EphemeralValue(int)
eph.update([42])
print(f"第一次读取: {eph.get()}")  # 42
# 再次读取时,EphemeralValue 已经被消费

# LastValue: 始终保留最后写入的值
lv = LastValue(int, "test")
lv.update([42])
print(f"第一次读取: {lv.get()}")  # 42
print(f"第二次读取: {lv.get()}")  # 42 (值仍然存在)
lv.update([100])
print(f"更新后读取: {lv.get()}")  # 100 (保留最新值)
```

这个实验展示了为什么 START 使用 `EphemeralValue`(每次输入只用一次),而 END 和 PREVIOUS 使用 `LastValue`(需要持久化直到被消费或更新)。