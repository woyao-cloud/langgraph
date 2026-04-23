# 第五章 Pregel 执行引擎（下）— _loop.py, _algo.py, _runner.py

上一章我们从宏观视角俯瞰了 Pregel 引擎的整体架构。本章深入引擎的三大核心模块：算法层 `_algo.py`、循环编排层 `_loop.py`、任务执行层 `_runner.py`。如果说 Pregel 是一座工厂，那么 `_algo.py` 是生产计划调度算法，`_loop.py` 是车间主任的每日流程表，`_runner.py` 则是并行流水线的调度器。

---

## 5.1 Java 桥梁

| Java 概念 | Python 对应 | 说明 |
|-----------|------------|------|
| `CompletableFuture` | `asyncio.Future` / `concurrent.futures.Future` | Java 的异步未来对象有两种 Python 等价物：异步事件循环用的 `asyncio.Future` 和线程池用的 `concurrent.futures.Future` |
| `ExecutorService.submit()` | `Submit` (自定义) / `concurrent.futures.ThreadPoolExecutor` | Java 通过 `ExecutorService` 提交任务；LangGraph 封装了 `Submit` 协议来抽象同步/异步两种提交方式 |
| `CompletableFuture.allOf()` | `asyncio.gather()` / `concurrent.futures.wait()` | Java 等待全部完成用 `allOf`；Python 异步用 `gather`，同步用 `wait(FIRST_COMPLETED)` |
| `try-with-resources` | `with` 语句 / `__enter__` + `__exit__` | Java 7 引入的自动资源管理；Python 通过上下文管理器协议实现，语义更简洁 |
| `MethodReference::apply` | `functools.partial` | Java 方法引用直接绑定方法；Python 用 `partial` 偏函数实现类似"预填参数"的效果 |
| `Interface` (structural) | `Protocol` | Java 的接口是名义类型（nominal typing）；Python 的 `typing.Protocol` 实现结构化子类型（structural subtyping），无需显式声明 `implements` |

---

## 5.2 Python 概念速查

### defaultdict

```python
from collections import defaultdict
# Java: Map<String, List<String>> map = new HashMap<>(); 需要 computeIfAbsent
# Python: 一行搞定
writes_by_channel: dict[str, list[Any]] = defaultdict(list)
writes_by_channel["messages"].append("hello")  # 自动初始化空列表
```

Java 中 `Map.get(key)` 返回 `null` 时需手动 `computeIfAbsent`，而 `defaultdict` 在键不存在时自动调用工厂函数创建默认值。

### itertools

```python
import itertools
# 链式拼接多个迭代器，Java 需要 Stream.concat 多次调用
all_tasks = itertools.chain(pull_tasks, push_tasks)
```

### partial 偏函数 vs Java 方法引用

```python
from functools import partial
# 相当于 Java: biFunction.bind(arg1) 或 methodRef.curry(arg)
call_fn = partial(_call, weakref.ref(task), retry_policy=retry_policy)
```

Java 的 `MethodReference::apply` 只绑定方法到对象，而 Python `partial` 还可以预填任意位置参数和关键字参数，更加灵活。

### Protocol 结构化子类型

```python
from typing import Protocol

class WritesProtocol(Protocol):
    @property
    def path(self) -> tuple[str | int | tuple, ...]: ...
    @property
    def writes(self) -> Sequence[tuple[str, Any]]: ...
```

任何具有 `path` 和 `writes` 属性的对象自动满足 `WritesProtocol`，无需 `implements` 声明。这类似于 Go 的接口——只要你实现了方法，你就实现了接口。

### async/await vs Java CompletableFuture

```python
# Python: 原生协程
async def atick(self, tasks):
    done, inflight = await asyncio.wait(futures, return_when=FIRST_COMPLETED)

# Java 等价写法
# CompletableFuture.allOf(futures).thenApply(...)
```

Python 的 `await` 挂起当前协程、释放事件循环，比 Java 的回调链更直观。

### 上下文管理器 vs try-with-resources

```python
# Python
with PregelLoop(...) as loop:
    loop.tick()

# Java 等价
try (PregelLoop loop = new PregelLoop(...)) {
    loop.tick();
}
```

Python 通过 `__enter__`/`__exit__` 魔术方法实现，异步版本用 `__aenter__`/`__aexit__`。

---

## 5.3 代码走读

### 5.3.1 `_algo.py` — 核心算法函数

这个文件实现了 Pregel 模型的核心"超步"逻辑，包含六个关键函数。

#### prepare_next_tasks()

```python
def prepare_next_tasks(
    checkpoint: Checkpoint,
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    ...
) -> dict[str, PregelTask] | dict[str, PregelExecutableTask]:
```

这是 Pregel 引擎的"大脑"——决定下一步哪些节点需要执行。核心逻辑分两步：

1. **消费 PUSH 任务**：从 `TASKS` channel（一个 `Topic` 类型）取出所有 `Send` 对象，每个 `Send` 生成一个 PUSH 任务。
2. **计算 PULL 任务**：对每个候选节点调用 `prepare_single_task()`，内部通过 `_triggers()` 检查该节点的 `triggers` 所对应的 channel 版本是否高于该节点上次"看到"的版本。

版本对比机制类似于 Java 中的乐观锁：每个 channel 维护一个单调递增的版本号，每个节点记录自己已处理的版本。当 `channel_versions[chan] > versions_seen[node][chan]` 时，节点被触发。

关键优化：当 `trigger_to_nodes`（channel 到节点的映射）可用时，只需检查被更新 channel 关联的节点，无需遍历全部节点——类似数据库的索引加速。

#### apply_writes()

```python
def apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    tasks: Iterable[WritesProtocol],
    get_next_version: GetNextVersion | None,
    trigger_to_nodes: Mapping[str, Sequence[str]],
) -> set[str]:
```

将已完成任务的写入应用到 channel 和 checkpoint 上，流程如下：

1. **更新 versions_seen**：记录每个节点已处理的 channel 版本
2. **消费已读 channel**：调用 `channels[chan].consume()` 通知 channel 数据已被消费（对 `Topic` 类型尤其重要，消费后清空）
3. **分组写入**：用 `defaultdict(list)` 按 channel 聚合写入值
4. **应用到 channel**：调用 `channels[chan].update(vals)` 更新值
5. **步进通知**：未被直接更新的 channel 也收到 `update(EMPTY_SEQ)` 通知，以触发条件边
6. **结束通知**：若无 channel 触发任何节点，调用 `channels[chan].finish()` 通知所有 channel 本轮结束

返回值 `updated_channels` 是本轮被更新的 channel 集合，供下一轮 `prepare_next_tasks()` 做优化过滤。

#### should_interrupt()

```python
def should_interrupt(
    checkpoint: Checkpoint,
    interrupt_nodes: All | Sequence[str],
    tasks: Iterable[PregelExecutableTask],
) -> list[PregelExecutableTask]:
```

检查是否需要中断执行。两个条件缺一不可：

1. 有 channel 自上次中断后被更新（`channel_versions > versions_seen[INTERRUPT]`）
2. 被触发的任务名在 `interrupt_nodes` 列表中（或 `interrupt_nodes == "*"` 表示所有节点）

这种设计使得中断是"有状态的"——同一个 channel 更新不会重复触发中断，除非有新的写入发生。

#### local_read()

```python
def local_read(
    scratchpad: PregelScratchpad,
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    task: WritesProtocol,
    select: list[str] | str,
    fresh: bool = False,
) -> dict[str, Any] | Any:
```

注入到每个任务配置中的读取函数，让条件边能读取到**包含当前任务写入**的"局部状态视图"。当 `fresh=True` 时，复制 channel 并应用当前写入后再读取——类似 Java 中的"读已提交"隔离级别。

#### prepare_single_task()

```python
def prepare_single_task(
    task_path: tuple[Any, ...],
    task_id_checksum: str | None,
    ...
) -> None | PregelTask | PregelExecutableTask:
```

根据 `task_path` 的第一个元素分发到不同的准备逻辑：

- `PUSH` + `Call` → `prepare_push_task_functional()`：函数调用型推送任务
- `PUSH` + `Send` → `prepare_push_task_send()`：Send 型推送任务
- `PULL` → 检查 `_triggers()` 后创建普通拉取任务

`task_path` 是任务的身份标识，如 `(PULL, "agent")` 或 `(PUSH, path, idx, parent_id, Call(...))`。任务 ID 通过 `xxhash` 或 `uuid5` 从 `task_path` 确定性生成，确保相同路径产生相同 ID——这对 checkpoint 恢复至关重要。

### 5.3.2 `_loop.py` — PregelLoop / AsyncPregelLoop

`PregelLoop` 是一个有状态的编排器，管理着一次 Pregel 执行的完整生命周期。

#### tick() 方法

```python
def tick(self) -> bool:
    # 1. 检查步数限制
    if self.step > self.stop:
        self.status = "out_of_steps"
        return False

    # 2. 准备下一批任务
    self.tasks = prepare_next_tasks(...)

    # 3. 无任务则完成
    if not self.tasks:
        self.status = "done"
        return False

    # 4. 匹配上一轮的 pending writes
    if not self.is_replaying and self.checkpoint_pending_writes:
        self._match_writes(self.tasks)

    # 5. 执行前中断检查
    if self.interrupt_before and should_interrupt(...):
        self.status = "interrupt_before"
        raise GraphInterrupt()

    return True
```

`tick()` 只做准备工作：调度任务、检查中断、匹配写入。实际执行由 `PregelRunner` 完成。返回 `True` 表示还有更多工作，调用者应继续循环。

#### after_tick() 方法

```python
def after_tick(self) -> None:
    # 1. 应用写入
    self.updated_channels = apply_writes(...)

    # 2. 输出流式结果
    self._emit("values", ...)

    # 3. 清除 pending writes
    self.checkpoint_pending_writes.clear()

    # 4. 保存检查点
    self._put_checkpoint({"source": "loop"})

    # 5. 执行后中断检查
    if self.interrupt_after and should_interrupt(...):
        self.status = "interrupt_after"
        raise GraphInterrupt()
```

`after_tick()` 是 `tick()` 的对应物——在任务执行完毕后调用，负责写入应用、检查点保存和后置中断检查。

#### 上下文管理器模式

```python
class PregelLoop:
    def __enter__(self) -> PregelLoop:
        # 初始化 checkpoint、channels
        self._first(...)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        return self._suppress_interrupt(exc_type, exc_val, exc_tb)
```

`PregelLoop` 同时实现了同步和异步上下文管理器。`__exit__` 中的关键逻辑是 `_suppress_interrupt()`：当异常是 `GraphInterrupt` 且这是顶层图时，**抑制异常**并保存最终状态。这确保中断不会冒泡到调用者——调用者通过 `loop.output` 获取结果。

异步版本 `AsyncPregelLoop` 使用 `__aenter__`/`__aexit__`，配合 `AsyncExitStack` 管理多个异步上下文。

#### _first() 方法

`_first()` 是入口初始化方法，处理三种场景：

1. **恢复（resume）**：从检查点恢复，将 `versions_seen[INTERRUPT]` 更新到当前版本
2. **新输入**：将输入映射为 channel 写入，应用后保存输入检查点
3. **Command 输入**：将 `Command` 解析为写入（含 `resume` 值）

此方法还处理了子图的检查点命名空间、重放（replay）状态传播等复杂逻辑。

### 5.3.3 `_runner.py` — PregelRunner

`PregelRunner` 负责并发执行任务并管理生命周期。

#### 核心数据结构：FuturesDict

```python
class FuturesDict(Generic[F, E], dict[F, PregelExecutableTask | None]):
    event: E
    callback: weakref.ref[Callable]
    counter: int
    done: set[F]
```

`FuturesDict` 继承自 `dict`，将 Future 映射到任务。每当添加新任务时，计数器加一；任务完成时回调 `on_done`，计数器减一。当所有任务完成（或需要停止其他任务）时，事件被设置——类似 Java 的 `CountDownLatch`。

#### tick() — 同步执行

```python
def tick(self, tasks, *, reraise=True, timeout=None, ...) -> Iterator[None]:
```

这是一个**生成器方法**（使用 `yield`），执行流程：

1. `yield` — 交还控制权给调用者（允许流式输出）
2. **快速路径**：单个任务无超时，直接同步执行（避免线程池开销）
3. **常规路径**：提交所有任务到线程池，用 `concurrent.futures.wait(FIRST_COMPLETED)` 等待
4. 每次有任务完成时 `yield`，允许调用者处理流式输出
5. 完成后调用 `_panic_or_proceed()` 处理异常或超时

生成器模式让调用者可以在每个任务完成后插入逻辑（如流式输出），而不需要回调——这比 Java 的 `addListener` 模式更优雅。

#### atick() — 异步执行

与 `tick()` 镜像对应，但使用 `asyncio.wait()` 替代 `concurrent.futures.wait()`，`asyncio.Event` 替代 `threading.Event`。

#### commit() — 写入提交

```python
def commit(self, task, exception):
    if isinstance(exception, asyncio.CancelledError):
        task.writes.append((ERROR, exception))
    elif isinstance(exception, GraphInterrupt):
        # 保存中断到检查点
        writes = [(INTERRUPT, exception.args[0])]
    elif exception:
        task.writes.append((ERROR, exception))
    else:
        # 正常完成：保存写入
        self.put_writes()(task.id, task.writes)
```

每个任务完成（成功或失败）后调用 `commit()`，将写入持久化到检查点。注意 `NO_WRITES` 标记——即使任务没有产生任何写入，也会写入 `(NO_WRITES, None)` 来标记任务已完成。

#### _should_stop_others() 与 _panic_or_proceed()

当一个任务失败（非 `GraphBubbleUp` 类型）时，取消所有其他正在执行的任务——类似 Java 中 `Future.cancel(true)` 的批量操作。`GraphInterrupt` 不被视为失败，会收集合并后统一抛出。

---

## 5.4 运行原理

将三个模块串联起来，一次完整的 Pregel 超步执行流程如下：

```
[初始化] PregelLoop.__enter__()
    └─ _first(): 映射输入/恢复状态 → 保存初始检查点

[循环] while loop.tick():
    ├─ prepare_next_tasks(): 版本对比 → 生成任务字典
    ├─ should_interrupt(): 前置中断检查
    │
    ├─ runner.tick(tasks):  ← 生成器，yield 交还控制
    │   ├─ 提交任务到线程池/协程
    │   ├─ wait(FIRST_COMPLETED) 等待
    │   ├─ yield → 调用者处理流式输出
    │   └─ commit(): 写入持久化
    │
    └─ loop.after_tick():
        ├─ apply_writes(): 更新 channel + 版本号
        ├─ _put_checkpoint(): 保存检查点
        └─ should_interrupt(): 后置中断检查

[清理] PregelLoop.__exit__()
    └─ _suppress_interrupt(): 处理中断/保存最终状态
```

关键设计：`tick()` 和 `runner.tick()` 都使用生成器（`yield`），形成一个**协作式调度**的双层循环。外层循环是超步迭代，内层循环（`runner.tick` 的 `yield`）是任务完成通知。这使得流式输出可以穿插在任务执行之间，而不是等到所有任务完成后才输出。

---

## 5.5 动手实验

### 实验 1：观察版本驱动的任务触发

```python
from langgraph.pregel._algo import prepare_next_tasks, increment

# 模拟：创建空检查点和 channel
# 修改某个 channel 的版本号，观察哪些节点被触发
# 核心观察：只有 channel_versions > versions_seen[node] 的节点才会被调度
```

### 实验 2：生成器协作调度

```python
from langgraph.pregel._runner import PregelRunner

# 创建一个简单的 PregelRunner，观察 tick() 生成器的 yield 行为
# 每次任务完成时 yield，调用者可以插入自定义逻辑
# 对比 Java 的 CountDownLatch + 回调模式
```

### 实验 3：中断与恢复

```python
from langgraph.pregel._loop import PregelLoop
from langgraph.errors import GraphInterrupt

# 设置 interrupt_before=["node_a"]
# 执行时观察 GraphInterrupt 被抑制，loop.status 变为 "interrupt_before"
# 重新创建 loop 并 resume，观察版本对比逻辑如何跳过已完成节点
```

### 实验 4：FuturesDict 并发控制

```python
import concurrent.futures
import threading

# 创建 FuturesDict，添加多个 Future
# 观察 counter 递增/递减和 event.set() 的时机
# 理解 _should_stop_others() 如何在首个失败时取消其他任务
# 对比 Java ExecutorService + Future.cancel() 的行为
```