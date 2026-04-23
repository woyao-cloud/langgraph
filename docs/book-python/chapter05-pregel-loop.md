# 第五章 Pregel 引擎——核心算法与循环

## Python 进阶

本章源码密集使用了以下 Python 进阶特性：

- **`defaultdict`**：`apply_writes()` 中按 channel 分组写入时使用 `defaultdict(list)`，避免手动判空初始化。
- **`itertools`**：配合生成器实现惰性迭代，在 `PregelRunner.tick()` 中控制执行节奏。
- **`partial` 偏函数**：`functools.partial` 大量用于将运行时依赖（如 `futures`、`scratchpad`）提前绑定到回调函数，实现配置注入。
- **`Protocol` 结构化子类型**：`WritesProtocol` 定义了写入协议的接口，不要求显式继承，只要具备 `path`、`name`、`writes`、`triggers` 属性即可。
- **`async/await` 双轨实现**：`PregelRunner.tick()` 与 `atick()`、`SyncPregelLoop` 与 `AsyncPregelLoop` 分别提供同步与异步版本，代码结构高度对称。
- **上下文管理器**：`SyncPregelLoop` 继承 `AbstractContextManager`，实现 `__enter__`/`__exit__`；`AsyncPregelLoop` 继承 `AbstractAsyncContextManager`，实现 `__aenter__`/`__aexit__`。
- **`ExitStack`**：同步循环用 `ExitStack` 管理多个上下文（`BackgroundExecutor` + `_suppress_interrupt`），保证退出时资源正确释放。
- **`deque`**：`PregelExecutableTask.writes` 使用 `deque` 而非 `list`，因为 `deque.extend` 是线程安全操作，而写入来自并发任务。
- **`datetime/timezone`**：调试输出中使用 `datetime.now(timezone.utc).isoformat()` 生成 UTC 时间戳。
- **生成器调度**：`PregelRunner.tick()` 返回 `Iterator[None]`，通过 `yield` 将控制权交还给调用者，实现协作式调度。

## 代码走读

### _algo.py —— 算法核心

#### WritesProtocol 与 PregelTaskWrites

```python
class WritesProtocol(Protocol):
    @property
    def path(self) -> tuple[str | int | tuple, ...]: ...
    @property
    def name(self) -> str: ...
    @property
    def writes(self) -> Sequence[tuple[str, Any]]: ...
    @property
    def triggers(self) -> Sequence[str]: ...
```

`WritesProtocol` 是一个 `Protocol`——Python 的结构化子类型（structural subtyping）。任何具有 `path`、`name`、`writes`、`triggers` 属性的对象自动满足此协议，无需显式声明继承。这让 `PregelExecutableTask` 和 `PregelTaskWrites` 都可以被传给 `apply_writes()` 而不需要公共基类。

```python
class PregelTaskWrites(NamedTuple):
    path: tuple[str | int | tuple, ...]
    name: str
    writes: Sequence[tuple[str, Any]]
    triggers: Sequence[str]
```

`PregelTaskWrites` 是最简实现，用于非任务来源的写入（如图输入、`update_state`）。`NamedTuple` 既不可变又轻量，且自动具备这些属性。

#### should_interrupt() —— 中断判定

```python
def should_interrupt(
    checkpoint: Checkpoint,
    interrupt_nodes: All | Sequence[str],
    tasks: Iterable[PregelExecutableTask],
) -> list[PregelExecutableTask]:
```

核心逻辑分两步：
1. 检查自上次中断以来是否有 channel 更新——遍历 `channel_versions`，与 `versions_seen[INTERRUPT]` 比较；
2. 如果有更新，从 `tasks` 中筛选匹配 `interrupt_nodes` 的任务。当 `interrupt_nodes == "*"` 时，排除 `TAG_HIDDEN` 标记的任务。

```python
any_updates_since_prev_interrupt = any(
    version > seen.get(chan, null_version)
    for chan, version in checkpoint["channel_versions"].items()
)
```

此处动态获取 `null_version`：取 `channel_versions` 中第一个值的类型，构造零值实例。这确保比较操作对不同版本类型（`int`、`str`）都有效。

#### apply_writes() —— 写入应用

这是整个超步中最关键的函数，负责将所有任务的写入应用到 channel 和 checkpoint。

```python
def apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    tasks: Iterable[WritesProtocol],
    get_next_version: GetNextVersion | None,
    trigger_to_nodes: Mapping[str, Sequence[str]],
) -> set[str]:
```

**步骤 1：排序**。任务按 `path[:3]` 排序，保证确定性：

```python
tasks = sorted(tasks, key=lambda t: task_path_str(t.path[:3]))
```

**步骤 2：更新 versions_seen**。将每个任务的触发 channel 的当前版本记录到 `versions_seen`：

```python
for task in tasks:
    checkpoint["versions_seen"].setdefault(task.name, {}).update(
        {chan: checkpoint["channel_versions"][chan]
         for chan in task.triggers
         if chan in checkpoint["channel_versions"]}
    )
```

**步骤 3：计算下一个全局版本号**：

```python
next_version = get_next_version(
    max(checkpoint["channel_versions"].values()) if checkpoint["channel_versions"] else None,
    None,
)
```

注意：所有 channel 在同一步内共享同一个 `next_version`。这保证了 `versions_seen` 比较的语义正确性——只要 channel 版本号大于 seen，就说明在本步之后更新过。

**步骤 4：消费触发 channel**。对于触发任务的 channel（非保留 channel），调用 `consume()` 标记已读，并更新版本：

```python
for chan in {chan for task in tasks for chan in task.triggers
            if chan not in RESERVED and chan in channels}:
    if channels[chan].consume() and next_version is not None:
        checkpoint["channel_versions"][chan] = next_version
```

**步骤 5：按 channel 分组写入**。使用 `defaultdict(list)` 分组：

```python
pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
for task in tasks:
    for chan, val in task.writes:
        if chan in (NO_WRITES, PUSH, RESUME, INTERRUPT, RETURN, ERROR):
            pass  # 特殊 channel 不写入
        elif chan in channels:
            pending_writes_by_channel[chan].append(val)
```

**步骤 6：应用写入到 channel**。对每个有写入的 channel 调用 `update()`：

```python
for chan, vals in pending_writes_by_channel.items():
    if channels[chan].update(vals) and next_version is not None:
        checkpoint["channel_versions"][chan] = next_version
        if channels[chan].is_available():
            updated_channels.add(chan)
```

`update()` 返回 `True` 表示值确实发生了变化。`is_available()` 检查 channel 是否可被读取（例如 `UntrackedValue` 不可用）。

**步骤 7：步进通知**。对未被本步更新的 channel 传入空序列，让它们知道新的一步已经开始：

```python
if bump_step:
    for chan in channels:
        if channels[chan].is_available() and chan not in updated_channels:
            if channels[chan].update(EMPTY_SEQ) and next_version is not None:
                checkpoint["channel_versions"][chan] = next_version
```

**步骤 8：终止通知**。如果本步没有 channel 能触发任何节点，通知所有 channel 结束：

```python
if bump_step and updated_channels.isdisjoint(trigger_to_nodes):
    for chan in channels:
        if channels[chan].finish() and next_version is not None:
            checkpoint["channel_versions"][chan] = next_version
```

#### prepare_next_tasks() —— 任务准备

```python
def prepare_next_tasks(
    checkpoint, pending_writes, processes, channels, managed,
    config, step, stop, *, for_execution, ...
) -> dict[str, PregelTask] | dict[str, PregelExecutableTask]:
```

函数通过 `@overload` 装饰器区分两种返回类型：`for_execution=False` 返回轻量 `PregelTask`，`for_execution=True` 返回完整 `PregelExecutableTask`。

核心流程：

1. **消费 PUSH 任务**：从 `TASKS` channel 取出 `Send` 对象，为每个 `Send` 调用 `prepare_single_task((PUSH, idx), ...)`。
2. **确定候选 PULL 节点**：利用 `trigger_to_nodes` 优化——如果已知哪些 channel 被更新，只检查被更新 channel 触发的节点：
```python
if updated_channels and trigger_to_nodes:
    triggered_nodes: set[str] = set()
    for channel in updated_channels:
        if node_ids := trigger_to_nodes.get(channel):
            triggered_nodes.update(node_ids)
    candidate_nodes: Iterable[str] = sorted(triggered_nodes)
```
3. **逐节点检查触发条件**：对每个候选节点调用 `_triggers()`，比较 `channel_versions` 与 `versions_seen`：
```python
def _triggers(channels, versions, seen, null_version, proc) -> bool:
    if seen is None:
        for chan in proc.triggers:
            if channels[chan].is_available():
                return True
    else:
        for chan in proc.triggers:
            if channels[chan].is_available() and versions.get(chan, null_version) > seen.get(chan, null_version):
                return True
    return False
```

当 `seen is None`（该节点从未执行过），只要任何触发 channel 有值就触发。否则，只有 channel 版本严格大于 seen 版本时才触发——这就是版本比较驱动的去重机制。

4. **构造 PregelExecutableTask**：对触发的节点，生成 task ID（使用 `_xxhash_str` 或 `_uuid5_str`），创建 `scratchpad`，准备输入值，注入 `CONFIG_KEY_SEND`、`CONFIG_KEY_READ`、`CONFIG_KEY_CALL` 等 config 键。

#### local_read() —— 局部读取

```python
def local_read(scratchpad, channels, managed, task, select, fresh=False):
```

此函数被注入到任务 config 的 `CONFIG_KEY_READ` 中，允许节点在条件边中读取包含自身写入的状态。关键设计：

- 当 `fresh=True` 时，拷贝所有 channel 并应用当前任务的写入，然后读取拷贝；
- 否则直接读取 channel 当前值（不包含未提交的写入）。

### _loop.py —— 循环控制

#### PregelLoop.tick() —— 单步执行

```python
def tick(self) -> bool:
    if self.step > self.stop:
        self.status = "out_of_steps"
        return False

    self.tasks = prepare_next_tasks(...)
    
    if not self.tasks:
        self.status = "done"
        return False

    if self.interrupt_before and should_interrupt(...):
        self.status = "interrupt_before"
        raise GraphInterrupt()

    return True
```

`tick()` 的职责是准备任务和执行前检查，并不实际运行任务。它返回 `True` 表示需要继续循环。注意 `raise GraphInterrupt()` 会被 `_suppress_interrupt` 捕获并转换为优雅的中断处理。

#### PregelLoop.after_tick() —— 步后处理

```python
def after_tick(self) -> None:
    writes = [w for t in self.tasks.values() for w in t.writes]
    self.updated_channels = apply_writes(...)
    
    if not self.updated_channels.isdisjoint(output_keys):
        self._emit("values", map_output_values, ...)
    
    self.checkpoint_pending_writes.clear()
    self.is_replaying = False
    self._put_checkpoint({"source": "loop"})
    
    if self.interrupt_after and should_interrupt(...):
        self.status = "interrupt_after"
        raise GraphInterrupt()
```

执行顺序至关重要：先 `apply_writes`，再 `_put_checkpoint`，最后检查中断。这保证了即使发生中断，checkpoint 也已保存。

#### SyncPregelLoop 上下文管理器

```python
class SyncPregelLoop(PregelLoop, AbstractContextManager):
    def __enter__(self) -> Self:
        # 加载 checkpoint
        if self.checkpoint_config[CONF].get(CONFIG_KEY_CHECKPOINT_ID):
            saved = self.checkpointer.get_tuple(self.checkpoint_config)
        else:
            saved = self.checkpointer.get_tuple(self.checkpoint_config)
        # 恢复状态
        self.checkpoint = saved.checkpoint
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = [...]
        # 启动后台执行器
        self.submit = self.stack.enter_context(BackgroundExecutor(self.config))
        # 从 checkpoint 恢复 channel
        self.channels, self.managed = channels_from_checkpoint(self.specs, self.checkpoint)
        # 注册中断抑制器
        self.stack.push(self._suppress_interrupt)
        # 执行第一步输入处理
        self.updated_channels = self._first(...)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.stack.__exit__(exc_type, exc_value, traceback)
```

`ExitStack` 的使用是关键设计：`BackgroundExecutor` 和 `_suppress_interrupt` 都需要正确的生命周期管理。`stack.push(self._suppress_interrupt)` 将其注册为异常处理器——当 `GraphInterrupt` 被抛出时，`_suppress_interrupt` 会被调用，保存 checkpoint 并抑制异常（返回 `True`）。

### _runner.py —— 并发执行

#### FuturesDict

```python
class FuturesDict(Generic[F, E], dict[F, PregelExecutableTask | None]):
```

这是一个泛型字典，键是 `Future`，值是任务或 `None`（`None` 表示 waiter 任务）。核心机制：

```python
def __setitem__(self, key, value):
    super().__setitem__(key, value)
    if value is not None:
        with self.lock:
            self.event.clear()
            self.counter += 1
        key.add_done_callback(partial(self.on_done, value))
```

每当添加一个任务 Future，`counter` 递增，`event` 清除（阻塞等待者）。当 Future 完成时：

```python
def on_done(self, task, fut):
    try:
        if cb := self.callback():
            cb(task, _exception(fut))
    finally:
        with self.lock:
            self.done.add(fut)
            self.counter -= 1
            if self.counter == 0 or _should_stop_others(self.done):
                self.event.set()
```

`counter == 0` 表示所有任务完成；`_should_stop_others` 返回 `True` 表示有任务失败需要取消其余任务。两种情况都设置 `event`，唤醒等待者。

#### PregelRunner.tick() —— 生成器调度

```python
def tick(self, tasks, *, reraise=True, timeout=None, ...) -> Iterator[None]:
    tasks = tuple(tasks)
    futures = FuturesDict(...)
    yield  # 将控制权交给调用者
    # 快速路径：单任务
    if len(tasks) == 1 and timeout is None and get_waiter is None:
        run_with_retry(t, ...)
        if not futures:
            return
    # 调度多任务
    for t in tasks:
        fut = self.submit()(run_with_retry, t, ...)
        futures[fut] = t
    # 等待完成
    while len(futures) > (1 if get_waiter else 0):
        done, inflight = concurrent.futures.wait(
            futures, return_when=FIRST_COMPLETED, ...
        )
        if _should_stop_others(done):
            break
        yield  # 交还控制权，允许流式输出
```

`yield` 的使用实现了协作式调度：每次有任务完成或需要检查状态时，生成器暂停，让上层循环可以处理流式输出。这是两层生成器模式——外层 `PregelLoop` 驱动步循环，内层 `PregelRunner.tick()` 驱动任务完成通知。

#### _should_stop_others() —— 失败检测

```python
def _should_stop_others(done: set[F]) -> bool:
    for fut in done:
        if fut.cancelled():
            continue
        elif exc := fut.exception():
            if not isinstance(exc, GraphBubbleUp) and fut not in SKIP_RERAISE_SET:
                return True
    return False
```

只有非 `GraphBubbleUp`（`GraphInterrupt` 的基类）且不在 `SKIP_RERAISE_SET` 中的异常才触发取消。`GraphInterrupt` 是正常的中断信号，不应取消其他任务。`SKIP_RERAISE_SET` 中的 Future 是 `call()` 创建的子任务，其异常应该传播到父任务而非全局取消。

## 运行原理

### 完整的 BSP 超步循环

```
                    ┌─────────────────────────┐
                    │     PregelLoop 循环      │
                    └─────────────────────────┘
                              │
               ┌──────────────▼───────────────┐
               │  1. prepare_next_tasks()     │
               │     - 消费 TASKS channel     │
               │     - 检查 PULL 触发条件     │
               │     - 生成 PregelExecutableTask│
               └──────────────┬───────────────┘
                              │
               ┌──────────────▼───────────────┐
               │  2. should_interrupt(before)  │
               │     - 检查 interrupt_before   │
               │     - 抛出 GraphInterrupt     │
               └──────────────┬───────────────┘
                              │
               ┌──────────────▼───────────────┐
               │  3. PregelRunner 执行任务     │
               │     - 提交到线程池/协程       │
               │     - 等待完成或失败          │
               │     - yield 交还控制权        │
               └──────────────┬───────────────┘
                              │
               ┌──────────────▼───────────────┐
               │  4. after_tick()              │
               │     - apply_writes()          │
               │     - 更新 channel_versions   │
               │     - _put_checkpoint()       │
               │     - should_interrupt(after) │
               └──────────────┬───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ 有更多任务？       │
                    │ Y → 回到步骤1      │
                    │ N → 结束循环       │
                    └───────────────────┘
```

数据流在每个超步中遵循严格的 "准备-检查-执行-写入-保存-检查" 顺序。`versions_seen` 的更新发生在 `apply_writes` 的开头（而非末尾），确保任务触发条件基于**本步开始时**的版本号。

## 实现细节

### Channel 版本号增量机制

版本号通过 `get_next_version` 函数生成。默认实现是简单的自增：

```python
def increment(current: int | None, channel: None) -> int:
    return current + 1 if current is not None else 1
```

当使用 checkpointer 时，版本号基于 `xxhash`——更短的哈希值，适合存储和比较。所有 channel 在同一步内获得相同的 `next_version`，这简化了版本比较逻辑。

### versions_seen 驱动的触发判定

核心判定在 `_triggers()` 中：`versions.get(chan, null_version) > seen.get(chan, null_version)`。这确保：
- 首次执行（`seen is None`）时，只要 channel 有值就触发；
- 后续执行时，只有 channel 版本号严格大于 seen 时才触发；
- 同一步内多个任务写入同一 channel 时，版本号只增加一次（因为所有写入在同一步应用）。

### PULL 与 PUSH 任务的区别

| 特性 | PULL 任务 | PUSH 任务 |
|------|----------|----------|
| 触发方式 | channel 版本变化 | `Send` 对象 |
| 路径格式 | `(PULL, node_name)` | `(PUSH, idx)` 或 `(PUSH, path, idx, task_id, Call)` |
| 输入来源 | 从 channel 读取 | `Send.arg` 或 `Call.input` |
| 任务 ID | 基于 checkpoint_ns + step + name + triggers | 基于 checkpoint_ns + step + name + PUSH + idx |

### Checkpoint 保存与异步持久化

`_put_checkpoint()` 的关键设计是"等待前一个保存完成"：

```python
self._put_checkpoint_fut = self.submit(
    self._checkpointer_put_after_previous,
    getattr(self, "_put_checkpoint_fut", None),  # 前一个保存的 Future
    self.checkpoint_config,
    copy_checkpoint(self.checkpoint),
    ...
)
```

`_checkpointer_put_after_previous` 先等待 `prev.result()`，再执行新的 `put()`。这保证了 checkpoint 按顺序写入存储，即使保存是异步的。`copy_checkpoint()` 创建深拷贝，防止后续修改影响正在保存的数据。

`durability` 参数控制保存时机：
- `"exit"` 模式下，只在循环退出时保存；
- 其他模式下，每步都保存。

### PregelRunner 的协作生成器模式

`tick()` 返回 `Iterator[None]`，调用者通过 `next()` 推进。每次 `yield` 都是让上层处理流式输出的机会。这种模式避免了回调地狱——上层代码可以按需决定何时推进执行、何时处理输出。

## 动手实验

### 实验 1：手动模拟一个 BSP 超步

```python
from collections import defaultdict
from langgraph.channels.last_value import LastValueChannel
from langgraph.pregel._algo import apply_writes, increment

# 创建两个 channel
channels = {"messages": LastValueChannel(int), "result": LastValueChannel(int)}
channels["messages"].update([42])

# 创建 checkpoint
checkpoint = {
    "id": "test-checkpoint",
    "v": 1,
    "ts": "",
    "channel_values": {},
    "channel_versions": {"messages": 1, "result": 0},
    "versions_seen": {"agent": {"messages": 0}},
    "pending_sends": [],
}

# 模拟一个任务的写入
class FakeWrites:
    path = ("PULL", "agent")
    name = "agent"
    writes = [("result", 100)]
    triggers = ("messages",)

# 应用写入
updated = apply_writes(
    checkpoint, channels, [FakeWrites()],
    increment, {"messages": ["agent"], "result": ["agent"]},
)

print(f"Updated channels: {updated}")
print(f"Channel versions: {checkpoint['channel_versions']}")
print(f"Versions seen: {checkpoint['versions_seen']}")
print(f"Channel values: messages={channels['messages'].get()}, result={channels['result'].get()}")
```

### 实验 2：追踪多步执行中的版本变化

```python
from langgraph.channels.last_value import LastValueChannel
from langgraph.pregel._algo import apply_writes, increment, _triggers, checkpoint_null_version

# 初始化
channels = {"state": LastValueChannel(str)}
channels["state"].update(["hello"])

checkpoint = {
    "id": "test", "v": 1, "ts": "",
    "channel_values": {},
    "channel_versions": {"state": 1},
    "versions_seen": {},
    "pending_sends": [],
}

class Task:
    def __init__(self, name, triggers, writes):
        self.path = (name,)
        self.name = name
        self.triggers = triggers
        self.writes = writes

# 第一步：agent 写入 "world"
class Step1Task(Task):
    path = ("PULL", "agent")
    name = "agent"
    triggers = ("state",)
    writes = [("state", "world")]

updated1 = apply_writes(checkpoint, channels, [Step1Task()], increment, {"state": ["agent"]})
print(f"Step 1 - versions: {checkpoint['channel_versions']}")
print(f"Step 1 - seen: {checkpoint['versions_seen']}")

# 检查第二步是否触发
null_ver = checkpoint_null_version(checkpoint)
class FakeProc:
    triggers = ["state"]

should_fire = _triggers(channels, checkpoint["channel_versions"],
                        checkpoint["versions_seen"].get("agent"), null_ver, FakeProc())
print(f"Step 2 should fire agent? {should_fire}")

# 如果 agent 在第一步已经 seen 版本1，但 channel 版本也是1，则不触发
# 这验证了版本比较的 "严格大于" 语义
```

### 实验 3：观察 _should_stop_others 的行为

```python
import concurrent.futures
from langgraph.pregel._runner import _should_stop_others, FuturesDict
from langgraph.errors import GraphInterrupt

# 模拟：一个成功、一个中断、一个失败
futs = set()

fut_success = concurrent.futures.Future()
fut_success.set_result("ok")
futs.add(fut_success)

fut_interrupt = concurrent.futures.Future()
fut_interrupt.set_exception(GraphInterrupt("user stopped"))
futs.add(fut_interrupt)

fut_error = concurrent.futures.Future()
fut_error.set_exception(RuntimeError("boom"))
futs.add(fut_error)

# GraphInterrupt 不触发取消，RuntimeError 触发
print(f"Should stop others: {_should_stop_others(futs)}")  # True

# 只有 GraphInterrupt 的情况
futs2 = {fut_success, fut_interrupt}
print(f"Should stop with only interrupt: {_should_stop_others(futs2)}")  # False
```