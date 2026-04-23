# 第五章 Pregel 引擎——算法与循环

> 源文件：`pregel/_algo.py`、`pregel/_loop.py`、`pregel/_runner.py`

## 5.1 功能概览

Pregel 引擎是 LangGraph 的执行心脏。它将你用 `StateGraph` 定义的节点和边，转化为一种称为**超步（Superstep）**的迭代循环，每一轮超步完成一次"写入-触发-执行"的同步推进。三个核心文件各司其职：

- **`_algo.py`**——纯算法层：决定"哪些节点应该运行"（`prepare_next_tasks`）、"如何把写入应用到 Channel 上"（`apply_writes`）、"何时打断执行"（`should_interrupt`），以及"节点如何读取自己的局部状态"（`local_read`）。它不涉及任何 I/O，纯粹是数据结构变换。
- **`_loop.py`**——循环编排层：`PregelLoop` 是一个有状态的上下文管理器，封装了"加载检查点 -> 准备任务 -> 执行 -> 写入 -> 保存检查点"的完整生命周期。`tick()` 和 `after_tick()` 分别对应超步的前半段（调度）和后半段（提交）。
- **`_runner.py`**——并发执行层：`PregelRunner` 负责在同步或异步环境中并发执行一批 `PregelExecutableTask`，在任务完成时立即提交写入（`commit`），并通过生成器将控制权交还给调用方以实现流式输出。

理解这三层，就理解了 LangGraph 运行时 90% 的行为。

## 5.2 应用场景

### 场景 A：理解你的图为什么循环（或不再循环）

当你的 StateGraph 出现"死循环"或"提前终止"时，根源几乎都在 `prepare_next_tasks` 和 `apply_writes` 的交互中。每个 PregelNode 维护一个 `triggers` 列表，只有当 `triggers` 中至少有一个 Channel 的版本号（`channel_versions`）大于该节点上次看到的版本（`versions_seen`）时，节点才会在下一轮超步被触发。`_triggers()` 函数的逻辑如下：

```python
def _triggers(channels, versions, seen, null_version, proc):
    if seen is None:
        # 从未执行过：只要 trigger channel 有值就触发
        for chan in proc.triggers:
            if channels[chan].is_available():
                return True
    else:
        # 已执行过：版本号严格递增才触发
        for chan in proc.triggers:
            if channels[chan].is_available() and \
               versions.get(chan, null_version) > seen.get(chan, null_version):
                return True
    return False
```

这意味着：如果一个节点的写入没有更新任何 Channel 的值（例如 `update` 方法返回 `False`），那么下一个超步中，下游节点不会被触发。这是理解"循环终止"的关键。当你定义了一个条件边路由到 `END`，图会终止，是因为没有新的写入产生版本号变化。

### 场景 B：用 interrupt_before / interrupt_after 调试

LangGraph 支持在节点执行前后打断执行，底层由 `should_interrupt()` 实现：

```python
should_interrupt(checkpoint, interrupt_nodes, tasks)
```

它检查两件事：(1) 是否有 Channel 自上次中断以来发生了更新（`any_updates_since_prev_interrupt`）；(2) 当前被触发的节点是否在 `interrupt_nodes` 列表中。只有两者同时满足才会打断。这解释了一个常见困惑：如果你设置了 `interrupt_before=["my_node"]` 但 `my_node` 没有被触发（上游没有写入），打断不会发生。

实际使用中，常见做法是在人机协作流程中设置中断：

```python
# 构建图时指定中断点
graph = builder.compile(
    checkpointer=memory_checkpointer,
    interrupt_before=["human_review"],  # 在审核节点前暂停
)

# 第一次调用：执行到 human_review 前中断
result = graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "1"}})

# 人工审核后继续
result = graph.invoke(None, config={"configurable": {"thread_id": "1"}})
```

### 场景 C：recursion_limit 如何防止无限循环

`PregelLoop` 中设置了 `self.stop` 作为超步上限，而 `tick()` 在 `self.step > self.stop` 时将状态设为 `"out_of_steps"` 并返回 `False`。每个超步完成后 `step += 1`，所以递归限制实际上限制了超步的数量，而不是递归调用的深度。默认值为 25。如果你的图需要超过 25 轮超步才能收敛（例如多轮对话 agent），就需要调高这个限制。常见原因是条件边形成环路但没有终止条件。

### 场景 D：并行节点为何看不到彼此的写入

`PregelRunner.tick()` 并发执行同一超步的所有任务。每个任务的写入存储在 `task.writes` 中，只在 `after_tick()` 调用 `apply_writes()` 时才统一应用到 Channel。这意味着在同一个超步中，并行执行的节点 A 无法读到节点 B 的输出——这正是 BSP 模型的核心语义：**超步内并行、超步间同步**。

如果你需要节点 A 的输出影响节点 B，唯一的办法是将 B 放在 A 的下一个超步中执行（通过边连接）。

### 场景 E：检查点的保存时机

`_put_checkpoint()` 在 `after_tick()` 中被调用。关键点在于 `durability` 参数：

- `"exit"`：仅在循环退出时保存检查点，性能最优但崩溃可能丢失状态。
- `"async"` 或其他值：每个超步结束都保存检查点，确保崩溃恢复。

`_put_checkpoint()` 使用 `_checkpointer_put_after_previous` 确保检查点按顺序写入：前一个 `put` 的 `Future` 完成后才开始下一个。这保证了即使在异步保存模式下，检查点的顺序也是正确的。

### 场景 F：Durability 模式的性能与安全权衡

`PregelLoop` 的 `durability` 属性直接影响两个行为：

1. **写入持久化**：`put_writes()` 中，`self.durability != "exit"` 时才调用 `checkpointer_put_writes`。当 durability 为 `"exit"` 时，中间写入不会立即持久化，而是在循环退出时统一保存。
2. **检查点保存**：`_put_checkpoint()` 中，`do_checkpoint` 的条件包含 `exiting or self.durability != "exit"`。

对于开发调试，`"exit"` 模式足够；对于生产环境，使用默认的持久化模式以确保状态安全。在长时间运行的人工审批流程中，建议使用非 `"exit"` 模式，否则如果服务器崩溃，将丢失中断点。

### 场景 G：追踪任务执行顺序

`PregelRunner.tick()` 是一个生成器，每次 `yield` 将控制权交还给调用方。这意味着在流式模式下，调用方可以在每个任务完成时获得中间输出，而不必等待所有任务完成。`_should_stop_others()` 函数会在某个任务抛出非 `GraphBubbleUp` 异常时终止其他所有任务。

这在调试时特别有用：通过 `graph.stream()` 而不是 `graph.invoke()` 执行图，你可以看到每个超步中哪些任务在执行、执行顺序如何、哪些任务最先完成。

## 5.3 Python 进阶

### defaultdict 与 deque

`_algo.py` 使用 `defaultdict(list)` 来按 Channel 名分组写入，而 `prepare_single_task` 中任务的写入缓冲区使用 `deque` 而非 `list`——因为 `deque.extend` 是线程安全的，可以在并发任务中安全追加写入而无需加锁。

### itertools 与确定性排序

`itertools` 模块在 `_algo.py` 中用于 `LazyAtomicCounter`，它内部使用 `itertools.count(0).__next__` 生成唯一的递增整数。这是一个线程安全的惰性计数器实现——首次调用时才创建 `count` 对象，利用了双重检查锁定模式。

### functools.partial 与依赖注入

`local_read` 使用 `functools.partial` 将当前任务的写入信息注入到配置中的 `CONFIG_KEY_READ` 函数。节点内部调用读取状态时，完全不知道底层实现。这是经典的依赖注入模式：

```python
configurable={
    CONFIG_KEY_READ: partial(
        local_read, scratchpad, channels, managed,
        PregelTaskWrites(task_path[:3], name, writes, triggers),
    ),
}
```

### Protocol 类型

`WritesProtocol` 是一个 `Protocol` 类，只定义了 `path`、`name`、`writes`、`triggers` 四个属性。这允许 `PregelTaskWrites`（简单命名元组）和 `PregelExecutableTask`（复杂任务对象）都作为 `apply_writes` 的参数，实现了结构化子类型（鸭子类型的类型安全版本）。

### async/await 与上下文管理器

`SyncPregelLoop` 和 `AsyncPregelLoop` 分别使用 `ExitStack` 和 `AsyncExitStack` 来管理资源（后台线程池/进程池、中断抑制等）。`__enter__` 方法中：

```python
self.submit = self.stack.enter_context(BackgroundExecutor(self.config))
self.stack.push(self._suppress_interrupt)
```

`BackgroundExecutor` 作为上下文管理器进入，确保循环结束时线程池被正确关闭；`_suppress_interrupt` 作为退出回调，在异常发生时保存最终检查点。

### 生成器调度模式

`PregelRunner.tick()` 和 `atick()` 是生成器/异步生成器，通过 `yield` 实现协作式调度。调用方在每次 `yield` 时可以检查输出、处理流式数据，而不需要回调或 Promise 链。这种模式在 Python 中被称为"基于生成器的事件循环"。

### walrus 运算符

`_runner.py` 中大量使用 `:=`（海象运算符），例如：

```python
if cb := self.callback():
    cb(task, _exception(fut))
```

```python
if next_task := schedule_task(task(), scratchpad.call_counter(), Call(...)):
```

这些用法让条件判断和赋值合二为一，在并发代码中避免了临时变量的作用域泄漏。

### weakref 与循环引用

`PregelRunner` 和 `_call` 函数大量使用 `weakref.ref` 和 `weakref.WeakMethod` 来持有回调引用，防止循环引用导致内存泄漏。`FuturesDict` 也使用 `weakref.WeakSet` 来跟踪需要跳过重新抛出的 Future。

## 5.4 代码走读

### prepare_next_tasks() 完整算法

```python
def prepare_next_tasks(checkpoint, pending_writes, processes, channels,
                       managed, config, step, stop, *, for_execution,
                       trigger_to_nodes=None, updated_channels=None, ...):
```

算法步骤：

1. **消费 PUSH 任务**：从 `TASKS` Channel（类型为 `Topic[Send]`）中取出所有 `Send` 对象，为每个创建一个 PUSH 任务。

2. **优化节点筛选**：如果提供了 `trigger_to_nodes` 和 `updated_channels`，只检查被更新 Channel 触发的节点；否则遍历所有节点。

3. **逐节点检查触发条件**：对每个候选节点调用 `_triggers()`，判断其 `triggers` Channel 是否有版本更新。

4. **创建任务对象**：对于被触发的节点，调用 `prepare_single_task()` 创建 `PregelTask`（仅包含元信息，用于检查点）或 `PregelExecutableTask`（包含完整的执行配置）。

关键优化点：当图很大但每步只有少量节点活跃时，`trigger_to_nodes` 可以跳过大量不需要检查的节点。这通过 `updated_channels` 与 `trigger_to_nodes` 的交叉查询实现：

```python
if updated_channels and trigger_to_nodes:
    triggered_nodes = set()
    for channel in updated_channels:
        if node_ids := trigger_to_nodes.get(channel):
            triggered_nodes.update(node_ids)
    candidate_nodes = sorted(triggered_nodes)
```

### apply_writes() 生命周期

```python
def apply_writes(checkpoint, channels, tasks, get_next_version, trigger_to_nodes):
```

1. **排序任务**：按 `task.path[:3]` 排序，保证写入顺序确定性。
2. **更新版本可见性**：将每个任务的 `triggers` Channel 版本记录到 `versions_seen` 中。
3. **计算下一个全局版本号**：取所有 Channel 版本的最大值，递增得到 `next_version`。这是一个关键设计——所有在同一超步中被更新的 Channel 共享同一个新版本号。
4. **消费被读 Channel**：对每个 trigger Channel 调用 `consume()`（例如 `Topic` 类型 Channel 在被消费后会清空）。
5. **按 Channel 分组写入**：`defaultdict(list)` 收集所有非特殊 Channel 的写入（过滤掉 `NO_WRITES`、`PUSH`、`RESUME` 等特殊写入）。
6. **更新 Channel**：对每个有写入的 Channel 调用 `update()`，若更新成功则递增版本号。
7. **通知未更新的 Channel**：对有值但本轮未被写入的 Channel 调用 `update(EMPTY_SEQ)`，通知它们新超步开始了。这确保了所有 Channel 都能感知到超步的推进。
8. **最终化**：如果没有 Channel 更新触发了任何节点（`updated_channels.isdisjoint(trigger_to_nodes)`），对所有 Channel 调用 `finish()`，通知它们图即将结束。

### should_interrupt() 中断检测

```python
def should_interrupt(checkpoint, interrupt_nodes, tasks):
    version_type = type(next(iter(checkpoint["channel_versions"].values()), None))
    null_version = version_type()  # 获取该版本类型的"零值"
    seen = checkpoint["versions_seen"].get(INTERRUPT, {})
    any_updates_since_prev_interrupt = any(
        version > seen.get(chan, null_version)
        for chan, version in checkpoint["channel_versions"].items()
    )
    return (
        [task for task in tasks if ...]
        if any_updates_since_prev_interrupt
        else []
    )
```

中断机制使用了一个特殊的 `INTERRUPT` 键在 `versions_seen` 中记录上次中断时的版本号。只有在版本号推进后才允许新的中断，防止重复中断。当调用 `Command(resume=...)` 继续执行时，`_first()` 方法会将所有当前 Channel 版本写入 `versions_seen[INTERRUPT]`，这样在下一次 `should_interrupt` 检查时，如果没有新的写入，就不会重复中断。

### PregelLoop.tick() 与 after_tick()

`tick()` 执行超步前半段：

1. 检查 `step > stop`（超步数耗尽）。
2. 调用 `prepare_next_tasks` 准备任务。
3. 如果没有任务，设置 `status = "done"` 并返回 `False`。
4. 匹配上一轮未完成的写入到当前任务（`_match_writes`）。
5. 检查是否需要打断（`interrupt_before`）。
6. 返回 `True` 表示还有更多工作要做。

`after_tick()` 执行超步后半段：

1. 收集所有任务的写入。
2. 调用 `apply_writes` 将写入应用到 Channel。
3. 发出 `values` 流式输出（如果有输出 Channel 被更新）。
4. 清空 `checkpoint_pending_writes`。
5. 保存检查点（`_put_checkpoint`）。
6. 检查是否需要打断（`interrupt_after`）。
7. 清除 `resuming` 标记。

### PregelRunner 并发执行

`PregelRunner.tick()` 的核心流程：

```python
# 1. 创建 FuturesDict 追踪所有任务的 Future
futures = FuturesDict(callback=weakref.WeakMethod(self.commit), ...)
# 2. yield 让调用方有机会设置流式输出
yield
# 3. 快速路径：单任务直接在当前线程执行
if len(tasks) == 1 and timeout is None and get_waiter is None:
    run_with_retry(t, ...)
    self.commit(t, None)
# 4. 多任务路径：提交到线程池
for t in tasks:
    fut = self.submit()(run_with_retry, t, ...)
    futures[fut] = t
# 5. 等待任意一个完成
while len(futures) > ...:
    done, inflight = concurrent.futures.wait(futures, FIRST_COMPLETED, ...)
    if _should_stop_others(done):
        break
    yield  # 让出控制权
```

`atick()` 是异步版本，使用 `asyncio.wait` 代替 `concurrent.futures.wait`。

### FuturesDict：自动追踪的 Future 集合

`FuturesDict` 继承自 `dict`，在 `__setitem__` 时自动为 Future 注册 `on_done` 回调，并维护一个计数器 `counter` 和事件 `event`。当所有 Future 都完成时，`event.set()` 被触发，允许 `_panic_or_proceed` 检查所有结果。`SKIP_RERAISE_SET` 是一个 `WeakSet`，用于标记那些异常应该由父任务处理而非全局抛出的 Future（例如 `_call` 产生的子任务）。

## 5.5 实现原理

### 完整 BSP 循环的数据流

```
                  ┌─────────────────────────────────────┐
                  │           PregelLoop 生命周期          │
                  │                                     │
                  │  __enter__                           │
                  │    |                                 │
                  │    v                                 │
                  │  _first() ─── 加载检查点、映射输入     │
                  │    |                                 │
                  │    v                                 │
                  │  ┌─── tick() <─────────────────────┐  │
                  │  │    |                            │  │
                  │  │    v                            │  │
                  │  │  prepare_next_tasks()           │  │
                  │  │    |                            │  │
                  │  │    v                            │  │
                  │  │  should_interrupt()?            │  │
                  │  │    |                            │  │
                  │  │    v                            │  │
                  │  │  PregelRunner.tick()            │  │
                  │  │  (并发执行所有任务)               │  │
                  │  │    |                            │  │
                  │  │    v                            │  │
                  │  │  after_tick()                    │  │
                  │  │    |                            │  │
                  │  │    ├── apply_writes()           │  │
                  │  │    ├── _put_checkpoint()        │  │
                  │  │    ├── should_interrupt()?       │  │
                  │  │    |                            │  │
                  │  │    └── step += 1 ───────────────┘  │
                  │  │                                    │
                  │  v  (无更多任务或被中断)               │
                  │  __exit__                            │
                  └─────────────────────────────────────┘
```

### 版本驱动的任务触发

每个 Channel 维护一个单调递增的版本号（`channel_versions`），每个节点记录它"上次看到"的版本（`versions_seen`）。只有当 `channel_versions[chan] > versions_seen[node][chan]` 时，节点才被认为"有新数据可读"。这种设计使得：

- **去重**：同一数据不会触发同一节点两次。
- **确定性**：版本的递增顺序由 `get_next_version` 函数决定，保证相同的写入顺序产生相同的触发结果。
- **中断恢复**：检查点中保存了完整的 `versions_seen`，恢复时可以精确重现上次中断的位置。

### PULL vs PUSH 任务

LangGraph 区分两种任务类型：

- **PULL 任务**：由 Channel 版本变化触发。节点定义了 `triggers`（关联的 Channel 列表），当这些 Channel 有新数据时，节点被"拉"入执行。路径格式为 `(PULL, node_name)`。
- **PUSH 任务**：由 `Send` 对象触发。当一个节点通过 `Send(node, arg)` 动态调度另一个节点时，会创建 PUSH 任务。PUSH 任务的路径格式为 `(PUSH, parent_path, write_idx, parent_id, call_or_none)`。

`prepare_single_task` 根据 `task_path[0]` 的值（`PULL` 或 `PUSH`）分派到不同的准备逻辑。

### 检查点保存流程

```
after_tick()
  |
  ├── 收集所有 task.writes
  ├── apply_writes() ─── 更新 Channel 和版本号
  ├── 清空 checkpoint_pending_writes
  ├── _put_checkpoint()
  |     |
  |     ├── create_checkpoint() ─── 从 Channel 快照创建新检查点
  |     |
  |     └── _checkpointer_put_after_previous()
  |           |
  |           ├── 等待前一个 put Future 完成
  |           └── 调用 checkpointer.put() ─── 保存到存储后端
  |
  └── should_interrupt()? ─── 检查是否需要在节点执行后中断
```

关键点：`_checkpointer_put_after_previous` 确保检查点按步骤顺序保存。它接收前一个保存操作的 Future，等待其完成后再开始当前保存。这防止了步骤 3 的检查点先于步骤 2 写入存储。

## 5.6 动手实验

### 实验 1：模拟 BSP 步骤

以下实验模拟了 BSP 超步的核心逻辑——版本驱动的触发机制：

```python
from collections import defaultdict

# 模拟 Channel 版本
channel_versions = {"messages": 1, "context": 0}
# 模拟节点上次看到的版本
versions_seen = {
    "agent": {"messages": 0, "context": 0},
    "tool":  {"messages": 0, "context": 0},
}

# 定义节点触发器
nodes = {
    "agent": {"triggers": ["messages"]},
    "tool":  {"triggers": ["context"]},
}

def simulate_trigger(channel_versions, versions_seen, node_name, triggers):
    """检查节点是否应该被触发（模拟 _triggers）"""
    null_version = 0
    seen = versions_seen.get(node_name)
    if seen is None:
        return True
    for chan in triggers:
        current = channel_versions.get(chan, null_version)
        last_seen = seen.get(chan, null_version)
        if current > last_seen:
            return True
    return False

# Step 1: messages 被更新，版本从 0 -> 1
print("Step 1:")
for name, node in nodes.items():
    triggered = simulate_trigger(
        channel_versions, versions_seen, name, node["triggers"]
    )
    print(f"  {name}: triggered={triggered}")
# agent: triggered=True  (messages v1 > v0)
# tool:  triggered=False (context v0 == v0)

# Step 2: agent 执行后更新 context，版本从 0 -> 2
channel_versions["context"] = 2
versions_seen["agent"]["messages"] = 1  # agent 已看过 messages v1

print("\nStep 2:")
for name, node in nodes.items():
    triggered = simulate_trigger(
        channel_versions, versions_seen, name, node["triggers"]
    )
    print(f"  {name}: triggered={triggered}")
# agent: triggered=False (messages v1 == v1)
# tool:  triggered=True  (context v2 > v0)
```

### 实验 2：追踪版本变化

以下实验追踪 `apply_writes` 中版本变化的完整生命周期：

```python
from collections import defaultdict

def simulate_apply_writes(tasks_writes, channel_versions, versions_seen,
                          trigger_to_nodes):
    """模拟 apply_writes 的核心逻辑"""
    updated_channels = set()

    # Step 1: 更新 versions_seen
    for task_name, triggers, writes in tasks_writes:
        if task_name not in versions_seen:
            versions_seen[task_name] = {}
        for chan in triggers:
            if chan in channel_versions:
                versions_seen[task_name][chan] = channel_versions[chan]

    # Step 2: 计算下一个版本号
    max_version = max(channel_versions.values()) if channel_versions else 0
    next_version = max_version + 1

    # Step 3: 按通道分组写入
    pending = defaultdict(list)
    for task_name, triggers, writes in tasks_writes:
        for chan, val in writes:
            pending[chan].append(val)

    # Step 4: 更新 Channel 版本
    for chan, vals in pending.items():
        channel_versions[chan] = next_version
        updated_channels.add(chan)
        print(f"  Channel '{chan}' updated: v{next_version-1} -> v{next_version}")

    # Step 5: 找出被触发的节点
    triggered = set()
    for chan in updated_channels:
        if node_ids := trigger_to_nodes.get(chan):
            triggered.update(node_ids)
    print(f"  Triggered nodes: {triggered}")
    return updated_channels

# 初始状态
channel_versions = {"messages": 0}
versions_seen = {}
trigger_to_nodes = {"messages": ["agent"]}

# 超步 1
print("=== Superstep 1: User input ===")
simulate_apply_writes(
    [("__input__", [], [("messages", "Hello")])],
    channel_versions, versions_seen, trigger_to_nodes
)

# 超步 2
print("\n=== Superstep 2: Agent responds ===")
simulate_apply_writes(
    [("agent", ["messages"], [("messages", "Hi there!")])],
    channel_versions, versions_seen, trigger_to_nodes
)

# 超步 3: agent 写入 context 通道
channel_versions["context"] = 0
trigger_to_nodes["context"] = ["tool"]
print("\n=== Superstep 3: Agent writes to context ===")
simulate_apply_writes(
    [("agent", ["messages"], [("context", "use_search")])],
    channel_versions, versions_seen, trigger_to_nodes
)
```

### 实验 3：should_interrupt 行为验证

```python
def simulate_should_interrupt(checkpoint, interrupt_nodes, task_names):
    """模拟 should_interrupt 的逻辑"""
    null_version = 0
    seen = checkpoint["versions_seen"].get("__interrupt__", {})

    any_updates = any(
        version > seen.get(chan, null_version)
        for chan, version in checkpoint["channel_versions"].items()
    )

    if not any_updates:
        return []

    if interrupt_nodes == "*":
        return task_names
    else:
        return [t for t in task_names if t in interrupt_nodes]

checkpoint = {
    "channel_versions": {"messages": 2, "context": 1},
    "versions_seen": {
        "__interrupt__": {"messages": 1},  # 上次中断时 messages=v1
        "agent": {"messages": 2},
    },
}

# context v1 > null_version=0, 有更新
result = simulate_should_interrupt(checkpoint, ["tool"], ["agent", "tool"])
print(f"Interrupt before: {result}")  # ['tool']

# 更新 INTERRUPT seen 到最新
checkpoint["versions_seen"]["__interrupt__"]["messages"] = 2
checkpoint["versions_seen"]["__interrupt__"]["context"] = 1
result = simulate_should_interrupt(checkpoint, ["agent"], ["agent", "tool"])
print(f"Interrupt after resume: {result}")  # [] (没有新更新)
```

这些实验揭示了 BSP 引擎的核心机制：版本号驱动的触发、超步内隔离的写入语义、以及中断的条件判断。理解这些，你就能预测图的行为，而不是靠试错来调试。