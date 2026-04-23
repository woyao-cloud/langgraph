# 第六章 状态持久化与检查点

检查点（Checkpoint）是 LangGraph 实现状态持久化的核心机制。没有检查点，图只是一次性的计算管道；有了检查点，图就拥有了记忆、恢复和时间旅行的能力。本章将系统讲解检查点的原理、选项和使用方法。

## 6.1 为什么需要检查点

### 没有检查点的局限

- **无法恢复**：图执行过程中如果发生崩溃或超时，所有中间状态丢失，必须从头开始。
- **无法中断**：不支持人机交互（Human-in-the-Loop），因为中断后无法恢复到中断点继续执行。
- **无法多人协作**：多个用户或进程无法共享和延续同一个会话状态。

### 有检查点的能力

- **崩溃恢复**：执行中断后，可以从最后一个检查点恢复，不丢失已完成的进度。
- **人机交互**：在关键节点暂停，等待人工审批或输入，然后继续执行。
- **时间旅行调试**：回溯到任意历史检查点，查看状态变化，甚至从历史状态重新执行。
- **多轮对话**：同一个 `thread_id` 下的多次调用共享状态，实现对话记忆。

## 6.2 Checkpointer 选项

### InMemorySaver：开发与测试用

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

`InMemorySaver` 将检查点存储在内存中的 `defaultdict` 中。程序退出后数据丢失，仅用于本地开发和测试。

### SqliteSaver：轻量级持久化

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 上下文管理器方式，确保资源正确释放
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke(input_data, config)
```

`SqliteSaver` 将检查点持久化到 SQLite 数据库文件。适合单进程、单机部署的轻量级场景。

### AsyncSqliteSaver：异步 SQLite

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    result = await graph.ainvoke(input_data, config)
```

异步版本的 SQLite 检查点保存器，适合 async 应用。

### PostgresSaver / AsyncPostgresSaver：生产环境

```python
# 同步版本
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/dbname"
) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

# 异步版本（推荐用于生产）
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/dbname"
) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

Postgres 检查点保存器是官方推荐的生产环境方案，支持高并发和持久化存储。

### 自定义 Checkpointer

可以通过继承 `BaseCheckpointSaver` 实现自定义检查点存储后端（如 Redis、MongoDB 等）。核心需要实现以下方法：

- `put` / `aput`：保存检查点
- `get_tuple` / `aget_tuple`：获取检查点元组
- `list` / `alist`：列出检查点
- `delete` / `adelete`：删除检查点

## 6.3 thread_id 与配置

### thread_id：会话唯一标识

`thread_id` 是检查点的核心索引键。每个 `thread_id` 对应一个独立的会话状态链。

```python
config_user_a = {"configurable": {"thread_id": "user-a-session-1"}}
config_user_b = {"configurable": {"thread_id": "user-b-session-1"}}

# 用户 A 的会话
graph.invoke({"messages": [("user", "你好")]}, config_user_a)

# 用户 B 的会话，完全独立
graph.invoke({"messages": [("user", "Hi")]}, config_user_b)
```

### checkpoint_id：特定检查点的标识

`checkpoint_id` 指向特定超步的检查点，用于时间旅行和回溯。可通过 `get_state_history()` 或 `get_state()` 获取。

```python
# 获取当前检查点 ID
state = graph.get_state(config)
checkpoint_id = state.config["configurable"]["checkpoint_id"]

# 使用特定检查点 ID 恢复执行
config_with_checkpoint = {
    "configurable": {
        "thread_id": "my-thread",
        "checkpoint_id": checkpoint_id,
    }
}
result = graph.invoke(None, config_with_checkpoint)
```

### 多用户场景

不同 `thread_id` 之间状态完全隔离，这是多用户并发场景的基础：

```python
import uuid

# 每个用户每次会话使用唯一 thread_id
user_configs = {
    "alice": {"configurable": {"thread_id": str(uuid.uuid4())}},
    "bob": {"configurable": {"thread_id": str(uuid.uuid4())}},
}

# Alice 的对话
graph.invoke({"messages": [("user", "你好")]}, user_configs["alice"])

# Bob 的对话，不受 Alice 影响
graph.invoke({"messages": [("user", "Hi")]}, user_configs["bob"])
```

## 6.4 get_state() 和 update_state()

### 查看当前状态

```python
state = graph.get_state(config)

print(state.values)    # 当前状态的字典值
print(state.next)      # 下一步要执行的节点名称元组
print(state.config)    # 包含 thread_id 和 checkpoint_id 的配置
print(state.metadata)  # 元数据：{"step": 3, "source": "loop", ...}
print(state.tasks)     # 待执行任务列表
print(state.interrupts) # 待处理的中断列表
```

### 查看下一步要执行的节点

```python
state = graph.get_state(config)
if state.next:
    print(f"接下来将执行：{', '.join(state.next)}")
```

这在人机交互场景中非常实用。例如图在审批节点前中断后，可以检查 `state.next` 来确认图将在哪里继续。

### 手动修改状态

`update_state()` 允许你直接修改图的状态，仿佛该更新来自某个特定节点。这在调试、回滚和测试中非常有用。

```python
# 修改状态
graph.update_state(
    config,
    values={"messages": [("assistant", "手动插入的回复")]},
    as_node="chatbot",  # 假装更新来自 chatbot 节点
)

# 也可以传 None 作为 values，仅指定 as_node 来推进状态
graph.update_state(config, values=None, as_node="chatbot")
```

**as_node 参数详解：**

`as_node` 决定了状态更新从哪个节点的视角写入。这影响两个方面：
1. **Reducer 计算**：某些 reducer 的行为可能依赖于写入节点的身份。
2. **图路由**：更新后，图会从 `as_node` 对应的出口边继续执行，决定下一个节点。

如果省略 `as_node`，LangGraph 会自动推断为最后一个更新状态的节点。如果有歧义，则必须显式指定。

```python
# 回滚示例：将状态恢复到之前的某个值
graph.update_state(
    config,
    values={"draft": "回滚到之前的草稿版本"},
    as_node="draft_node",
)

# 推进执行：从 as_node 的出口边继续
state_after = graph.get_state(config)
print(state_after.next)  # 根据 draft_node 的出边决定
```

## 6.5 get_state_history()

`get_state_history()` 返回当前 `thread_id` 下所有检查点的迭代器，按时间倒序排列。

```python
for history_state in graph.get_state_history(config):
    step = history_state.metadata.get("step", -1)
    print(f"步骤 {step}:")
    print(f"  状态: {history_state.values}")
    print(f"  下一步: {history_state.next}")
    print(f"  检查点 ID: {history_state.config['configurable']['checkpoint_id']}")
    print()
```

### 从历史检查点恢复

这是检查点最强大的功能之一——"时间旅行"。你可以从任意历史检查点重新执行图，探索不同的路径。

```python
# 找到步骤 2 的检查点
target_state = None
for history_state in graph.get_state_history(config):
    if history_state.metadata.get("step") == 2:
        target_state = history_state
        break

if target_state:
    # 从该检查点恢复并继续执行
    fork_config = {
        "configurable": {
            "thread_id": "fork-thread",  # 使用新 thread_id 避免覆盖原会话
            "checkpoint_id": target_state.config["configurable"]["checkpoint_id"],
        }
    }
    result = graph.invoke({"messages": [("user", "换个方向")]}, fork_config)
```

注意：如果从历史检查点恢复并继续执行，建议使用新的 `thread_id`，否则会覆盖原会话的后续检查点。

### 限制历史查询

```python
# 只查看最近的 5 个检查点
for state in graph.get_state_history(config, limit=5):
    print(state.metadata["step"])

# 查看某个检查点之前的所有检查点
for state in graph.get_state_history(config, before=some_config):
    print(state.metadata["step"])
```

## 6.6 持久化模式（Durability）

`durability` 参数控制检查点写入的时机和可靠性，是性能与安全性的权衡。

```python
result = graph.invoke(
    input_data,
    config,
    durability="sync",  # 可选 "sync" / "async" / "exit"
)
```

### "sync"：同步持久化

在下一个超步开始之前，确保当前超步的检查点已完全写入持久化存储。这是最安全的模式，但也是最慢的，因为每个超步都要等待 I/O 完成。

适用场景：对数据一致性要求极高的金融交易、审批流程等。

### "async"：异步持久化（默认）

检查点在后台异步写入，下一个超步不必等待写入完成。这是默认模式，在安全性和性能之间取得平衡。检查点通常会在下一个超步执行期间完成写入。

适用场景：大多数生产环境，特别是使用 Postgres 等数据库的场景。

### "exit"：退出时持久化

检查点仅在图执行结束时批量写入。这是最快的模式，但风险最大——如果程序崩溃，所有未写入的检查点都会丢失。

适用场景：对性能要求极高且可以容忍数据丢失的场景，如批量数据处理、非关键性分析。

## 6.7 完整示例：带检查点的多轮对话代理

以下是一个完整的可运行示例，展示检查点在多轮对话中的应用：

```python
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def human_review(state: ChatState) -> dict:
    """人机交互节点：暂停等待用户确认"""
    last_msg = state["messages"][-1]
    # 发出中断，等待人工确认
    approval = interrupt(f"请确认是否继续发送此消息：{last_msg.content}")
    if approval == "yes":
        return {}
    return {"messages": [HumanMessage(content="用户取消了此回复")]}

# 构建图
builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot)
builder.add_node("review", human_review)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", "review")
builder.add_edge("review", END)

checkpointer = InMemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["review"],  # 在审查节点前自动中断
)

# === 执行流程 ===
config = {"configurable": {"thread_id": "chat-session-1"}}

# 第一轮对话
result = graph.invoke(
    {"messages": [HumanMessage(content="你好，请介绍一下自己")]},
    config,
)
# 图在 review 节点前中断

# 查看中断状态
state = graph.get_state(config)
print(f"中断中，下一步：{state.next}")  # ('review',)
print(f"待处理中断：{state.interrupts}")

# 提供确认后继续
for event in graph.stream(
    Command(resume="yes"),  # 恢复执行，传入确认值
    config,
    stream_mode="updates",
):
    print(event)

# 第二轮对话（状态自动延续）
result = graph.invoke(
    {"messages": [HumanMessage(content="你能做什么？")]},
    config,
)

# 查看状态历史
print("\n=== 状态历史 ===")
for history in graph.get_state_history(config):
    step = history.metadata.get("step", -1)
    msg_count = len(history.values.get("messages", []))
    print(f"步骤 {step}: 共 {msg_count} 条消息")

# 从历史检查点回溯
target = None
for history in graph.get_state_history(config):
    if history.metadata.get("step") == 1:
        target = history
        break

if target:
    # 创建分支：从步骤 1 开始走不同的路径
    fork_config = {
        "configurable": {
            "thread_id": "chat-fork-1",
            "checkpoint_id": target.config["configurable"]["checkpoint_id"],
        }
    }
    fork_result = graph.invoke(
        {"messages": [HumanMessage(content="换一个话题：讲个笑话")]},
        fork_config,
    )
    print(f"\n分支结果：{fork_result['messages'][-1].content}")
```

此示例完整展示了检查点的三大核心能力：
1. **断点恢复**：图在 `review` 节点前中断，用户通过 `Command(resume=...)` 恢复执行。
2. **多轮对话**：同一 `thread_id` 下的多次 `invoke` 自动延续状态。
3. **时间旅行**：通过 `get_state_history()` 回溯历史，使用新 `thread_id` 从历史检查点创建分支。