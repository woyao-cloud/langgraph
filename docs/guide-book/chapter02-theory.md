# 第二章：理论基础

LangGraph 不是一个凭空出现的设计。它的核心概念——图、状态、超步、检查点——都有深厚的理论根基。理解这些根基，你不仅能"知其然"，还能"知其所以然"：为什么 LangGraph 选择这样的架构，而不是别的？

## 2.1 BSP 模型：从 Google Pregel 到 LangGraph

### 什么是 BSP？

BSP（Bulk Synchronous Parallel，批量同步并行）是一种并行计算模型，由 Leslie Valiant 在 1990 年提出。它的核心思想很简单：

> 把计算分成一系列**超步**（superstep），每个超步包含三个阶段：**计算 → 通信 → 同步屏障**。

用伪代码表示：

```
while 未完成:
    # 阶段 1：每个节点独立计算
    for each node in parallel:
        node.compute()
    
    # 阶段 2：节点间交换消息
    for each message in queue:
        deliver(message)
    
    # 阶段 3：全局同步屏障——所有节点完成后才进入下一轮
    barrier()
```

这个模型的威力在于：**节点间通信只在超步边界发生，超步内部每个节点是独立计算的。** 这意味着：

- **确定性**：给定相同的初始状态和输入，执行结果总是相同的。
- **可恢复性**：每个超步结束后，系统处于一个一致的、可记录的状态。
- **可推理性**：你不需要考虑并发竞争——因为超步之间没有重叠。

### Pregel：BSP 在图计算中的实践

Google 在 2010 年发表了 Pregel 论文，将 BSP 模型应用到大图计算（如 PageRank）。Pregel 的核心抽象：

- **Vertex**（顶点）：图中的计算单元，每个顶点维护自己的状态。
- **Message**（消息）：顶点之间通过发送消息通信。
- **Superstep**（超步）：每轮所有顶点并行计算，然后交换消息，最后同步。

一个 Pregel 超步的执行过程：

```
超步 S：
  1. 每个 Vertex V 读取在超步 S-1 中收到的所有消息
  2. V 执行用户定义的 Compute() 函数，可能：
     - 更新自己的状态
     - 发送消息给其他顶点（在超步 S+1 才送达）
     - 投票终止（Vote to halt）
  3. 所有 Vertex 完成后，进入超步 S+1
```

### LangGraph 的 BSP 实现

LangGraph 的执行模型直接继承了 Pregel 的 BSP 设计：

| Pregel 概念 | LangGraph 对应 | 说明 |
|------------|---------------|------|
| Vertex | Node | 图中的计算单元 |
| Message | Channel 写入 | 节点间通过通道传递数据 |
| Superstep | 一个执行步骤 | 所有并行节点执行一轮 |
| Compute() | 节点函数 | 处理输入、更新状态 |
| Vote to halt | 到达 END 节点 | 终止条件 |

LangGraph 的执行循环：

```
while 图未到达终点:
    # 超步开始
    1. 确定本轮要执行的节点（基于边的拓扑和条件）
    2. 并行执行所有就绪的节点
    3. 收集所有节点的输出，通过通道合并到共享状态
    4. 检查点保存当前状态（如果配置了检查点）
    # 超步结束，进入下一轮
```

**为什么 BSP 是正确的选择？** 因为 LLM 工作流天然适合批量同步模型：

- LLM 调用是耗时的（几百毫秒到几十秒），并行执行多个调用可以显著减少总时间。
- 但并行执行需要确保：所有并行节点的输出合并后，状态是一致的——这正是 BSP 同步屏障提供的保证。
- LLM 工作流需要可恢复性——如果中途崩溃，你需要从一个一致的状态恢复，而不是从一个不确定的中间状态。

## 2.2 Actor 模型与消息传递

### Actor 模型是什么？

Actor 模型由 Carl Hewitt 在 1973 年提出，是并发计算的另一个重要模型。它的核心思想是：

> 一切都是 Actor。每个 Actor 有自己的私有状态，只能通过**异步消息**与其他 Actor 通信。Actor 收到消息后，可以：创建新 Actor、发送消息给其他 Actor、决定下一个行为。

```python
# Actor 模型的伪代码
class Actor:
    def receive(self, message):
        # 处理消息
        self.state = update(self.state, message)
        # 发送新消息
        send(other_actor, new_message)
        # 可能创建新 Actor
        new_actor = Actor(behavior)
```

Actor 模型的关键特征：

- **封装性**：每个 Actor 的状态是私有的，只能通过消息传递访问。
- **异步性**：消息发送后不等待回复，立即继续执行。
- **位置透明性**：Actor 可以在同一进程内，也可以在分布式节点上。

### LangGraph 的 Send：受 Actor 模型启发

LangGraph 的 `Send` 功能直接借鉴了 Actor 模型的消息传递思想：

```python
from langgraph.types import Send

def router_node(state: State):
    # 动态决定要"激活"哪些节点——类似 Actor 模型中向多个 Actor 发消息
    return [
        Send("researcher", {"topic": "AI 安全"}),
        Send("researcher", {"topic": "AI 伦理"}),
        Send("researcher", {"topic": "AI 监管"}),
    ]
```

`Send` 允许你在运行时动态决定图中的哪些节点要执行，以及传递什么数据——这正是 Actor 模型中"向特定 Actor 发送特定消息"的思想。

### 为什么 LangGraph 选择 BSP 而非纯 Actor？

纯 Actor 模型有两个特性在 LLM 工作流中是有问题的：

1. **完全异步**——Actor 之间没有全局同步点，消息可以随时到达。这意味着状态在不同时间点是不一致的，难以做检查点和恢复。

2. **非确定性**——消息到达的顺序取决于调度器，同一组 Actor 的执行顺序可能每次不同。这在调试和复现 LLM 应用时是灾难性的。

LangGraph 的选择是：**借用 Actor 模型的消息传递思想（Send），但用 BSP 的同步屏障来保证确定性和可恢复性。** 这是一种务实的折中——你得到了 Actor 模型的灵活性（动态路由、fan-out），但没有放弃 BSP 的确定性保证。

| 特性 | 纯 Actor 模型 | LangGraph |
|------|-------------|-----------|
| 通信方式 | 异步消息 | 同步通道 + Send |
| 执行顺序 | 非确定 | 确定性（超步内并行） |
| 状态一致性 | 弱（最终一致） | 强（超步边界一致） |
| 检查点 | 困难 | 天然支持（每超步一次） |
| 动态路由 | 天然支持 | 通过 Send 支持 |

## 2.3 有限状态机与有状态图

### 传统 FSM

有限状态机（Finite State Machine, FSM）是计算机科学中最基础的模型之一：

```
状态集合 S = {s0, s1, s2, ...}
转移集合 T = {(s_i, event, s_j), ...}
初始状态 s0
接受状态集合 F ⊆ S
```

FSM 的核心特征：

- **离散状态**：系统在任何时刻处于有限个状态之一。
- **确定性转移**：给定当前状态和输入，下一个状态是确定的。
- **无内部存储**：状态本身不携带复杂的数据结构（只有"哪个状态"的信息）。

### LangGraph 的 StateGraph：不只是 FSM

LangGraph 的 `StateGraph` 借用了 FSM 的思想，但做了关键扩展：

**1. 状态携带数据，不只是标签**

传统 FSM 的状态是一个标签（如 "idle"、"running"、"stopped"），而 StateGraph 的状态是一个完整的数据结构：

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 完整的对话历史
    current_task: str                          # 当前任务描述
    scores: dict                               # 评分数据
    retry_count: int                           # 重试次数
```

这意味着系统的状态不仅仅是"在哪一步"，还包含"到目前为止积累了什么数据"。

**2. 条件边实现动态转移**

传统 FSM 的转移是固定的：在状态 s_i，如果收到事件 e，一定转移到状态 s_j。StateGraph 的条件边允许运行时决定：

```python
def should_continue(state: State) -> str:
    if state["retry_count"] >= 3:
        return "give_up"
    if state["scores"]["quality"] > 0.8:
        return "success"
    return "retry"

graph.add_conditional_edges("evaluate", should_continue)
```

**3. 并行执行和子图**

传统 FSM 在任何时刻只能处于一个状态。StateGraph 允许同时执行多个节点（并行），并且支持子图（另一个 StateGraph 作为节点嵌入），这是传统 FSM 无法表达的。

### StateGraph vs 传统 FSM

| 特性 | 传统 FSM | StateGraph |
|------|---------|-----------|
| 状态表示 | 离散标签 | 带数据的 TypedDict |
| 转移方式 | 固定转移表 | 条件函数 |
| 并行性 | 不支持 | 支持 |
| 嵌套 | 不支持 | 子图 |
| 循环 | 有限状态 → 有限转移 | 递归限制防止无限循环 |

**为什么 StateGraph 不是传统 FSM？** 因为它结合了三种模型的优点：

- FSM 的**离散步骤**思想（每一步是一个明确的节点）
- 数据流图的**数据传递**思想（状态在节点间流动和更新）
- BSP 的**并行同步**思想（超步内的并行执行和超步间的同步）

## 2.4 检查点理论：事件溯源与 CQRS

### 为什么需要检查点？

LLM 工作流有两个特征让检查点成为必需：

1. **长时间运行**——一个复杂工作流可能涉及数十次 LLM 调用，总耗时可能从几分钟到几小时。如果在第 20 步崩溃了，你不希望从头来过。

2. **需要人机协作**——在某些步骤，你需要暂停执行，等待人工输入，然后继续。检查点让你可以保存当前状态，在任意时间点恢复。

### 事件溯源 vs 快照

分布式系统中有两种主要的状态持久化方式：

**事件溯源（Event Sourcing）**：不保存当前状态，而是保存所有改变状态的事件。要恢复状态，需要重放所有事件。

```
事件日志：
  [创建订单, 添加商品A, 添加商品B, 修改地址, 应用折扣]
  
恢复状态 = 重放所有事件 → 当前状态
```

**状态快照（Snapshot）**：定期保存当前完整状态。恢复时直接读取最近的快照。

```
快照：
  Step 5: State = {items: [A, B], address: "...", discount: 0.1}
  
恢复状态 = 读取快照 → 直接得到当前状态
```

### LangGraph 为什么选择快照？

LangGraph 选择在每个超步结束时保存**完整状态快照**，而不是事件日志。这个选择基于三个理由：

**1. 简单性**——快照恢复只需要一次读取操作，不需要重放复杂的事件序列。对于 LLM 应用开发者来说，"从第 N 步恢复"比"重放 N 个事件"更容易理解和调试。

**2. 可恢复性**——LLM 调用是昂贵的（时间和金钱）。事件溯源要求重放，意味着你可能需要重新执行 LLM 调用。而快照恢复是零成本的——状态已经保存在那里了。

**3. 可中断性**——当你需要暂停执行等待人工输入时，快照提供了一个自然的中断点：保存当前快照，等人工输入后，从快照恢复并注入新的输入。

```python
from langgraph.checkpoint.memory import MemorySaver

# 配置检查点
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# 第一次运行——执行到某个中断点
config = {"configurable": {"thread_id": "conversation-1"}}
result = app.invoke({"messages": [...]}, config)

# 之后从检查点恢复——不需要重新执行之前的步骤
result2 = app.invoke({"messages": [...]}, config)  # 自动从上次检查点继续
```

### CQRS 与检查点

CQRS（Command Query Responsibility Segregation）是事件溯源的伴生模式，核心思想是读写分离：写入模型和读取模型可以不同。

LangGraph 的检查点设计暗合了 CQRS 的思想：

- **写入模型**：节点执行时更新状态，通过 reducer 合并写入。
- **读取模型**：检查点保存的是合并后的完整状态，可以随时查询。
- **分离点**：超步边界就是写入和读取的分离点——超步内并行写入，超步边界同步读取。

## 2.5 图论基础：有向图、DAG 与拓扑排序

### 有向图

LangGraph 的 StateGraph 是一个**有向图**（Directed Graph）：

- **顶点**（Vertex）：即节点（Node），如 `chatbot`、`researcher`、`summarizer`。
- **有向边**（Directed Edge）：从一个节点指向另一个节点，表示执行转移的方向。
- **允许环**：StateGraph 允许环（循环），这是它与 DAG 的关键区别。

```
START → chatbot → evaluator → (quality > 0.8?) 
                                   ├── yes → END
                                   └── no  → chatbot  ← 环！
```

**为什么允许环？** 因为 LLM 工作流天然需要迭代——生成、评估、修正、再评估。禁止环意味着禁止自我修正，这对于 AI 应用来说是严重的限制。

### 防止无限循环

虽然允许环，但 LangGraph 通过**递归限制**（recursion limit）防止无限循环：

```python
# 默认递归限制是 25 步
result = app.invoke(input, config={"recursion_limit": 50})
```

每次执行一个超步，递归计数器加 1。当超过限制时，抛出 `RecursionError`。这是一个安全阀——你的图可以有环，但不能无限循环。

### DAG 与拓扑排序

如果 StateGraph 没有环，它就是一个 DAG（有向无环图）。DAG 有一个重要性质：可以**拓扑排序**——找到一种执行顺序，使得每个节点在其所有前驱节点之后执行。

LangGraph 在编译图时，会进行类似拓扑排序的验证：

- 检查所有节点是否可达（从 START 出发能到达）。
- 检查所有节点是否能到达 END（没有"死胡同"节点）。
- 检查条件边的所有可能路径是否都有效。

```python
# 编译时会验证图的结构正确性
app = graph.compile()  # 如果图有结构问题，这里会报错
```

### 编译过程的验证

`compile()` 方法不仅是验证，还是优化：

1. **结构验证**——确保每个节点至少有一条入边和一条出边（START 和 END 除外），确保没有孤立节点。
2. **边注册**——将用户定义的边转化为内部数据结构，便于执行时快速查找"下一个要执行的节点"。
3. **条件边解析**——对条件边，注册路由函数及其可能的目标节点。
4. **子图展平**——如果包含子图，将其展平为内部节点和边。

编译后的图是一个不可变结构——执行时不会改变拓扑，只会根据条件边选择路径。

## 2.6 总结：LangGraph = BSP + 通道 + 检查点

现在我们可以用一个等式来总结 LangGraph 的理论架构：

```
LangGraph = BSP 执行模型 + 通道状态管理 + 检查点持久化
```

- **BSP 执行模型**：将计算组织为超步序列，每个超步内节点并行执行，超步间同步通信。这保证了确定性和可恢复性。
- **通道状态管理**：每个状态键对应一个通道，通道定义了如何合并多个并行写入（reducer），如何处理数据类型（Annotated[type, reducer]）。这解决了并行节点同时写入同一状态键时的冲突问题。
- **检查点持久化**：每个超步结束后保存完整状态快照，支持中断恢复和时间旅行调试。这使得 LLM 工作流可以跨越长时间运行和多次交互。

这三个理论支柱不是孤立的——它们相互依赖：

- BSP 的同步屏障使得检查点是**一致的**（所有并行写入已经合并）。
- 通道的 reducer 机制使得并行写入是**可合并的**（不需要加锁或串行化）。
- 检查点使得 BSP 执行是**可恢复的**（从任意超步边界重新开始）。

理解了这些理论基础，下一章我们将深入 LangGraph 的核心概念——State、Node、Edge、Channel 和 Checkpoint——看看它们如何在代码中体现。