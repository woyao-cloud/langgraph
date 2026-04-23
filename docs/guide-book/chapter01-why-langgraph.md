# 第一章：LangGraph 解决什么问题

## 1.1 纯函数式 LLM 调用的局限

当你第一次调用一个大语言模型的 API 时，体验是令人震撼的——你给它一段文字，它返回一段似乎理解一切的回答。但当你尝试用它构建真正的应用时，一个根本性的问题立刻浮现：**它是无状态的**。

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "我叫张三"}]
)
# 模型记住了"张三"

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "我叫什么？"}]
)
# 模型完全不知道你是谁——因为上一次的对话没有传递过来
```

这不是一个可以通过简单修补来解决的问题。无状态性带来了三个根本局限：

**无状态**——每次调用都是从零开始。你必须手动把所有上下文塞进 prompt，当上下文超过模型的上下文窗口时，你就不得不做截断或摘要，而这意味着信息丢失。

**无记忆**——模型不记得上次说过什么。在一次多轮对话中，你需要在每次请求中发送完整的对话历史。在跨会话的场景中（比如用户明天再来），你必须自己设计存储和检索机制。

**无分支**——模型的输出是一条直线。如果你需要"根据用户的回答决定走路径 A 还是路径 B"，你必须在应用层编写条件逻辑。当分支变多、嵌套变深时，代码很快变成一团意大利面。

这三个局限叠加在一起，意味着：**纯 LLM 调用只是一个无状态的函数，而不是一个可编排的工作流。** 你可以把它想象成一个只有计算能力但没有存储器的 CPU——它每次执行完就清空，无法构建有意义的程序。

## 1.2 为什么需要"图"？

面对上述局限，最常见的直觉是：那我就用代码把它们串起来。于是你开始写这样的代码：

```python
# 线性管道：A → B → C
result_a = llm_call("分析用户输入")
result_b = llm_call("基于分析生成方案", context=result_a)
result_c = llm_call("评估方案可行性", context=result_b)
```

这就是**线性管道**（Pipeline）模式。它对于简单的、步骤固定的任务够用了。但当你的应用变得复杂时，管道模式开始力不从心：

- 你需要**条件分支**：根据分析结果，有时候走路径 B，有时候走路径 C。
- 你需要**并行执行**：同时让三个 Agent 各自研究一个子问题，然后汇总结果。
- 你需要**循环**：验证结果不通过时，回到上一步重新生成。
- 你需要**人机协作**：在关键节点暂停，等待人类审批后再继续。
- 你需要**持久化**：如果程序在第三步崩溃了，能从第二步的检查点恢复，而不是从头来过。

这些需求指向一个共同的抽象：**有向图**。

在图中，节点是处理步骤（LLM 调用、工具执行、人工审核等），边是步骤之间的转移关系。图可以表达分支、循环、并行——这些是线性管道无法优雅处理的模式。而"有状态"意味着图中的每个节点都可以读写一份共享状态，不需要你手动在各步骤间传递参数。

**从线性管道到有状态工作流，是从"指令式编程"到"声明式编排"的跨越。** 你不再需要用 if-else 和 for 循环来控制流程，而是通过定义图的拓扑结构来表达意图。

## 1.3 LangGraph 的核心价值主张

LangGraph 提供了五个核心能力，每一个都直指纯 LLM 调用的一个局限：

| 能力 | 解决的问题 | 没有它时你会怎样 |
|------|-----------|-----------------|
| **有状态** | 节点间共享数据 | 手动传递参数，容易出错 |
| **可分支** | 条件执行路径 | 深层嵌套的 if-else |
| **可持久化** | 状态自动保存 | 崩溃后从头再来 |
| **可中断** | 暂停等待外部输入 | 无法实现人机协作 |
| **可恢复** | 从检查点继续执行 | 长流程不敢中途停止 |

这五个能力组合在一起，构成了一个完整的**有状态工作流引擎**：

- **有状态**：StateGraph 维护一份共享状态，每个节点读取并更新它。状态通过类型化的通道（Channel）管理，支持合并多个并行写入。
- **可分支**：条件边（conditional edge）让图的拓扑根据运行时数据动态变化，而不是在编译时固定。
- **可持久化**：检查点（Checkpoint）在每个超步结束时自动保存完整状态快照，无需你手动管理。
- **可中断**：中断点（interrupt）让你在任意节点前暂停执行，等待人工输入或外部确认后再恢复。
- **可恢复**：通过 `thread_id` 标识一个会话，从任意检查点恢复执行，实现真正的长程任务管理。

用一个比喻来说：如果纯 LLM 调用是一次性的计算器，LangGraph 是一台有存储、有分支、有断点恢复能力的完整计算机。

## 1.4 与 LangChain 的关系和区别

LangChain 和 LangGraph 经常被放在一起讨论，但它们解决的是不同层次的问题：

**LangChain 是工具链**——它提供了"把 LLM 和各种工具连接起来"的抽象：Prompt 模板、输出解析器、检索器、工具调用格式化等。它的核心隐喻是 **Chain（链）**——一个线性的、无状态的组合序列。你用 LangChain 来回答"怎么把 LLM 和数据库、搜索引擎、代码执行器组合在一起"这个问题。

**LangGraph 是工作流引擎**——它提供了"编排有状态的多步骤流程"的抽象：状态定义、节点函数、边连接、检查点持久化。它的核心隐喻是 **Graph（图）**——一个可以有分支、有循环、有状态持久化的有向图。你用 LangGraph 来回答"怎么让多个步骤按照复杂逻辑协同运行"这个问题。

两者的关系可以这样理解：

- LangChain 提供了**积木**（模型封装、工具接口、检索器等），而 LangGraph 提供了**搭建图纸**（状态管理、流程编排、检查点）。
- 它们可以独立使用，也可以组合使用——LangGraph 的节点中完全可以调用 LangChain 的 Chain 或 Agent。
- LangChain 更适合简单的、线性的、无状态的任务；LangGraph 更适合复杂的、有分支的、有状态的流程。

```
LangChain:  Prompt → LLM → Parser → Output（线性链）
LangGraph:  State ─→ [Node A] ─→ [Node B] ─→ [Node C]
                         ↑              │       ↙      ↓
                         └──────── [Condition] ─→ [Node D]
                         （有状态图，可分支、可循环、可持久化）
```

## 1.5 与其他框架的对比

### LangGraph vs CrewAI

CrewAI 的核心抽象是 **Agent + Role + Task**。你定义一组有角色的 Agent，给他们分配 Task，CrewAI 负责协调它们完成目标。它的优势是上手快——定义角色、分配任务、运行，非常直觉。

但 CrewAI 的局限在于：流程编排能力较弱。你无法精确控制 Agent 之间的交互拓扑，无法在运行时动态改变流程，也无法实现细粒度的中断和恢复。CrewAI 更适合"几个人一起完成一个任务"的场景，而不适合"一个复杂的多步骤工作流"。

### LangGraph vs AutoGen

AutoGen（微软）的核心抽象是 **多 Agent 对话**。多个 Agent 通过对话来协同完成任务。它的优势在于自然地模拟了人类团队协作的过程。

AutoGen 的局限与 CrewAI 类似：对话是主要的编排方式，难以表达精确的流程逻辑。当你需要"如果 X 则走路径 A，否则走路径 B"这样的条件分支时，AutoGen 的对话式编排就显得力不从心。

### LangGraph vs Temporal

Temporal 是一个通用的分布式工作流引擎，支持长时间运行的任务、检查点恢复、人机协作等。从能力上说，Temporal 和 LangGraph 有很大的重叠——都支持状态持久化、可恢复执行、条件分支。

关键区别在于：Temporal 是**通用的工作流引擎**，不专门服务于 LLM 应用。它需要你用 SDK 编写工作流代码，部署 Temporal Server，管理 Worker 进程。对于 LLM 应用来说，Temporal 太重了——你不需要一个完整的分布式系统来编排一个五步的 Agent 流程。LangGraph 更轻量、更贴近 LLM 开发者的心智模型。

### LangGraph vs Airflow

Airflow 是数据管道编排工具，核心抽象是 **DAG（有向无环图）**。它擅长定时调度数据处理任务，但不适合 LLM 工作流——因为 LLM 工作流需要**循环**（迭代优化、自我修正），而 Airflow 的 DAG 天然不允许环。此外，Airflow 没有内建的 LLM 集成和状态管理。

| 框架 | 核心抽象 | 适合场景 | 不适合场景 |
|------|---------|---------|-----------|
| LangGraph | 有状态有向图 | 复杂 LLM 工作流 | 简单线性管道 |
| CrewAI | Agent + Role | 角色协作任务 | 精确流程控制 |
| AutoGen | 多 Agent 对话 | 讨论式协作 | 条件分支密集的流程 |
| Temporal | 分布式工作流 | 通用长程任务 | 快速 LLM 原型 |
| Airflow | DAG | 数据管道调度 | 需要循环的 LLM 流程 |

## 1.6 LangGraph 的适用边界

### 该用 LangGraph 的场景

- **多步骤 LLM 工作流**：需要多个 LLM 调用按特定顺序或条件组合执行。
- **需要人机协作**：在流程的关键节点需要人工审核、确认或输入。
- **需要状态持久化**：流程可能很长（跨小时甚至跨天），需要断点恢复。
- **需要动态分支**：根据中间结果决定下一步走哪条路径。
- **需要并行执行**：同时运行多个处理分支，然后汇总结果。

### 不该用 LangGraph 的场景

- **简单的单次 LLM 调用**：如果只需要一次 prompt-response，直接调用 API 就够了。
- **纯线性无状态管道**：如果步骤固定、无分支、不需要持久化，用 LangChain 的 Chain 或普通 Python 函数更简单。
- **非 LLM 工作流**：如果你的任务不涉及 LLM（比如纯数据处理管道），Airflow 或 Temporal 可能更合适。
- **极度低延迟场景**：LangGraph 的检查点和状态管理有额外开销，如果延迟要求在毫秒级，需要评估是否值得。

**经验法则**：如果你的流程可以用一个简单的函数链表达，不需要状态、分支、或持久化，那就不要用 LangGraph。只有当你需要图结构带来的能力时，才值得引入这个抽象。

## 1.7 一个最小可运行示例

让我们用一段最简单的代码来感受 LangGraph 的核心概念。这个示例创建一个带状态的简单图，它会根据用户输入决定走哪条路径：

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. 定义状态——图的共享数据结构
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表，add_messages 是一个 reducer
    next_step: str  # 决定下一步走哪条路径

# 2. 定义节点——处理状态的函数
def chatbot(state: State) -> dict:
    """主对话节点：根据最后一条消息决定下一步"""
    last_message = state["messages"][-1]
    # 简单逻辑：如果用户说"计算"，走计算路径；否则继续对话
    if "计算" in last_message.content:
        return {"next_step": "calculator"}
    return {"next_step": "chat"}

def calculator(state: State) -> dict:
    """计算节点：执行数学运算"""
    return {
        "messages": [{"role": "assistant", "content": "计算结果是 42"}],
        "next_step": "end"
    }

def responder(state: State) -> dict:
    """普通回复节点"""
    return {
        "messages": [{"role": "assistant", "content": "我是一个有状态的助手"}],
        "next_step": "end"
    }

# 3. 定义路由函数——决定条件边怎么走
def route_by_step(state: State) -> str:
    if state["next_step"] == "calculator":
        return "calculator"
    return "responder"

# 4. 构建图
graph = StateGraph(State)

# 添加节点
graph.add_node("chatbot", chatbot)
graph.add_node("calculator", calculator)
graph.add_node("responder", responder)

# 添加边
graph.add_edge(START, "chatbot")  # 从起点到 chatbot
graph.add_conditional_edges("chatbot", route_by_step)  # chatbot 根据条件走不同路径
graph.add_edge("calculator", END)  # calculator 直接到终点
graph.add_edge("responder", END)  # responder 直接到终点

# 5. 编译并运行
app = graph.compile()

# 运行图
result = app.invoke({
    "messages": [{"role": "user", "content": "帮我计算一下"}],
    "next_step": ""
})

print(result["messages"][-1].content)  # 输出：计算结果是 42

# 再次运行——注意图是有状态的
result2 = app.invoke({
    "messages": [{"role": "user", "content": "你好"}],
    "next_step": ""
})

print(result2["messages"][-1].content)  # 输出：我是一个有状态的助手
```

这个示例虽然简单，但已经展示了 LangGraph 的三个核心概念：

1. **状态（State）**——`State` 是图的共享数据结构，所有节点都可以读写它。
2. **节点（Node）**——`chatbot`、`calculator`、`responder` 是处理状态的函数，每个返回状态的部分更新。
3. **边（Edge）**——包括从 `START` 到 `chatbot` 的固定边，以及从 `chatbot` 到 `calculator` 或 `responder` 的条件边。

当你运行这个图时，LangGraph 会自动管理状态的传递、条件分支的执行，以及流程的完整性验证。这就是"图"带来的价值——你只需要声明**做什么**和**怎么连**，执行引擎帮你处理剩下的。

在下一章中，我们将深入 LangGraph 背后的理论基础：BSP 模型、Actor 模型、有限状态机和检查点理论——理解这些，你就能真正理解 LangGraph 为什么这样设计。