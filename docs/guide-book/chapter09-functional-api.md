# 第九章：函数式 API

## @entrypoint 装饰器

### 基本用法：将函数转换为 Pregel 图

`@entrypoint` 装饰器将普通 Python 函数转换为 LangGraph 工作流。与 `StateGraph` 不同，函数式 API 以函数为中心，适合线性和简单的工作流：

```python
from langgraph.func import entrypoint, task


@task
def process_item(item: str) -> str:
    return f"已处理: {item}"


@entrypoint()
def workflow(items: list[str]) -> list[str]:
    """最简单的 entrypoint：接收输入，返回输出"""
    results = []
    for item in items:
        result = process_item(item).result()
        results.append(result)
    return results


# 直接调用
result = workflow.invoke(["苹果", "香蕉", "橙子"])
print(result)
# ['已处理: 苹果', '已处理: 香蕉', '已处理: 橙子']
```

### checkpointer 参数：启用持久化

传入 `checkpointer` 后，entrypoint 获得状态持久化能力：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint


@entrypoint(checkpointer=InMemorySaver())
def persistent_workflow(data: str) -> str:
    return f"处理完成: {data}"


config = {"configurable": {"thread_id": "thread-1"}}

# 第一次调用
result = persistent_workflow.invoke("输入A", config)
print(result)  # 处理完成: 输入A

# 第二次调用（同一线程）
result = persistent_workflow.invoke("输入B", config)
print(result)  # 处理完成: 输入B
```

### previous 参数：访问上次调用的返回值

当启用 checkpointer 时，可以通过 `previous` 参数获取同一线程上一次调用的返回值：

```python
from typing import Any, Optional
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint


@entrypoint(checkpointer=InMemorySaver())
def counter(value: int, *, previous: Any = None) -> int:
    """累加计数器：每次调用将新值加到之前的结果上"""
    prev = previous or 0
    result = prev + value
    print(f"上次结果: {prev}, 本次输入: {value}, 新结果: {result}")
    return result


config = {"configurable": {"thread_id": "counter-1"}}

counter.invoke(10, config)  # 上次结果: 0, 本次输入: 10, 新结果: 10
counter.invoke(5, config)   # 上次结果: 10, 本次输入: 5, 新结果: 15
counter.invoke(3, config)   # 上次结果: 15, 本次输入: 3, 新结果: 18
```

### context_schema：运行时上下文

`context_schema` 参数定义运行时上下文的类型，通过 `runtime` 参数访问：

```python
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint
from langgraph.typing import ContextT


class AppContext(TypedDict):
    user_id: str
    permissions: list[str]


@entrypoint(checkpointer=InMemorySaver(), context_schema=AppContext)
def contextual_workflow(data: str, *, previous: Any = None) -> str:
    return f"处理: {data}"
```

### entrypoint.final：分离返回值和保存值

默认情况下，`previous` 参数接收的是函数的返回值。但有时我们希望返回给调用者的值与保存到 checkpoint 的值不同。`entrypoint.final` 可以分离这两个值：

```python
from typing import Any
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint


@entrypoint(checkpointer=InMemorySaver())
def workflow(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    prev = previous or 0
    # value: 返回给调用者
    # save: 保存到 checkpoint，下次调用时通过 previous 获取
    return entrypoint.final(value=prev, save=2 * number)


config = {"configurable": {"thread_id": "final-demo"}}

# 第一次调用：previous=None, 返回 0, 保存 2*3=6
result1 = workflow.invoke(3, config)
print(result1)  # 0

# 第二次调用：previous=6, 返回 6, 保存 2*1=2
result2 = workflow.invoke(1, config)
print(result2)  # 6

# 第三次调用：previous=2, 返回 2, 保存 2*5=10
result3 = workflow.invoke(5, config)
print(result3)  # 2
```

### 与 StateGraph 的对比：何时用哪个

| 特性 | 函数式 API (`@entrypoint`) | StateGraph |
|------|--------------------------|------------|
| 定义方式 | 装饰函数 | 构建图 |
| 适用场景 | 线性/简单工作流 | 复杂状态机 |
| 状态管理 | 函数参数和返回值 | 显式 State TypedDict |
| 节点间流转 | 函数内部逻辑 | 边和条件边 |
| 优势 | 简单直观，Pythonic | 灵活，可视化强 |

- **选择 `@entrypoint`**：工作流步骤固定、逻辑简单，不需要复杂的条件分支
- **选择 `StateGraph`**：工作流需要条件路由、循环、并行执行、子图等复杂结构

## @task 装饰器

### 基本用法：定义可并行执行的子任务

`@task` 定义的工作单元可以被 `entrypoint` 调用，支持并行执行：

```python
import time
from langgraph.func import entrypoint, task


@task
def slow_computation(x: int) -> int:
    """模拟耗时计算"""
    time.sleep(1)
    return x * x


@entrypoint()
def parallel_workflow(numbers: list[int]) -> list[int]:
    """并行执行多个 task"""
    # 创建 futures
    futures = [slow_computation(n) for n in numbers]
    # 获取结果
    results = [f.result() for f in futures]
    return results


# 三个任务并行执行，总耗时约 1 秒而非 3 秒
result = parallel_workflow.invoke([1, 2, 3])
print(result)  # [1, 4, 9]
```

### SyncAsyncFuture：同步 .result() 和异步 await

`@task` 返回的是 `SyncAsyncFuture`，同时支持同步和异步调用：

```python
import asyncio
from langgraph.func import entrypoint, task


@task
def sync_task(x: int) -> int:
    return x + 1


@task
async def async_task(x: int) -> int:
    await asyncio.sleep(0.1)
    return x * 2


# 同步 entrypoint
@entrypoint()
def sync_workflow(data: int) -> int:
    future = sync_task(data)
    return future.result()  # 同步获取结果


# 异步 entrypoint
@entrypoint()
async def async_workflow(data: int) -> int:
    future = async_task(data)
    return await future  # 异步获取结果


# 异步并行
@entrypoint()
async def async_parallel(data: list[int]) -> list[int]:
    futures = [async_task(n) for n in data]
    results = await asyncio.gather(*futures)
    return results
```

### retry_policy：任务重试

`@task` 支持 `retry_policy` 参数，在任务失败时自动重试：

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

# 自定义重试策略
custom_retry = RetryPolicy(
    initial_interval=0.5,   # 首次重试等待 0.5 秒
    backoff_factor=2.0,     # 每次等待时间翻倍
    max_interval=10.0,      # 最大等待 10 秒
    max_attempts=3,         # 最多重试 3 次（含首次）
    jitter=True,            # 添加随机抖动避免惊群
)


@task(retry_policy=custom_retry)
def unreliable_task(data: str) -> str:
    """可能失败的任务，会自动重试"""
    import random
    if random.random() < 0.5:
        raise ValueError("随机失败")
    return f"成功: {data}"


@entrypoint()
def robust_workflow(data: str) -> str:
    return unreliable_task(data).result()
```

### cache_policy：结果缓存

`@task` 支持 `cache_policy` 参数，缓存任务结果避免重复计算：

```python
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy


@task(cache_policy=CachePolicy(ttl=300))  # 缓存 5 分钟
def expensive_computation(query: str) -> str:
    """耗时计算，结果会被缓存"""
    print(f"执行计算: {query}")
    return f"计算结果: {query}"


@entrypoint()
def cached_workflow(query: str) -> str:
    return expensive_computation(query).result()

# 相同输入不会重复计算（在 TTL 内）
```

### name 参数：自定义任务名

默认情况下，任务名使用函数名。可以通过 `name` 参数自定义：

```python
@task(name="summarizer")
def my_summarization_task(text: str) -> str:
    return f"摘要: {text[:50]}"


@entrypoint()
def workflow(text: str) -> str:
    return my_summarization_task(text).result()
```

## @task 的并行执行

### 多个 @task 同时执行

`@task` 的核心优势是并行执行——调用 `@task` 函数后立即返回 future，所有 task 可以同时运行：

```python
import time
from langgraph.func import entrypoint, task


@task
def fetch_data(source: str) -> dict:
    """模拟从不同数据源获取数据"""
    time.sleep(1)  # 模拟网络延迟
    return {"source": source, "data": f"来自{source}的数据"}


@task
def analyze_data(data: dict) -> dict:
    """分析数据"""
    return {"analysis": f"分析结果: {data['source']}"}


@entrypoint()
def pipeline(query: str) -> dict:
    # 同时从多个源获取数据
    futures = [
        fetch_data("数据库"),
        fetch_data("API"),
        fetch_data("缓存"),
    ]

    # 等待所有结果
    results = [f.result() for f in futures]

    # 再进行分析
    analysis = analyze_data(results[0]).result()

    return {"query": query, "data": results, "analysis": analysis}
```

### asyncio.gather 模式

在异步 entrypoint 中，可以使用 `asyncio.gather` 实现真正的异步并行：

```python
import asyncio
from langgraph.func import entrypoint, task


@task
async def async_fetch(url: str) -> str:
    await asyncio.sleep(0.5)
    return f"来自 {url} 的数据"


@entrypoint()
async def async_pipeline(urls: list[str]) -> list[str]:
    # 异步并行获取所有数据
    futures = [async_fetch(url) for url in urls]
    results = await asyncio.gather(*futures)
    return results


# 调用
result = await async_pipeline.ainvoke(["url1", "url2", "url3"])
```

### 与 Send fan-out 的区别

| 特性 | @task 并行 | Send fan-out |
|------|-----------|-------------|
| 定义方式 | 在 entrypoint 中调用 | 通过条件边返回 Send 列表 |
| 状态管理 | 函数参数和返回值 | 独立状态，通过 reducer 合并 |
| 适用场景 | 简单的并行计算 | 需要图结构的中并行分支 |
| 与 StateGraph | 不适用 | 必须在 StateGraph 中使用 |
| 调试复杂度 | 低 | 中 |

- **选择 `@task`**：在 entrypoint 内部并行执行独立计算
- **选择 `Send`**：需要在图结构中定义动态并行分支，结果需要合并回全局状态

## @entrypoint + @task + interrupt 组合

### 在 entrypoint 中使用 interrupt

函数式 API 同样支持人机交互，使用 `interrupt()` 和 `Command`：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
import uuid


@task
def generate_report(data: str) -> str:
    return f"报告内容基于: {data}"


@entrypoint(checkpointer=InMemorySaver())
def review_workflow(data: str) -> dict:
    # 先生成报告
    report = generate_report(data).result()

    # 请求人类审核
    review = interrupt({
        "message": "请审核以下报告",
        "report": report,
    })

    # 审核通过后返回
    return {
        "report": report,
        "review": review,
    }


config = {"configurable": {"thread_id": "review-1"}}

# 第一次调用：生成报告并中断
for chunk in review_workflow.stream("销售数据", config):
    print(chunk)

# 人类审核后恢复
for chunk in review_workflow.stream(Command(resume="通过，无需修改"), config):
    print(chunk)
# {'review_workflow': {'report': '报告内容基于: 销售数据', 'review': '通过，无需修改'}}
```

### 在 task 中使用 interrupt

`interrupt()` 也可以在 `@task` 中使用，效果与在 entrypoint 中相同：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command


@task
def approval_task(amount: float) -> str:
    if amount > 1000:
        result = interrupt(f"金额 {amount} 超过阈值，需要审批")
        return f"审批结果: {result}"
    return "自动通过"


@entrypoint(checkpointer=InMemorySaver())
def approval_workflow(data: dict) -> str:
    result = approval_task(data["amount"]).result()
    return result


config = {"configurable": {"thread_id": "approval-thread"}}

# 大金额需要审批
for chunk in approval_workflow.stream({"amount": 5000}, config):
    print(chunk)

# 提交审批
for chunk in approval_workflow.stream(Command(resume="批准"), config):
    print(chunk)
```

## 完整示例

### 示例 1：简单 entrypoint 工作流

```python
from langgraph.func import entrypoint, task


@task
def greet(name: str) -> str:
    return f"你好, {name}!"


@task
def format_message(greeting: str, extra: str) -> str:
    return f"{greeting} {extra}"


@entrypoint()
def hello_workflow(name: str) -> str:
    greeting = greet(name).result()
    message = format_message(greeting, "欢迎使用 LangGraph").result()
    return message


result = hello_workflow.invoke("开发者")
print(result)  # 你好, 开发者! 欢迎使用 LangGraph
```

### 示例 2：并行任务 + 汇总

```python
import time
from langgraph.func import entrypoint, task


@task
def search_web(query: str) -> str:
    time.sleep(0.5)
    return f"网页搜索结果: {query}"


@task
def search_database(query: str) -> str:
    time.sleep(0.3)
    return f"数据库查询结果: {query}"


@task
def search_cache(query: str) -> str:
    time.sleep(0.1)
    return f"缓存命中: {query}"


@task
def aggregate(results: list[str]) -> str:
    return " | ".join(results)


@entrypoint()
def search_workflow(query: str) -> str:
    # 三个搜索任务并行执行
    web_future = search_web(query)
    db_future = search_database(query)
    cache_future = search_cache(query)

    # 等待所有结果
    web_result = web_future.result()
    db_result = db_future.result()
    cache_result = cache_future.result()

    # 汇总
    final = aggregate([web_result, db_result, cache_result]).result()
    return final


result = search_workflow.invoke("LangGraph 教程")
print(result)
# 网页搜索结果: LangGraph 教程 | 数据库查询结果: LangGraph 教程 | 缓存命中: LangGraph 教程
```

### 示例 3：带人机交互的 entrypoint

```python
from typing import Any, Optional
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command


@task
def analyze_document(text: str) -> dict:
    """分析文档并返回关键指标"""
    return {
        "length": len(text),
        "has_keywords": any(kw in text for kw in ["重要", "紧急", "保密"]),
        "summary": text[:100] + "...",
    }


@entrypoint(checkpointer=InMemorySaver())
def document_review_workflow(text: str, *, previous: Any = None) -> dict:
    """文档审核工作流：分析 -> 人工审核 -> 归档"""

    # 步骤1：自动分析文档
    analysis = analyze_document(text).result()

    # 步骤2：请求人类审核
    review_result = interrupt({
        "message": "请审核此文档",
        "analysis": analysis,
        "document_preview": text[:200],
    })

    # 步骤3：根据审核结果处理
    if review_result.get("approved"):
        return {
            "status": "approved",
            "analysis": analysis,
            "reviewer": review_result.get("reviewer", "unknown"),
            "comment": review_result.get("comment", ""),
        }
    else:
        return {
            "status": "rejected",
            "analysis": analysis,
            "reviewer": review_result.get("reviewer", "unknown"),
            "reason": review_result.get("reason", "未说明原因"),
        }


# --- 使用流程 ---
config = {"configurable": {"thread_id": "doc-review-1"}}

# 提交文档，触发分析和中止
for chunk in document_review_workflow.stream(
    "这是一份重要的项目方案文档，包含紧急需求...",
    config,
):
    print(chunk)

# 审核人提供反馈并恢复
for chunk in document_review_workflow.stream(
    Command(resume={
        "approved": True,
        "reviewer": "张经理",
        "comment": "方案可行，批准执行",
    }),
    config,
):
    print(chunk)
# 输出: {'document_review_workflow': {'status': 'approved', 'analysis': {...}, 'reviewer': '张经理', ...}}
```

### 示例 4：entrypoint.final 分离输出和状态

```python
from typing import Any
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint


@entrypoint(checkpointer=InMemorySaver())
def session_manager(action: str, *, previous: Any = None) -> entrypoint.final[dict, dict]:
    """
    会话管理器：返回用户友好的消息，但保存完整的会话状态。

    value: 返回给用户的简洁消息
    save: 保存到 checkpoint 的完整状态（供下次调用使用）
    """
    prev = previous or {"history": [], "count": 0}

    # 更新状态
    new_history = prev["history"] + [action]
    new_count = prev["count"] + 1

    # 返回给用户：简洁消息
    # 保存到 checkpoint：完整状态
    return entrypoint.final(
        value={
            "message": f"操作 '{action}' 已完成 (第 {new_count} 次)",
            "total_actions": new_count,
        },
        save={
            "history": new_history,
            "count": new_count,
        },
    )


config = {"configurable": {"thread_id": "session-1"}}

# 第一次调用
result1 = session_manager.invoke("登录", config)
print(result1)  # {'message': '操作 登录 已完成 (第 1 次)', 'total_actions': 1}

# 第二次调用
result2 = session_manager.invoke("查询", config)
print(result2)  # {'message': '操作 查询 已完成 (第 2 次)', 'total_actions': 2}

# 第三次调用
result3 = session_manager.invoke("退出", config)
print(result3)  # {'message': '操作 退出 已完成 (第 3 次)', 'total_actions': 3}

# 注意：每次返回给用户的是 value（简洁消息），
# 但 previous 获取的是 save（完整状态），包含历史记录和计数。
```

## 小结

- `@entrypoint` 将普通函数转为工作流，参数 `checkpointer` 启用持久化，`previous` 访问上次返回值
- `entrypoint.final` 分离返回值（给用户）和保存值（给 checkpoint），适合需要不同输出/存储格式的场景
- `@task` 定义可并行执行的子任务，返回 `SyncAsyncFuture`，支持同步 `.result()` 和异步 `await`
- `@task` 支持 `retry_policy`（自动重试）、`cache_policy`（结果缓存）、`name`（自定义名称）
- 多个 `@task` 可以并行执行，异步模式使用 `asyncio.gather` 获得最佳性能
- `interrupt()` 可在 `@entrypoint` 和 `@task` 中使用，配合 `Command(resume=...)` 恢复
- 函数式 API 适合线性工作流，StateGraph 适合复杂状态机——根据需求选择