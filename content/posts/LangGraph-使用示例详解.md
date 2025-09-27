---
title: "LangGraph 源码剖析 - 使用示例详解"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['源码分析', '技术文档']
categories: ['技术分析']
description: "LangGraph 源码剖析 - 使用示例详解的深入技术分析文档"
keywords: ['源码分析', '技术文档']
author: "技术分析师"
weight: 1
---

## 1. 快速入门示例

### 1.1 最简单的聊天机器人

```python
"""
最基础的LangGraph聊天机器人示例
展示核心概念：状态图、消息处理、模型调用
"""

from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, add_messages
from langgraph.checkpoint.memory import InMemorySaver

# 1. 定义状态类型
class ChatState(TypedDict):
    """
    聊天状态定义
    - messages: 消息历史列表，使用add_messages reducer自动合并
    """
    messages: Annotated[list[AnyMessage], add_messages]

# 2. 创建模型实例
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 3. 定义节点函数
def chatbot_node(state: ChatState) -> dict[str, list[AnyMessage]]:
    """
    聊天机器人节点：调用LLM生成回复
    
    参数：
        state: 当前状态，包含消息历史
    
    返回：
        包含新AI消息的状态更新
    """
    # 获取消息历史
    messages = state["messages"]
    
    # 调用模型生成回复
    response = model.invoke(messages)
    
    # 返回状态更新
    return {"messages": [response]}

# 4. 构建状态图
def create_simple_chatbot():
    """创建简单聊天机器人图"""
    
    # 创建状态图
    graph = StateGraph(ChatState)
    
    # 添加节点
    graph.add_node("chatbot", chatbot_node)
    
    # 设置入口和出口
    graph.set_entry_point("chatbot")
    graph.set_finish_point("chatbot")
    
    # 编译图（包含内存检查点）
    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)

# 5. 使用示例
def main():
    """主函数：演示聊天机器人使用"""
    
    # 创建聊天机器人
    chatbot = create_simple_chatbot()
    
    # 配置会话
    config = {"configurable": {"thread_id": "conversation-1"}}
    
    # 进行对话
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', '退出']:
            break
        
        # 调用聊天机器人
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config
        )
        
        # 显示回复
        ai_message = result["messages"][-1]
        print(f"Bot: {ai_message.content}")

if __name__ == "__main__":
    main()
```

### 1.2 带工具的智能体

```python
"""
使用工具的ReAct智能体示例
展示：工具调用、条件路由、循环执行
"""

from typing import TypedDict, Annotated, Literal
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
import requests
import json

# 1. 定义工具函数
@tool
def search_web(query: str) -> str:
    """
    搜索网络内容的工具
    
    参数：
        query: 搜索查询字符串
    
    返回：
        搜索结果摘要
    """
    # 这里使用一个模拟的搜索API
    try:
        # 实际使用时替换为真实的搜索API
        url = f"https://api.search.example.com/search?q={query}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return f"搜索结果：{data.get('summary', '未找到相关信息')}"
        else:
            return f"搜索失败，状态码：{response.status_code}"
    except Exception as e:
        return f"搜索出错：{str(e)}"

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    
    参数：
        city: 城市名称
    
    返回：
        天气信息字符串
    """
    # 模拟天气API调用
    weather_data = {
        "北京": "晴天，温度25°C，湿度60%",
        "上海": "多云，温度22°C，湿度70%",
        "广州": "雨天，温度28°C，湿度85%",
        "深圳": "晴天，温度30°C，湿度55%",
    }
    
    return weather_data.get(city, f"未找到{city}的天气信息，请尝试其他城市")

@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式
    
    参数：
        expression: 数学表达式字符串
    
    返回：
        计算结果
    """
    try:
        # 安全的表达式计算（仅支持基本运算）
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不安全字符"
        
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

# 2. 状态定义
class AgentState(TypedDict):
    """智能体状态"""
    messages: Annotated[list[AnyMessage], add_messages]
    
    # 可选：添加额外状态
    conversation_count: int  # 对话轮数
    last_tool_used: str     # 最后使用的工具

# 3. 智能体节点
def agent_node(state: AgentState) -> dict:
    """
    智能体节点：分析用户输入并决定是否使用工具
    """
    messages = state["messages"]
    
    # 创建绑定工具的模型
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    model_with_tools = model.bind_tools([search_web, get_weather, calculate])
    
    # 调用模型
    response = model_with_tools.invoke(messages)
    
    # 更新对话计数
    current_count = state.get("conversation_count", 0)
    
    return {
        "messages": [response],
        "conversation_count": current_count + 1,
    }

# 4. 构建工具调用图
def create_tool_agent():
    """创建带工具的智能体图"""
    
    # 创建工具节点
    tools = [search_web, get_weather, calculate]
    tool_node = ToolNode(tools)
    
    # 创建状态图
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    
    # 添加边
    graph.add_edge(START, "agent")
    
    # 添加条件边：根据是否有工具调用决定路由
    graph.add_conditional_edges(
        "agent",
        tools_condition,  # 预构建的条件函数
        {
            "tools": "tools",   # 有工具调用时去执行工具
            END: END,           # 无工具调用时结束
        }
    )
    
    # 工具执行后回到智能体
    graph.add_edge("tools", "agent")
    
    # 编译图
    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)

# 5. 使用示例
def demo_tool_agent():
    """演示工具智能体的使用"""
    
    agent = create_tool_agent()
    config = {"configurable": {"thread_id": "tool-agent-1"}}
    
    # 测试用例
    test_queries = [
        "今天北京的天气如何？",
        "搜索一下LangGraph的最新信息",
        "计算 (25 + 15) * 2 的结果",
        "你好，请介绍一下你自己",
    ]
    
    print("=== 工具智能体演示 ===\n")
    
    for query in test_queries:
        print(f"用户：{query}")
        
        # 调用智能体
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config
        )
        
        # 显示最后的AI回复
        final_message = result["messages"][-1]
        print(f"智能体：{final_message.content}")
        print(f"对话轮数：{result.get('conversation_count', 0)}")
        print("-" * 50)

if __name__ == "__main__":
    demo_tool_agent()
```

## 2. 高级功能示例

### 2.1 人机交互中断

```python
"""
人机交互中断示例
展示：中断机制、人工审核、工作流控制
"""

from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

# 1. 状态定义
class ReviewState(TypedDict):
    """审核工作流状态"""
    messages: Annotated[list[AnyMessage], add_messages]
    
    # 审核相关状态
    content_to_review: str      # 待审核内容
    review_required: bool       # 是否需要审核
    human_feedback: str         # 人工反馈
    approval_status: str        # 审批状态：pending/approved/rejected

# 2. 内容生成节点
def content_generator(state: ReviewState) -> dict:
    """
    内容生成器：生成需要审核的内容
    """
    messages = state["messages"]
    
    # 使用模型生成内容
    model = ChatOpenAI(model="gpt-3.5-turbo")
    
    # 添加系统提示，指示生成的内容类型
    system_prompt = (
        "你是一个内容生成助手。根据用户请求生成内容，"
        "如果内容涉及敏感话题或可能需要人工审核的内容，"
        "请在回复中标注 [NEEDS_REVIEW]"
    )
    
    full_messages = [
        {"role": "system", "content": system_prompt},
        *messages
    ]
    
    response = model.invoke(full_messages)
    generated_content = response.content
    
    # 检查是否需要审核
    needs_review = "[NEEDS_REVIEW]" in generated_content
    
    return {
        "messages": [response],
        "content_to_review": generated_content,
        "review_required": needs_review,
        "approval_status": "pending" if needs_review else "approved"
    }

# 3. 人工审核节点（中断点）
def human_review_node(state: ReviewState) -> dict:
    """
    人工审核节点：这里会中断执行，等待人工干预
    """
    content = state["content_to_review"]
    
    # 这里使用interrupt来暂停执行
    # 实际应用中，这里会发送审核请求给人工审核员
    interrupt(
        f"内容需要人工审核：\n{content}\n\n"
        f"请审核上述内容并提供反馈。"
    )
    
    # 这部分代码在人工干预后会继续执行
    return {"approval_status": "under_review"}

# 4. 决策节点
def approval_decision(state: ReviewState) -> Literal["approved", "rejected", "needs_revision"]:
    """
    审批决策：基于人工反馈决定下一步
    """
    feedback = state.get("human_feedback", "").lower()
    
    if "approve" in feedback or "通过" in feedback:
        return "approved"
    elif "reject" in feedback or "拒绝" in feedback:
        return "rejected"
    else:
        return "needs_revision"

# 5. 内容修订节点
def content_revision(state: ReviewState) -> dict:
    """
    内容修订：基于反馈修改内容
    """
    original_content = state["content_to_review"]
    feedback = state["human_feedback"]
    
    model = ChatOpenAI(model="gpt-3.5-turbo")
    
    revision_prompt = (
        f"原始内容：{original_content}\n\n"
        f"反馈意见：{feedback}\n\n"
        f"请根据反馈意见修改内容，确保符合要求。"
    )
    
    revised_content = model.invoke([{"role": "user", "content": revision_prompt}])
    
    return {
        "messages": [revised_content],
        "content_to_review": revised_content.content,
        "approval_status": "revised"
    }

# 6. 最终确认节点
def final_confirmation(state: ReviewState) -> dict:
    """
    最终确认：输出最终结果
    """
    status = state["approval_status"]
    content = state["content_to_review"]
    
    if status == "approved":
        final_message = AIMessage(content=f"内容已通过审核：\n{content}")
    elif status == "rejected":
        final_message = AIMessage(content="内容未通过审核，已被拒绝。")
    else:
        final_message = AIMessage(content=f"修订后的内容：\n{content}")
    
    return {"messages": [final_message]}

# 7. 构建审核工作流图
def create_review_workflow():
    """创建人工审核工作流图"""
    
    graph = StateGraph(ReviewState)
    
    # 添加节点
    graph.add_node("generate", content_generator)
    graph.add_node("human_review", human_review_node) 
    graph.add_node("revise", content_revision)
    graph.add_node("confirm", final_confirmation)
    
    # 添加边
    graph.add_edge(START, "generate")
    
    # 条件边：是否需要审核
    graph.add_conditional_edges(
        "generate",
        lambda state: "review" if state["review_required"] else "confirm",
        {
            "review": "human_review",
            "confirm": "confirm"
        }
    )
    
    # 审核后的条件边
    graph.add_conditional_edges(
        "human_review",
        approval_decision,
        {
            "approved": "confirm",
            "rejected": "confirm", 
            "needs_revision": "revise"
        }
    )
    
    # 修订后回到确认
    graph.add_edge("revise", "confirm")
    graph.add_edge("confirm", END)
    
    # 编译图，设置中断点
    memory = InMemorySaver()
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["human_review"]  # 在人工审核前中断
    )

# 8. 使用示例
def demo_human_in_loop():
    """演示人机交互工作流"""
    
    workflow = create_review_workflow()
    config = {"configurable": {"thread_id": "review-workflow-1"}}
    
    # 第一步：提交需要审核的内容请求
    print("=== 步骤1: 提交内容请求 ===")
    user_request = "请帮我写一篇关于人工智能伦理的文章，需要涵盖一些争议性观点"
    
    result = workflow.invoke(
        {"messages": [HumanMessage(content=user_request)]},
        config
    )
    
    print("生成的内容（等待审核）：")
    print(result["content_to_review"])
    print(f"审核状态：{result['approval_status']}")
    
    # 第二步：模拟人工审核反馈
    print("\n=== 步骤2: 人工审核 ===")
    # 这里模拟人工审核员提供反馈
    human_feedback = "内容总体不错，但需要增加更多平衡性的观点，减少争议性表述"
    
    # 恢复执行，提供人工反馈
    result = workflow.invoke(
        {
            "human_feedback": human_feedback,
            "approval_status": "needs_revision"
        },
        config
    )
    
    print("最终结果：")
    final_message = result["messages"][-1]
    print(final_message.content)

if __name__ == "__main__":
    demo_human_in_loop()
```

### 2.2 多智能体协作

```python
"""
多智能体协作示例
展示：智能体角色分工、消息传递、协作完成复杂任务
"""

from typing import TypedDict, Annotated, Literal
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import InMemorySaver

# 1. 多智能体状态定义
class MultiAgentState(TypedDict):
    """多智能体协作状态"""
    messages: Annotated[list[AnyMessage], add_messages]
    
    # 任务相关状态
    task_description: str           # 任务描述
    current_stage: str             # 当前阶段
    
    # 各智能体的输出
    researcher_output: str         # 研究员输出
    writer_output: str            # 写作者输出
    reviewer_output: str          # 审核员输出
    
    # 协作状态
    collaboration_history: list[dict]  # 协作历史
    final_result: str             # 最终结果

# 2. 智能体角色定义
class AgentRoles:
    """定义各个智能体的角色和提示词"""
    
    RESEARCHER = """
    你是一个专业的研究员。你的职责是：
    1. 分析给定的任务和主题
    2. 收集相关信息和数据
    3. 提供详细的研究报告
    4. 为后续写作提供可靠的信息基础
    
    请始终保持客观、准确、详细。
    """
    
    WRITER = """
    你是一个专业的写作者。你的职责是：
    1. 基于研究员提供的信息
    2. 创作高质量的内容
    3. 确保内容结构清晰、逻辑性强
    4. 适应不同的写作风格和要求
    
    请注重内容的可读性和吸引力。
    """
    
    REVIEWER = """
    你是一个严格的审核员。你的职责是：
    1. 审查写作者的内容
    2. 检查事实准确性
    3. 评估内容质量和完整性
    4. 提供改进建议
    
    请保持公正、严格的审核标准。
    """

# 3. 各智能体节点实现
def researcher_agent(state: MultiAgentState) -> dict:
    """
    研究员智能体：负责信息收集和分析
    """
    task = state["task_description"]
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    messages = [
        SystemMessage(content=AgentRoles.RESEARCHER),
        HumanMessage(content=f"任务：{task}\n请进行深入研究并提供详细报告。")
    ]
    
    response = model.invoke(messages)
    
    # 记录协作历史
    collaboration_entry = {
        "agent": "researcher",
        "action": "research",
        "output": response.content,
        "timestamp": "now"  # 在实际应用中使用真实时间戳
    }
    
    current_history = state.get("collaboration_history", [])
    
    return {
        "messages": [AIMessage(content=f"[研究员] {response.content}")],
        "researcher_output": response.content,
        "current_stage": "research_completed",
        "collaboration_history": current_history + [collaboration_entry]
    }

def writer_agent(state: MultiAgentState) -> dict:
    """
    写作者智能体：基于研究结果进行创作
    """
    task = state["task_description"]
    research_result = state["researcher_output"]
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    messages = [
        SystemMessage(content=AgentRoles.WRITER),
        HumanMessage(content=f"""
任务：{task}

研究员提供的资料：
{research_result}

请基于以上资料创作高质量内容。
""")
    ]
    
    response = model.invoke(messages)
    
    # 记录协作历史
    collaboration_entry = {
        "agent": "writer",
        "action": "write",
        "output": response.content,
        "based_on": "researcher_output"
    }
    
    current_history = state.get("collaboration_history", [])
    
    return {
        "messages": [AIMessage(content=f"[写作者] {response.content}")],
        "writer_output": response.content,
        "current_stage": "writing_completed",
        "collaboration_history": current_history + [collaboration_entry]
    }

def reviewer_agent(state: MultiAgentState) -> dict:
    """
    审核员智能体：审核写作质量并提供反馈
    """
    task = state["task_description"]
    research_result = state["researcher_output"]
    written_content = state["writer_output"]
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    messages = [
        SystemMessage(content=AgentRoles.REVIEWER),
        HumanMessage(content=f"""
任务：{task}

研究资料：
{research_result}

写作内容：
{written_content}

请审核以上内容的质量、准确性和完整性，并提供评价和改进建议。
""")
    ]
    
    response = model.invoke(messages)
    
    # 记录协作历史
    collaboration_entry = {
        "agent": "reviewer", 
        "action": "review",
        "output": response.content,
        "reviewed": ["researcher_output", "writer_output"]
    }
    
    current_history = state.get("collaboration_history", [])
    
    return {
        "messages": [AIMessage(content=f"[审核员] {response.content}")],
        "reviewer_output": response.content,
        "current_stage": "review_completed",
        "collaboration_history": current_history + [collaboration_entry]
    }

def coordinator_agent(state: MultiAgentState) -> dict:
    """
    协调员智能体：整合各智能体的输出，产生最终结果
    """
    task = state["task_description"]
    research = state["researcher_output"]
    writing = state["writer_output"] 
    review = state["reviewer_output"]
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    messages = [
        SystemMessage(content="""
你是一个项目协调员。你的职责是：
1. 整合各个智能体的工作成果
2. 基于审核意见优化最终输出
3. 确保最终结果满足任务要求
4. 提供项目总结
"""),
        HumanMessage(content=f"""
任务：{task}

研究员报告：
{research}

写作者内容：
{writing}

审核员反馈：
{review}

请整合以上信息，产生最终的高质量结果。
""")
    ]
    
    response = model.invoke(messages)
    
    # 记录最终协作结果
    collaboration_entry = {
        "agent": "coordinator",
        "action": "coordinate",
        "output": response.content,
        "integrated": ["research", "writing", "review"]
    }
    
    current_history = state.get("collaboration_history", [])
    
    return {
        "messages": [AIMessage(content=f"[协调员] 最终结果：\n{response.content}")],
        "final_result": response.content,
        "current_stage": "completed",
        "collaboration_history": current_history + [collaboration_entry]
    }

# 4. 协作流程控制
def should_continue_collaboration(state: MultiAgentState) -> Literal["continue", "end"]:
    """
    决定是否继续协作流程
    """
    stage = state["current_stage"]
    
    if stage == "completed":
        return "end"
    else:
        return "continue"

def next_agent(state: MultiAgentState) -> Literal["writer", "reviewer", "coordinator"]:
    """
    决定下一个执行的智能体
    """
    stage = state["current_stage"]
    
    if stage == "research_completed":
        return "writer"
    elif stage == "writing_completed":
        return "reviewer"
    elif stage == "review_completed":
        return "coordinator"
    else:
        return "coordinator"  # 默认

# 5. 构建多智能体协作图
def create_multi_agent_system():
    """创建多智能体协作系统"""
    
    graph = StateGraph(MultiAgentState)
    
    # 添加智能体节点
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("reviewer", reviewer_agent)
    graph.add_node("coordinator", coordinator_agent)
    
    # 设置协作流程
    graph.add_edge(START, "researcher")
    
    # 使用条件边控制智能体间的协作流程
    graph.add_conditional_edges(
        "researcher",
        next_agent,
        {
            "writer": "writer",
            "reviewer": "reviewer",
            "coordinator": "coordinator"
        }
    )
    
    graph.add_conditional_edges(
        "writer",
        next_agent,
        {
            "reviewer": "reviewer",
            "coordinator": "coordinator"
        }
    )
    
    graph.add_conditional_edges(
        "reviewer", 
        next_agent,
        {
            "coordinator": "coordinator"
        }
    )
    
    graph.add_conditional_edges(
        "coordinator",
        should_continue_collaboration,
        {
            "continue": "researcher",  # 如果需要，可以重新开始流程
            "end": END
        }
    )
    
    # 编译系统
    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)

# 6. 使用演示
def demo_multi_agent_collaboration():
    """演示多智能体协作"""
    
    system = create_multi_agent_system()
    config = {"configurable": {"thread_id": "multi-agent-1"}}
    
    # 定义任务
    task = "写一篇关于人工智能在教育领域应用的深度分析文章，包括现状、挑战和未来趋势"
    
    print("=== 多智能体协作演示 ===")
    print(f"任务：{task}\n")
    
    # 启动协作流程
    result = system.invoke(
        {
            "task_description": task,
            "current_stage": "starting",
            "collaboration_history": []
        },
        config
    )
    
    # 输出协作过程
    print("=== 协作历史 ===")
    for entry in result["collaboration_history"]:
        print(f"智能体：{entry['agent']}")
        print(f"操作：{entry['action']}")
        print(f"输出摘要：{entry['output'][:100]}...")
        print("-" * 50)
    
    print("\n=== 最终结果 ===")
    print(result["final_result"])

if __name__ == "__main__":
    demo_multi_agent_collaboration()
```

## 3. 实际应用场景示例

### 3.1 客服智能体系统

```python
"""
客服智能体系统示例
展示：意图识别、多轮对话、知识库查询、工单创建
"""

from typing import TypedDict, Annotated, Literal, Optional
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
import uuid
from datetime import datetime

# 1. 客服系统状态定义
class CustomerServiceState(TypedDict):
    """客服系统状态"""
    messages: Annotated[list[AnyMessage], add_messages]
    
    # 用户信息
    user_id: Optional[str]
    user_name: Optional[str]
    user_tier: str              # VIP, Premium, Regular
    
    # 对话上下文
    intent: Optional[str]       # 用户意图
    problem_category: Optional[str]  # 问题分类
    urgency_level: str         # 紧急程度
    
    # 处理状态
    resolved: bool             # 是否已解决
    ticket_id: Optional[str]   # 工单ID
    satisfaction_score: Optional[int]  # 满意度评分

# 2. 知识库工具
class KnowledgeBase:
    """模拟知识库"""
    
    def __init__(self):
        # 模拟知识库文档
        knowledge_docs = [
            Document(
                page_content="账户密码重置：用户可以通过邮箱或手机验证码重置密码。步骤：1.点击忘记密码 2.输入邮箱 3.查收验证码 4.设置新密码",
                metadata={"category": "account", "type": "password_reset"}
            ),
            Document(
                page_content="退款政策：商品在7天内可以无理由退款，需要保持商品完好。特殊商品（如定制商品）不支持退款。",
                metadata={"category": "refund", "type": "policy"}
            ),
            Document(
                page_content="订单状态查询：用户可以在'我的订单'页面查看订单状态。状态包括：已下单、已支付、配送中、已完成。",
                metadata={"category": "order", "type": "status"}
            ),
            Document(
                page_content="VIP会员权益：享受免费包邮、专属客服、优先退换货、会员折扣等特权。",
                metadata={"category": "membership", "type": "vip"}
            ),
            Document(
                page_content="技术故障报告：如遇到APP崩溃、网页加载异常等技术问题，请提供设备信息和错误截图。",
                metadata={"category": "technical", "type": "bug_report"}
            )
        ]
        
        # 创建向量存储
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(knowledge_docs, embeddings)
    
    def search(self, query: str, k: int = 3) -> list[str]:
        """搜索相关知识"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

# 初始化知识库
kb = KnowledgeBase()

# 3. 客服工具定义
@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索知识库获取相关信息
    
    参数：
        query: 搜索查询
    
    返回：
        相关知识库内容
    """
    try:
        results = kb.search(query, k=2)
        if results:
            return f"找到相关信息：\n" + "\n\n".join(results)
        else:
            return "抱歉，未找到相关信息。"
    except Exception as e:
        return f"知识库搜索出错：{str(e)}"

@tool 
def create_support_ticket(
    user_id: str,
    problem_description: str,
    category: str,
    urgency: str = "normal"
) -> str:
    """
    创建客服工单
    
    参数：
        user_id: 用户ID
        problem_description: 问题描述
        category: 问题分类
        urgency: 紧急程度 (low/normal/high/urgent)
    
    返回：
        工单ID
    """
    ticket_id = f"TICKET-{uuid.uuid4().hex[:8].upper()}"
    
    # 模拟工单创建
    ticket_data = {
        "id": ticket_id,
        "user_id": user_id,
        "description": problem_description,
        "category": category,
        "urgency": urgency,
        "status": "open",
        "created_at": datetime.now().isoformat()
    }
    
    return f"工单已创建，工单号：{ticket_id}。我们会在24小时内联系您处理此问题。"

@tool
def escalate_to_human(reason: str, user_tier: str) -> str:
    """
    升级至人工客服
    
    参数：
        reason: 升级原因
        user_tier: 用户等级
    
    返回：
        升级结果
    """
    if user_tier == "VIP":
        wait_time = "2-3分钟"
    elif user_tier == "Premium":
        wait_time = "5-10分钟"
    else:
        wait_time = "15-20分钟"
    
    return f"正在为您转接人工客服，预计等待时间：{wait_time}。转接原因：{reason}"

# 4. 意图识别节点
def intent_classifier(state: CustomerServiceState) -> dict:
    """
    意图识别：分析用户消息，识别意图和问题分类
    """
    messages = state["messages"]
    if not messages:
        return {}
    
    last_message = messages[-1]
    user_message = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    classification_prompt = f"""
分析以下用户消息，识别意图和问题分类：

用户消息："{user_message}"

请识别：
1. 意图类型：consultation(咨询), complaint(投诉), technical_support(技术支持), account_issue(账户问题), order_inquiry(订单查询)
2. 问题分类：account, order, refund, technical, membership, other
3. 紧急程度：low, normal, high, urgent

以JSON格式回复：
{{"intent": "意图类型", "category": "问题分类", "urgency": "紧急程度"}}
"""
    
    response = model.invoke([{"role": "user", "content": classification_prompt}])
    
    try:
        import json
        result = json.loads(response.content)
        
        return {
            "intent": result.get("intent", "consultation"),
            "problem_category": result.get("category", "other"),
            "urgency_level": result.get("urgency", "normal")
        }
    except:
        # 如果解析失败，使用默认值
        return {
            "intent": "consultation",
            "problem_category": "other", 
            "urgency_level": "normal"
        }

# 5. 客服智能体节点
def customer_service_agent(state: CustomerServiceState) -> dict:
    """
    客服智能体：提供个性化客服服务
    """
    messages = state["messages"]
    user_tier = state.get("user_tier", "Regular")
    intent = state.get("intent", "consultation")
    category = state.get("problem_category", "other")
    
    # 根据用户等级调整服务策略
    if user_tier == "VIP":
        service_level = "尊贵的VIP用户，我会为您提供最优质的服务。"
    elif user_tier == "Premium":
        service_level = "感谢您对我们的支持，我会认真为您解决问题。"
    else:
        service_level = "感谢您的咨询，我会尽力为您提供帮助。"
    
    # 创建带工具的客服模型
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    model_with_tools = model.bind_tools([
        search_knowledge_base,
        create_support_ticket,
        escalate_to_human
    ])
    
    # 构建系统提示
    system_prompt = f"""
你是一个专业的客服智能体。你的特点：
- 友好、耐心、专业
- {service_level}
- 当前用户意图：{intent}
- 问题分类：{category}

服务原则：
1. 优先使用知识库查找答案
2. 对于复杂问题，创建工单跟进
3. 必要时升级至人工客服
4. 始终保持礼貌和专业
"""
    
    full_messages = [
        SystemMessage(content=system_prompt),
        *messages
    ]
    
    response = model_with_tools.invoke(full_messages)
    
    return {"messages": [response]}

# 6. 解决状态检查
def check_resolution(state: CustomerServiceState) -> Literal["resolved", "needs_followup", "escalate"]:
    """
    检查问题是否已解决
    """
    messages = state["messages"]
    if not messages:
        return "needs_followup"
    
    last_message = messages[-1]
    
    # 简单的解决状态检查逻辑
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]['name']
        if tool_name == "escalate_to_human":
            return "escalate"
        elif tool_name == "create_support_ticket":
            return "needs_followup"
    
    # 可以加入更复杂的NLP分析来判断用户是否满意
    urgency = state.get("urgency_level", "normal")
    if urgency in ["high", "urgent"]:
        return "escalate"
    
    return "resolved"

# 7. 满意度收集节点
def satisfaction_survey(state: CustomerServiceState) -> dict:
    """
    满意度调查节点
    """
    survey_message = AIMessage(content="""
感谢您使用我们的客服服务！为了持续改进服务质量，请您对本次服务进行评价：

⭐⭐⭐⭐⭐ 非常满意
⭐⭐⭐⭐ 满意
⭐⭐⭐ 一般
⭐⭐ 不满意
⭐ 非常不满意

您还有其他问题需要帮助吗？
""")
    
    return {
        "messages": [survey_message],
        "resolved": True
    }

# 8. 构建客服系统图
def create_customer_service_system():
    """创建客服系统图"""
    
    # 创建工具节点
    tools = [search_knowledge_base, create_support_ticket, escalate_to_human]
    tool_node = ToolNode(tools)
    
    graph = StateGraph(CustomerServiceState)
    
    # 添加节点
    graph.add_node("classify_intent", intent_classifier)
    graph.add_node("customer_service", customer_service_agent)
    graph.add_node("tools", tool_node)
    graph.add_node("satisfaction", satisfaction_survey)
    
    # 添加边
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "customer_service")
    
    # 条件边：是否需要调用工具
    graph.add_conditional_edges(
        "customer_service",
        tools_condition,
        {
            "tools": "tools",
            END: "satisfaction"
        }
    )
    
    # 工具执行后回到客服
    graph.add_edge("tools", "customer_service")
    
    # 满意度调查后结束
    graph.add_edge("satisfaction", END)
    
    # 编译系统
    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)

# 9. 使用演示
def demo_customer_service():
    """演示客服系统"""
    
    cs_system = create_customer_service_system()
    
    # 模拟不同用户的对话
    test_cases = [
        {
            "config": {"configurable": {"thread_id": "vip-user-1"}},
            "state": {
                "user_id": "VIP001",
                "user_name": "张总",
                "user_tier": "VIP"
            },
            "query": "我的VIP订单延迟发货了，这严重影响了我的商务计划"
        },
        {
            "config": {"configurable": {"thread_id": "regular-user-1"}}, 
            "state": {
                "user_id": "REG001",
                "user_name": "小明",
                "user_tier": "Regular"
            },
            "query": "我忘记了登录密码，该怎么重置？"
        }
    ]
    
    print("=== 客服系统演示 ===\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"--- 案例 {i}: {case['state']['user_name']} ({case['state']['user_tier']}) ---")
        print(f"用户问题: {case['query']}")
        
        # 初始化状态并调用系统
        initial_state = {
            **case['state'],
            "messages": [HumanMessage(content=case['query'])],
            "resolved": False
        }
        
        result = cs_system.invoke(initial_state, case['config'])
        
        print("客服回复:")
        for msg in result["messages"]:
            if hasattr(msg, 'content') and not msg.content.startswith('['):
                print(f"  {msg.content}")
        
        print(f"意图识别: {result.get('intent', 'N/A')}")
        print(f"问题分类: {result.get('problem_category', 'N/A')}")
        print(f"紧急程度: {result.get('urgency_level', 'N/A')}")
        print(f"解决状态: {'已解决' if result.get('resolved', False) else '处理中'}")
        print("="*60)

if __name__ == "__main__":
    demo_customer_service()
```

### 3.2 文档处理工作流

```python
"""
智能文档处理工作流示例
展示：文档解析、内容提取、信息验证、格式转换
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import InMemorySaver
import json
import re
from datetime import datetime

# 1. 文档处理状态定义
class DocumentProcessingState(TypedDict):
    """文档处理工作流状态"""
    messages: Annotated[list[AnyMessage], add_messages]
    
    # 文档信息
    document_id: str
    document_type: str          # invoice, contract, report, email
    raw_content: str            # 原始文档内容
    
    # 处理结果
    extracted_entities: Dict[str, Any]  # 提取的实体信息
    structured_data: Dict[str, Any]     # 结构化数据
    validation_results: Dict[str, Any]  # 验证结果
    
    # 处理状态
    processing_stage: str       # parsing, extracting, validating, formatting
    errors: List[str]          # 处理错误列表
    confidence_score: float    # 整体置信度

# 2. 文档解析工具
@tool
def parse_document_content(raw_content: str, document_type: str) -> str:
    """
    解析文档内容，提取关键信息
    
    参数：
        raw_content: 原始文档内容
        document_type: 文档类型
    
    返回：
        解析后的结构化信息
    """
    
    try:
        # 模拟不同类型文档的解析逻辑
        if document_type == "invoice":
            # 发票信息提取
            patterns = {
                "invoice_number": r"发票号码?[：:]\s*(\w+)",
                "amount": r"金额[：:]\s*[￥¥]?(\d+(?:\.\d{2})?)",
                "date": r"日期[：:]\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
                "company": r"公司[：:]?\s*([^\n]+)"
            }
        
        elif document_type == "contract":
            # 合同信息提取
            patterns = {
                "contract_number": r"合同编号[：:]\s*(\w+)",
                "parties": r"甲方[：:]\s*([^\n]+)",
                "amount": r"合同金额[：:]\s*[￥¥]?(\d+(?:\.\d{2})?)",
                "effective_date": r"生效日期[：:]\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
                "expiry_date": r"到期日期[：:]\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})"
            }
        
        else:
            # 通用模式
            patterns = {
                "date": r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
                "amount": r"[￥¥$]?(\d+(?:\.\d{2})?)",
                "email": r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                "phone": r"(\d{3}-?\d{4}-?\d{4}|\d{11})"
            }
        
        # 执行模式匹配
        extracted = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, raw_content)
            if matches:
                extracted[key] = matches[0] if len(matches) == 1 else matches
        
        return json.dumps(extracted, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"解析错误：{str(e)}"

@tool
def validate_extracted_data(extracted_data: str, document_type: str) -> str:
    """
    验证提取的数据的有效性和完整性
    
    参数：
        extracted_data: 提取的数据（JSON字符串）
        document_type: 文档类型
    
    返回：
        验证结果
    """
    
    try:
        data = json.loads(extracted_data)
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "confidence": 1.0
        }
        
        # 根据文档类型进行特定验证
        if document_type == "invoice":
            # 发票验证规则
            if "invoice_number" not in data:
                validation_results["errors"].append("缺少发票号码")
                validation_results["is_valid"] = False
            
            if "amount" in data:
                try:
                    amount = float(data["amount"])
                    if amount <= 0:
                        validation_results["errors"].append("金额必须大于0")
                        validation_results["is_valid"] = False
                except ValueError:
                    validation_results["errors"].append("金额格式不正确")
                    validation_results["is_valid"] = False
            
            if "date" in data:
                # 验证日期格式
                date_str = data["date"]
                try:
                    datetime.strptime(date_str.replace('/', '-'), '%Y-%m-%d')
                except ValueError:
                    validation_results["warnings"].append("日期格式可能不标准")
                    validation_results["confidence"] *= 0.8
        
        elif document_type == "contract":
            # 合同验证规则
            required_fields = ["contract_number", "parties", "amount"]
            for field in required_fields:
                if field not in data:
                    validation_results["errors"].append(f"缺少必填字段：{field}")
                    validation_results["is_valid"] = False
            
            # 验证日期逻辑
            if "effective_date" in data and "expiry_date" in data:
                try:
                    effective = datetime.strptime(data["effective_date"].replace('/', '-'), '%Y-%m-%d')
                    expiry = datetime.strptime(data["expiry_date"].replace('/', '-'), '%Y-%m-%d')
                    if effective >= expiry:
                        validation_results["errors"].append("生效日期不能晚于到期日期")
                        validation_results["is_valid"] = False
                except ValueError:
                    validation_results["warnings"].append("日期格式验证失败")
                    validation_results["confidence"] *= 0.7
        
        # 计算最终置信度
        if validation_results["errors"]:
            validation_results["confidence"] *= 0.5
        if validation_results["warnings"]:
            validation_results["confidence"] *= 0.9
        
        return json.dumps(validation_results, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"验证错误：{str(e)}"

@tool
def format_structured_output(validated_data: str, validation_results: str, output_format: str = "json") -> str:
    """
    将验证后的数据格式化为指定输出格式
    
    参数：
        validated_data: 验证后的数据
        validation_results: 验证结果
        output_format: 输出格式 (json, xml, csv)
    
    返回：
        格式化后的数据
    """
    
    try:
        data = json.loads(validated_data)
        validation = json.loads(validation_results)
        
        # 准备最终输出
        final_output = {
            "data": data,
            "validation": validation,
            "processed_at": datetime.now().isoformat(),
            "format_version": "1.0"
        }
        
        if output_format.lower() == "json":
            return json.dumps(final_output, ensure_ascii=False, indent=2)
        
        elif output_format.lower() == "xml":
            # 简单的XML转换
            xml_output = "<document_processing>\n"
            xml_output += f"  <processed_at>{final_output['processed_at']}</processed_at>\n"
            xml_output += f"  <is_valid>{validation['is_valid']}</is_valid>\n"
            xml_output += f"  <confidence>{validation['confidence']}</confidence>\n"
            xml_output += "  <data>\n"
            
            for key, value in data.items():
                xml_output += f"    <{key}>{value}</{key}>\n"
            
            xml_output += "  </data>\n</document_processing>"
            return xml_output
        
        elif output_format.lower() == "csv":
            # CSV格式输出
            csv_lines = ["field,value,confidence"]
            base_confidence = validation["confidence"]
            
            for key, value in data.items():
                csv_lines.append(f"{key},{value},{base_confidence}")
            
            return "\n".join(csv_lines)
        
        else:
            return json.dumps(final_output, ensure_ascii=False, indent=2)
            
    except Exception as e:
        return f"格式化错误：{str(e)}"

# 3. 文档处理节点实现
def document_parser(state: DocumentProcessingState) -> dict:
    """
    文档解析节点：解析原始文档内容
    """
    
    raw_content = state["raw_content"]
    doc_type = state["document_type"]
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    # 使用LLM增强解析能力
    parsing_prompt = f"""
请分析以下{doc_type}文档，提取关键信息：

文档内容：
{raw_content}

请提取并识别：
1. 所有重要的实体信息
2. 日期、金额、编号等关键数据
3. 联系信息和地址
4. 其他与文档类型相关的重要信息

以JSON格式返回提取结果。
"""
    
    response = model.invoke([{"role": "user", "content": parsing_prompt}])
    
    # 尝试解析LLM返回的JSON
    try:
        extracted = json.loads(response.content)
    except json.JSONDecodeError:
        # 如果LLM返回不是有效JSON，使用工具解析
        extracted = {"llm_extraction": response.content}
    
    return {
        "messages": [AIMessage(content=f"文档解析完成，提取到 {len(extracted)} 个数据字段")],
        "extracted_entities": extracted,
        "processing_stage": "extracting"
    }

def data_validator(state: DocumentProcessingState) -> dict:
    """
    数据验证节点：验证提取数据的准确性和完整性
    """
    
    extracted_data = state["extracted_entities"]
    doc_type = state["document_type"]
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    validation_prompt = f"""
请验证以下从{doc_type}文档中提取的数据：

提取的数据：
{json.dumps(extracted_data, ensure_ascii=False, indent=2)}

验证要求：
1. 检查数据的完整性和准确性
2. 验证数据格式是否符合标准
3. 检查数据之间的逻辑一致性
4. 评估提取质量和置信度

请返回验证结果，包括错误、警告和置信度评分。
"""
    
    response = model.invoke([{"role": "user", "content": validation_prompt}])
    
    # 解析验证结果
    validation_results = {
        "validation_summary": response.content,
        "timestamp": datetime.now().isoformat()
    }
    
    # 基于内容计算置信度
    content = response.content.lower()
    if "错误" in content or "error" in content:
        confidence = 0.6
    elif "警告" in content or "warning" in content:
        confidence = 0.8
    else:
        confidence = 0.95
    
    return {
        "messages": [AIMessage(content=f"数据验证完成，置信度：{confidence:.2f}")],
        "validation_results": validation_results,
        "confidence_score": confidence,
        "processing_stage": "validating"
    }

def output_formatter(state: DocumentProcessingState) -> dict:
    """
    输出格式化节点：生成最终的结构化输出
    """
    
    extracted_data = state["extracted_entities"]
    validation_results = state["validation_results"]
    confidence = state["confidence_score"]
    
    # 构建结构化输出
    structured_output = {
        "document_metadata": {
            "document_id": state["document_id"],
            "document_type": state["document_type"],
            "processing_timestamp": datetime.now().isoformat(),
            "confidence_score": confidence
        },
        "extracted_data": extracted_data,
        "validation_results": validation_results,
        "processing_summary": {
            "total_fields_extracted": len(extracted_data),
            "validation_passed": confidence > 0.7,
            "processing_stage": "completed"
        }
    }
    
    return {
        "messages": [AIMessage(content="文档处理完成，已生成结构化输出")],
        "structured_data": structured_output,
        "processing_stage": "completed"
    }

# 4. 处理质量检查
def quality_check(state: DocumentProcessingState) -> str:
    """
    质量检查：决定是否需要重新处理或人工干预
    """
    
    confidence = state.get("confidence_score", 0.0)
    errors = state.get("errors", [])
    
    if errors:
        return "error_handling"
    elif confidence < 0.5:
        return "manual_review"
    elif confidence < 0.8:
        return "validation_needed"
    else:
        return "completed"

# 5. 构建文档处理工作流
def create_document_processing_workflow():
    """创建文档处理工作流图"""
    
    # 创建工具节点
    tools = [parse_document_content, validate_extracted_data, format_structured_output]
    from langgraph.prebuilt import ToolNode
    tool_node = ToolNode(tools)
    
    graph = StateGraph(DocumentProcessingState)
    
    # 添加处理节点
    graph.add_node("parse", document_parser)
    graph.add_node("validate", data_validator)
    graph.add_node("format", output_formatter)
    graph.add_node("tools", tool_node)
    
    # 定义处理流程
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "validate")
    graph.add_edge("validate", "format")
    
    # 质量检查的条件边
    graph.add_conditional_edges(
        "format",
        quality_check,
        {
            "completed": END,
            "validation_needed": "validate",
            "manual_review": END,  # 实际应用中可以路由到人工审核
            "error_handling": END
        }
    )
    
    # 编译工作流
    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)

# 6. 使用演示
def demo_document_processing():
    """演示文档处理工作流"""
    
    workflow = create_document_processing_workflow()
    
    # 模拟文档样本
    sample_documents = [
        {
            "document_id": "INV-2024-001",
            "document_type": "invoice",
            "raw_content": """
发票号码: INV-2024-001
开票日期: 2024-01-15
公司: 北京科技有限公司
金额: ¥15,800.00
税率: 13%
联系电话: 010-12345678
""",
            "config": {"configurable": {"thread_id": "doc-processing-1"}}
        },
        {
            "document_id": "CONTRACT-2024-001",
            "document_type": "contract",
            "raw_content": """
合同编号: CONTRACT-2024-001
甲方: 上海贸易公司
乙方: 深圳制造有限公司
合同金额: ¥500,000.00
生效日期: 2024-02-01
到期日期: 2024-12-31
签署地点: 上海市
""",
            "config": {"configurable": {"thread_id": "doc-processing-2"}}
        }
    ]
    
    print("=== 智能文档处理工作流演示 ===\n")
    
    for doc in sample_documents:
        print(f"--- 处理文档: {doc['document_id']} ({doc['document_type']}) ---")
        
        # 初始状态
        initial_state = {
            "document_id": doc["document_id"],
            "document_type": doc["document_type"],
            "raw_content": doc["raw_content"],
            "processing_stage": "starting",
            "errors": [],
            "confidence_score": 0.0
        }
        
        # 执行处理流程
        result = workflow.invoke(initial_state, doc["config"])
        
        # 输出处理结果
        print("处理摘要:")
        print(f"  - 处理阶段: {result['processing_stage']}")
        print(f"  - 置信度: {result['confidence_score']:.2f}")
        print(f"  - 提取字段数: {len(result.get('extracted_entities', {}))}")
        
        if result.get("structured_data"):
            print("结构化输出:")
            summary = result["structured_data"]["processing_summary"]
            print(f"  - 验证通过: {summary['validation_passed']}")
            print(f"  - 提取字段: {summary['total_fields_extracted']}")
        
        print("提取的数据:")
        for key, value in result.get("extracted_entities", {}).items():
            print(f"  - {key}: {value}")
        
        print("="*60)

if __name__ == "__main__":
    demo_document_processing()
```

## 4. 总结和最佳实践

### 4.1 使用示例总结

以上示例展示了LangGraph在不同场景下的应用：

1. **基础聊天机器人**：展示了最简单的状态图构建和消息处理
2. **工具调用智能体**：演示了ReAct模式和条件路由的使用
3. **人机交互中断**：展示了中断机制在审核流程中的应用
4. **多智能体协作**：演示了复杂的智能体协作和任务分工
5. **客服系统**：实际业务场景中的综合应用
6. **文档处理**：展示了结构化数据处理和工作流自动化

### 4.2 代码结构最佳实践

1. **状态设计**：
   - 使用TypedDict定义清晰的状态结构
   - 合理使用Annotated和reducer函数
   - 包含必要的元数据和处理状态

2. **节点实现**：
   - 保持节点功能单一和职责清晰
   - 使用适当的错误处理机制
   - 返回明确的状态更新

3. **图构建**：
   - 合理设计边和条件路由
   - 使用检查点实现状态持久化
   - 设置适当的中断点

4. **工具集成**：
   - 使用@tool装饰器定义工具函数
   - 提供清晰的工具描述和参数说明
   - 实现适当的错误处理和回退机制

### 4.3 性能优化建议

1. **并发处理**：利用工具的并行执行能力
2. **状态管理**：避免状态过于复杂，影响性能
3. **检查点优化**：合理设置检查点频率
4. **资源管理**：注意模型调用的成本控制

这些示例为开发者提供了全面的参考，展示了如何使用LangGraph构建各种复杂的AI应用程序。
