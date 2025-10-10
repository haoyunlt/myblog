---
title: "VoiceHelper源码剖析 - 09Agent服务"
date: 2025-10-10T09:00:00+08:00
draft: false
tags: ["源码剖析", "VoiceHelper", "AI Agent", "LangGraph", "任务规划", "工具调用"]
categories: ["VoiceHelper", "源码剖析"]
description: "Agent服务详解：任务规划与执行、自我反思机制、工具调用管理、Multi-Agent协作、LangGraph工作流、权限管理"
weight: 10
---

# VoiceHelper-09-Agent服务

## 1. 模块概览

### 1.1 职责边界

**核心职责**:
- **任务规划**:使用LLM将复杂任务分解为可执行步骤
- **任务执行**:调用工具和LLM完成每个步骤
- **自我反思**:验证执行结果,自动纠正错误
- **工具调用**:管理和调用各种工具(搜索、计算、文件操作等)
- **多Agent协作**:多个Agent通过消息传递协同工作
- **权限管理**:工具访问控制和审计日志

**输入**:
- HTTP请求(任务执行、规划、工具调用)
- 任务描述(自然语言)
- 可用工具列表
- 上下文信息

**输出**:
- 任务执行计划(分步骤)
- 执行结果
- 反思和改进建议
- 工具调用结果
- 任务状态和进度

**上下游依赖**:
- **上游**:API网关、前端客户端
- **下游**:
  - LLM Router服务(调用LLM生成计划/执行)
  - GraphRAG服务(知识检索)
  - 外部API(搜索引擎、数据库等)
  - Redis(任务状态缓存)

**生命周期**:
- **启动**:加载配置 → 初始化LangGraph工作流 → 注册工具 → 初始化权限管理器 → 监听HTTP(:8003)
- **运行**:接收任务 → 规划 → 执行 → 反思 → 返回结果
- **关闭**:停止接收请求 → 等待现有任务完成 → 保存状态

---

### 1.2 模块架构图

```mermaid
flowchart TB
    subgraph "Agent Service :8003"
        direction TB
        
        subgraph "API层"
            API_EXEC[Execute API<br/>执行任务]
            API_PLAN[Plan API<br/>生成计划]
            API_TOOL[Tool API<br/>工具调用]
            API_STATUS[Status API<br/>查询状态]
        end
        
        subgraph "LangGraph工作流"
            direction LR
            NODE_PLANNER[Planner Node<br/>任务规划]
            NODE_EXECUTOR[Executor Node<br/>步骤执行]
            NODE_CRITIC[Critic Node<br/>结果反思]
            NODE_SYNTH[Synthesizer Node<br/>结果综合]
            
            NODE_PLANNER -->|execute| NODE_EXECUTOR
            NODE_PLANNER -->|refine| NODE_PLANNER
            NODE_PLANNER -->|end| NODE_SYNTH
            
            NODE_EXECUTOR -->|continue| NODE_EXECUTOR
            NODE_EXECUTOR -->|reflect| NODE_CRITIC
            NODE_EXECUTOR -->|synthesize| NODE_SYNTH
            
            NODE_CRITIC -->|replan| NODE_PLANNER
        end
        
        subgraph "工具系统"
            TOOL_REG[ToolRegistry<br/>工具注册表]
            TOOL_EXEC[ToolExecutor<br/>工具执行器]
            TOOL_PERM[PermissionManager<br/>权限管理器]
            
            TOOL_REG --> TOOL_EXEC
            TOOL_EXEC --> TOOL_PERM
        end
        
        subgraph "反思系统"
            REFLECT_AGENT[ReflectionAgent<br/>反思Agent]
            REFLECT_VAL[Validation<br/>结果验证]
            REFLECT_CORR[Correction<br/>错误纠正]
            REFLECT_IMP[Improvement<br/>结果改进]
            
            REFLECT_AGENT --> REFLECT_VAL & REFLECT_CORR & REFLECT_IMP
        end
        
        subgraph "多Agent系统"
            MULTI_SYS[MultiAgentSystem<br/>多Agent系统]
            AGENT_COORD[Coordinator<br/>协调者]
            AGENT_PLAN[Planner<br/>规划者]
            AGENT_EXEC[Executor<br/>执行者]
            AGENT_CRITIC[Critic<br/>评论者]
            
            MULTI_SYS --> AGENT_COORD & AGENT_PLAN & AGENT_EXEC & AGENT_CRITIC
        end
        
        API_EXEC --> NODE_PLANNER
        API_PLAN --> NODE_PLANNER
        API_TOOL --> TOOL_EXEC
        
        NODE_EXECUTOR --> TOOL_EXEC
        NODE_CRITIC --> REFLECT_AGENT
    end
    
    subgraph "外部依赖"
        LLM[LLM Router<br/>GPT-4/Claude]
        GRAPHRAG[GraphRAG<br/>知识检索]
        REDIS[Redis<br/>任务状态]
    end
    
    NODE_PLANNER -.调用.-> LLM
    NODE_EXECUTOR -.调用.-> LLM
    NODE_EXECUTOR -.调用.-> GRAPHRAG
    API_STATUS -.读取.-> REDIS
    
    style NODE_PLANNER fill:#87CEEB
    style NODE_EXECUTOR fill:#FFB6C1
    style NODE_CRITIC fill:#98FB98
    style NODE_SYNTH fill:#DDA0DD
```

### 架构要点说明

#### 1. LangGraph状态图工作流
- **Planner节点**:使用LLM将任务分解为步骤
- **Executor节点**:循环执行每个步骤
- **Critic节点**:反思验证执行结果
- **Synthesizer节点**:综合最终答案
- **条件边**:根据状态动态路由(execute/refine/reflect/synthesize)

#### 2. 工具系统
- **ToolRegistry**:注册和管理工具
- **ToolExecutor**:执行工具调用
- **PermissionManager**:权限控制和审计
- **内置工具**:search/calculator/read_file等

#### 3. 反思机制
- **Validation**:验证输出准确性、完整性、相关性
- **Correction**:自动纠正错误(迭代最多3次)
- **Improvement**:改进输出质量
- **Learning**:积累反思历史用于学习

#### 4. 多Agent协作
- **角色定义**:Coordinator/Planner/Executor/Critic/Specialist
- **消息传递**:点对点、广播、订阅
- **任务分配**:智能选择合适的Agent
- **协作决策**:投票、共识、仲裁

---

## 2. 对外API列表与规格

### 2.1 执行任务

**基本信息**:
- 名称:`ExecuteTask`
- 协议与方法:HTTP POST `/api/v1/execute`
- 幂等性:否(每次执行可能产生不同结果)
- Content-Type:`application/json`

**请求结构体**:
```python
class ExecuteRequest(BaseModel):
    task: str                      # 任务描述
    tools: Optional[List[str]]     # 可用工具列表
    context: Optional[Dict]        # 上下文信息
    max_iterations: int = 10       # 最大迭代次数(1-50)
```

**字段表**:
| 字段 | 类型 | 必填 | 默认 | 约束 | 说明 |
|------|------|---:|------|------|------|
| task | string | 是 | - | 长度≤5000 | 任务描述(自然语言) |
| tools | array | 否 | [] | 工具名称列表 | 可用工具,空表示使用全部 |
| context | object | 否 | {} | JSON对象 | 上下文信息(变量、约束等) |
| max_iterations | int | 否 | 10 | 1-50 | 最大迭代次数,防止死循环 |

**响应结构体**:
```python
{
    "code": 0,
    "message": "success",
    "data": {
        "task_id": "uuid-xxx",              # 任务ID
        "status": "processing",             # 状态:processing/completed/failed
        "message": "任务已创建，正在后台执行",
        "elapsed_time": 0.123               # 耗时(秒)
    }
}
```

**入口函数与核心代码**:
```python
# algo/agent-service/app/routes.py

@router.post("/execute")
async def execute_task(
    request: ExecuteRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    执行Agent任务
    
    流程:
    1. 生成task_id
    2. 初始化LangGraph工作流
    3. 后台执行任务(异步)
    4. 立即返回task_id
    """
    start_time = time.time()
    task_id = str(uuid.uuid4())
    
    logger.business("Agent执行请求", context={
        "task_id": task_id,
        "task": request.task[:100],
        "tools": request.tools,
        "max_iterations": request.max_iterations,
    })
    
    # ✅ 使用LangGraph完整工作流
    from core.langgraph_workflow import CompleteLangGraphWorkflow
    
    # 配置工作流
    config = {
        'llm_config': {
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 2000
        },
        'tools': request.tools or [],
        'max_iterations': request.max_iterations
    }
    
    workflow = CompleteLangGraphWorkflow(config)
    
    # 后台执行任务
    async def run_agent_task():
        try:
            logger.info(f"开始执行任务: {task_id}")
            
            result = await workflow.run(
                task=request.task,
                context=request.context,
                max_iterations=request.max_iterations
            )
            
            if result.get('success'):
                logger.info(
                    f"任务执行完成: {task_id}",
                    iterations=result.get('iterations'),
                    final_result_length=len(result.get('final_result', ''))
                )
            else:
                logger.error(
                    f"任务执行失败: {task_id}",
                    error=result.get('error')
                )
            
            # 此处省略将结果保存到Redis/数据库的逻辑
            
        except Exception as e:
            logger.error(f"任务执行异常: {task_id}: {e}", exc_info=True)
    
    background_tasks.add_task(run_agent_task)
    
    elapsed_time = time.time() - start_time
    
    return success_response({
        "task_id": task_id,
        "status": "processing",
        "message": "任务已创建，正在后台执行",
        "elapsed_time": elapsed_time,
    })
```

**调用链与核心函数**:

```python
# 1. CompleteLangGraphWorkflow.run() - 工作流执行
async def run(
    self,
    task: str,
    context: Optional[Dict[str, Any]] = None,
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    运行工作流
    
    流程:
    1. 初始化状态
    2. 执行状态图(自动路由)
    3. 返回最终结果
    """
    logger.info(f"启动LangGraph工作流: {task[:100]}")
    
    # 初始状态
    initial_state = {
        "task": task,
        "context": context or {},
        "plan": [],
        "current_step": 0,
        "execution_results": [],
        "need_reflection": False,
        "need_replan": False,
        "iterations": 0,
        "max_iterations": max_iterations,
        "final_result": "",
        "error": ""
    }
    
    # 执行工作流(状态图自动路由)
    result = await self.app.ainvoke(initial_state)
    
    logger.info("工作流执行完成")
    
    return {
        "success": True,
        "final_result": result.get("final_result", ""),
        "execution_results": result.get("execution_results", []),
        "iterations": result.get("iterations", 0),
        "error": result.get("error", "")
    }

# 2. _planning_node() - 规划节点
async def _planning_node(self, state: AgentState) -> Dict:
    """
    规划节点 - 任务分解
    
    使用LLM将任务分解为具体步骤
    
    返回:
    {
        "plan": [
            {
                "step_number": 1,
                "description": "搜索相关文档",
                "tool": "search",
                "input": {"query": "..."},
                "expected_output": "文档列表"
            },
            ...
        ],
        "current_step": 0,
        "iterations": iterations + 1
    }
    """
    task = state["task"]
    context = state["context"]
    iterations = state.get("iterations", 0)
    
    logger.info(f"开始任务规划 (迭代 {iterations}): {task[:100]}")
    
    # 构建规划提示
    planning_prompt = f"""
    任务: {task}
    
    上下文: {json.dumps(context, ensure_ascii=False, indent=2)}
    
    请将任务分解为具体的执行步骤。每个步骤包含:
    1. step_number: 步骤编号
    2. description: 步骤描述
    3. tool: 需要使用的工具 (如果需要)
    4. input: 工具输入参数
    5. expected_output: 预期输出
    
    返回JSON格式的计划列表。
    """
    
    # 调用LLM生成计划
    response = await self.llm.ainvoke([
        SystemMessage(content="你是一个任务规划专家,擅长将复杂任务分解为可执行步骤。"),
        HumanMessage(content=planning_prompt)
    ])
    
    # 解析计划
    plan_text = response.content
    plan = self._parse_plan(plan_text)
    
    logger.info(f"规划完成: {len(plan)}个步骤")
    
    return {
        "plan": plan,
        "current_step": 0,
        "iterations": iterations + 1,
        "need_replan": False
    }

# 3. _execution_node() - 执行节点
async def _execution_node(self, state: AgentState) -> Dict:
    """
    执行节点 - 执行当前步骤
    
    流程:
    1. 检查是否完成所有步骤
    2. 执行当前步骤(调用工具或LLM)
    3. 保存执行结果
    4. 更新状态
    """
    plan = state["plan"]
    current_step = state["current_step"]
    execution_results = state.get("execution_results", [])
    
    if current_step >= len(plan):
        logger.info("所有步骤已执行完毕")
        return {
            "current_step": current_step,
            "need_reflection": False
        }
    
    step = plan[current_step]
    logger.info(
        f"执行步骤 {current_step + 1}/{len(plan)}: "
        f"{step.get('description', '')}"
    )
    
    # 执行步骤
    result = await self._execute_step(step, state)
    
    # 保存结果
    execution_results.append({
        "step": current_step,
        "step_description": step.get('description'),
        "result": result,
        "success": True,
        "error": None
    })
    
    logger.info(f"步骤 {current_step + 1} 执行成功")
    
    return {
        "current_step": current_step + 1,
        "execution_results": execution_results,
        "need_reflection": True  # 执行后需要反思
    }

# 4. _execute_step() - 执行单个步骤
async def _execute_step(
    self,
    step: Dict[str, Any],
    state: AgentState
) -> Any:
    """
    执行单个步骤
    
    根据步骤定义:
    - 如果指定了tool,使用tool_executor执行
    - 否则使用LLM执行
    """
    tool_name = step.get('tool')
    
    if tool_name and self.tool_executor:
        # 使用工具执行
        tool_input = step.get('input', {})
        
        result = await self.tool_executor.ainvoke({
            "tool": tool_name,
            "tool_input": tool_input
        })
        
        return result
    else:
        # 使用LLM执行
        prompt = f"""
        步骤描述: {step.get('description')}
        输入: {step.get('input', {})}
        
        请完成这个步骤并返回结果。
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

# 5. _reflection_node() - 反思节点
async def _reflection_node(self, state: AgentState) -> Dict:
    """
    反思节点 - 验证执行结果
    
    使用LLM评估:
    1. 执行结果是否符合预期?
    2. 是否需要调整计划?
    3. 是否有错误需要处理?
    4. 下一步应该做什么?
    
    返回:
    {
        "need_replan": true/false
    }
    """
    execution_results = state.get("execution_results", [])
    plan = state["plan"]
    
    logger.info(f"开始反思: {len(execution_results)}个执行结果")
    
    # 构建反思提示
    reflection_prompt = f"""
    计划: {json.dumps(plan, ensure_ascii=False, indent=2)}
    
    执行结果: {json.dumps(execution_results, ensure_ascii=False, indent=2)}
    
    请评估执行结果:
    1. 执行结果是否符合预期?
    2. 是否需要调整计划?
    3. 是否有错误需要处理?
    4. 下一步应该做什么?
    
    返回JSON格式:
    {{
        "assessment": "评估结论",
        "need_replan": true/false,
        "suggestions": "改进建议"
    }}
    """
    
    response = await self.llm.ainvoke([
        SystemMessage(content="你是一个结果验证专家,擅长评估任务执行质量。"),
        HumanMessage(content=reflection_prompt)
    ])
    
    # 解析反思结果
    reflection_text = response.content
    reflection = self._parse_reflection(reflection_text)
    
    need_replan = reflection.get('need_replan', False)
    
    logger.info(f"反思完成: need_replan={need_replan}")
    
    return {
        "need_replan": need_replan
    }

# 6. _synthesis_node() - 综合节点
async def _synthesis_node(self, state: AgentState) -> Dict:
    """
    综合节点 - 生成最终结果
    
    综合所有执行结果,生成完整的最终答案
    """
    task = state["task"]
    execution_results = state.get("execution_results", [])
    
    logger.info("开始综合最终结果")
    
    # 构建综合提示
    synthesis_prompt = f"""
    原始任务: {task}
    
    执行结果: {json.dumps(execution_results, ensure_ascii=False, indent=2)}
    
    请综合所有执行结果,生成完整的最终答案。
    答案应该:
    1. 直接回答原始任务
    2. 整合所有相关信息
    3. 结构清晰,逻辑连贯
    """
    
    response = await self.llm.ainvoke([
        SystemMessage(content="你是一个信息综合专家,擅长整合多个信息源。"),
        HumanMessage(content=synthesis_prompt)
    ])
    
    final_result = response.content
    
    logger.info("最终结果综合完成")
    
    return {
        "final_result": final_result
    }

# 7. 条件边函数
def _should_execute(self, state: AgentState) -> str:
    """
    判断是否应该执行
    
    决策逻辑:
    - iterations >= max_iterations -> "end"
    - need_replan == True -> "refine"
    - plan为空 -> "end"
    - 否则 -> "execute"
    """
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 10)
    need_replan = state.get("need_replan", False)
    plan = state.get("plan", [])
    
    if iterations >= max_iterations:
        logger.warning(f"达到最大迭代次数: {max_iterations}")
        return "end"
    
    if need_replan:
        logger.info("需要重新规划")
        return "refine"
    
    if not plan or len(plan) == 0:
        logger.warning("计划为空")
        return "end"
    
    return "execute"

def _should_reflect(self, state: AgentState) -> str:
    """
    判断是否需要反思
    
    决策逻辑:
    - current_step >= len(plan) -> "synthesize"
    - need_reflection == True -> "reflect"
    - 否则 -> "continue"
    """
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    need_reflection = state.get("need_reflection", False)
    
    if current_step >= len(plan):
        logger.info("所有步骤已完成,进入综合阶段")
        return "synthesize"
    
    if need_reflection:
        logger.info("需要反思验证")
        return "reflect"
    
    return "continue"
```

**时序图(任务执行→结果综合)**:
```mermaid
sequenceDiagram
    autonumber
    participant Client as 客户端
    participant API as Execute API
    participant Workflow as LangGraph工作流
    participant Planner as Planner Node
    participant Executor as Executor Node
    participant Critic as Critic Node
    participant Synth as Synthesizer Node
    participant LLM as LLM Router
    participant Tools as 工具系统
    
    Client->>API: POST /api/v1/execute<br/>{task, tools, max_iterations}
    API->>API: task_id = uuid.uuid4()
    API->>Workflow: 初始化CompleteLangGraphWorkflow(config)
    
    API->>Workflow: 后台执行workflow.run(task, context)
    API-->>Client: 200 OK<br/>{task_id, status: "processing"}
    
    rect rgb(135, 206, 235)
    note right of Workflow: Planner阶段
    Workflow->>Planner: _planning_node(state)
    Planner->>LLM: 请将任务分解为步骤
    LLM-->>Planner: plan列表[{step_number, description, tool, input}]
    Planner->>Planner: _parse_plan(plan_text)
    Planner-->>Workflow: {plan, current_step=0, iterations++}
    end
    
    Workflow->>Workflow: _should_execute(state)<br/>→ "execute"
    
    loop 执行每个步骤
        rect rgb(255, 182, 193)
        note right of Workflow: Executor阶段
        Workflow->>Executor: _execution_node(state)
        Executor->>Executor: step = plan[current_step]
        
        alt 步骤需要工具
            Executor->>Tools: tool_executor.ainvoke(tool_name, input)
            Tools-->>Executor: tool_result
        else 步骤不需要工具
            Executor->>LLM: 请完成步骤: {description, input}
            LLM-->>Executor: step_result
        end
        
        Executor->>Executor: execution_results.append(result)
        Executor-->>Workflow: {current_step++, need_reflection=True}
        end
        
        Workflow->>Workflow: _should_reflect(state)<br/>→ "reflect"
        
        rect rgb(152, 251, 152)
        note right of Workflow: Critic阶段
        Workflow->>Critic: _reflection_node(state)
        Critic->>LLM: 评估执行结果,是否需要重新规划?
        LLM-->>Critic: {assessment, need_replan, suggestions}
        Critic-->>Workflow: {need_replan}
        end
        
        alt need_replan == true
            Workflow->>Workflow: _should_execute(state)<br/>→ "refine"
            Workflow->>Planner: 重新规划
        else current_step < len(plan)
            Workflow->>Workflow: _should_reflect(state)<br/>→ "continue"
        end
    end
    
    Workflow->>Workflow: _should_reflect(state)<br/>→ "synthesize"
    
    rect rgb(221, 160, 221)
    note right of Workflow: Synthesizer阶段
    Workflow->>Synth: _synthesis_node(state)
    Synth->>LLM: 综合所有执行结果<br/>生成最终答案
    LLM-->>Synth: final_result
    Synth-->>Workflow: {final_result}
    end
    
    Workflow-->>API: {success, final_result, execution_results, iterations}
    
    note right of Client: 客户端轮询查询状态
    Client->>API: GET /api/v1/tasks/{task_id}
    API-->>Client: {status: "completed", result: final_result}
```

---

### 2.2 生成任务计划

**基本信息**:
- 名称:`PlanTask`
- 协议与方法:HTTP POST `/api/v1/plan`
- 幂等性:否(每次可能生成不同计划)

**请求结构体**:
```python
class PlanRequest(BaseModel):
    task: str                        # 任务描述
    available_tools: List[str] = []  # 可用工具列表
```

**响应结构体**:
```python
{
    "code": 0,
    "message": "success",
    "data": {
        "task": "原始任务",
        "steps": [
            {
                "step_number": 1,
                "description": "分析任务需求",
                "tool": null,
                "estimated_time": 1.0
            },
            {
                "step_number": 2,
                "description": "搜索相关文档",
                "tool": "search",
                "estimated_time": 5.0
            },
            {
                "step_number": 3,
                "description": "整合结果",
                "tool": null,
                "estimated_time": 1.0
            }
        ],
        "total_steps": 3,
        "estimated_total_time": 7.0
    }
}
```

**核心实现**:
```python
@router.post("/plan")
async def plan_task(request: PlanRequest, http_request: Request):
    """
    生成任务计划
    
    使用LangGraph工作流的Planner节点生成计划
    """
    logger.business("任务规划请求", context={
        "task": request.task[:100],
    })
    
    from core.langgraph_workflow import CompleteLangGraphWorkflow
    
    # 配置工作流
    config = {
        'llm_config': {'model': 'gpt-4', 'temperature': 0.7},
        'tools': request.available_tools or [],
        'max_iterations': 1
    }
    
    workflow = CompleteLangGraphWorkflow(config)
    
    # 执行规划节点
    state = {
        "task": request.task,
        "context": {},
        "plan": [],
        "current_step": 0,
        "execution_results": [],
        "need_reflection": False,
        "need_replan": False,
        "iterations": 0,
        "max_iterations": 1,
        "final_result": "",
        "error": ""
    }
    
    result = await workflow._planning_node(state)
    plan = result.get('plan', [])
    
    # 转换为Step格式
    steps = []
    for step_data in plan:
        steps.append(Step(
            step_number=step_data.get('step_number', len(steps) + 1),
            description=step_data.get('description', ''),
            tool=step_data.get('tool'),
            estimated_time=step_data.get('estimated_time', 5.0)
        ))
    
    # 如果规划失败,使用默认计划
    if not steps:
        logger.warning("LLM规划失败,使用默认计划")
        steps = [
            Step(step_number=1, description="分析任务需求", tool=None, estimated_time=1.0),
            Step(step_number=2, description="执行主要任务", tool=None, estimated_time=5.0),
            Step(step_number=3, description="整合结果", tool=None, estimated_time=1.0),
        ]
    
    logger.info(f"任务规划完成: {len(steps)}个步骤")
    
    return success_response({
        "task": request.task,
        "steps": [s.dict() for s in steps],
        "total_steps": len(steps),
        "estimated_total_time": sum(s.estimated_time or 0 for s in steps),
    })
```

---

### 2.3 工具调用

**基本信息**:
- 名称:`CallTool`
- 协议与方法:HTTP POST `/api/v1/tools/call`
- 幂等性:取决于具体工具

**请求结构体**:
```python
class ToolCallRequest(BaseModel):
    tool_name: str                 # 工具名称
    parameters: Dict[str, Any]     # 工具参数
```

**响应结构体**:
```python
{
    "code": 0,
    "message": "success",
    "data": {
        "tool_name": "search",
        "result": "搜索结果...",
        "success": true
    }
}
```

**获取工具列表**:
```python
@router.get("/tools")
async def list_tools():
    """获取可用工具列表"""
    tools = [
        {
            "name": "search",
            "description": "搜索工具",
            "category": "information",
            "parameters": [
                {"name": "query", "type": "string", "required": True}
            ]
        },
        {
            "name": "calculator",
            "description": "计算器",
            "category": "computation",
            "parameters": [
                {"name": "expression", "type": "string", "required": True}
            ]
        },
        {
            "name": "read_file",
            "description": "读取文件内容",
            "category": "system",
            "parameters": [
                {"name": "path", "type": "string", "required": True}
            ]
        },
    ]
    
    return success_response({
        "tools": tools,
        "count": len(tools),
    })
```

---

## 3. 核心功能实现

### 3.1 反思机制(Self-Reflection)

**核心实现**:
```python
# algo/agent-service/core/agent/reflection.py

class ReflectionAgent:
    """
    反思Agent
    
    功能:
    1. 输出验证：检查输出是否符合预期
    2. 错误检测：识别潜在错误
    3. 自动纠正：尝试修正错误
    4. 经验积累：学习历史反思
    """
    
    def __init__(
        self,
        llm_client: Any,
        enable_auto_correction: bool = True,
        max_reflection_iterations: int = 3,
        confidence_threshold: float = 0.7
    ):
        self.llm = llm_client
        self.enable_auto_correction = enable_auto_correction
        self.max_reflection_iterations = max_reflection_iterations
        self.confidence_threshold = confidence_threshold
        
        # 反思历史（用于学习）
        self.reflection_history: List[ReflectionResult] = []
    
    async def reflect(
        self,
        task: str,
        output: Any,
        context: Optional[Dict[str, Any]] = None,
        reflection_type: ReflectionType = ReflectionType.VALIDATION
    ) -> ReflectionResult:
        """
        执行反思（主入口）
        
        流程:
        1. 根据类型选择反思方法(validate/correct/improve)
        2. 记录反思历史
        3. 如果未通过且启用自动纠正,迭代纠正(最多3次)
        4. 返回反思结果
        """
        if reflection_type == ReflectionType.VALIDATION:
            result = await self._validate_output(task, output, context)
        elif reflection_type == ReflectionType.CORRECTION:
            result = await self._correct_output(task, output, context)
        elif reflection_type == ReflectionType.IMPROVEMENT:
            result = await self._improve_output(task, output, context)
        else:
            result = await self._validate_output(task, output, context)
        
        # 记录反思历史
        self.reflection_history.append(result)
        
        # 如果未通过且启用自动纠正，尝试纠正
        if not result.passed and self.enable_auto_correction:
            corrected = await self._iterative_correction(task, output, context)
            if corrected:
                result = corrected
        
        return result
    
    async def _validate_output(
        self,
        task: str,
        output: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ReflectionResult:
        """
        验证输出
        
        使用LLM从4个维度评估:
        1. 准确性(accuracy_score): 0-1
        2. 完整性(completeness_score): 0-1
        3. 相关性(relevance_score): 0-1
        4. 逻辑性(logic_score): 0-1
        
        综合置信度:
        confidence = accuracy*0.4 + completeness*0.3 + relevance*0.2 + logic*0.1
        """
        prompt = f"""请作为一个严格的评审者，验证以下任务的输出是否正确和完整。

任务: {task}

输出:
{self._format_output(output)}

请从以下维度评估:
1. 准确性：输出是否准确无误
2. 完整性：是否完整回答了任务
3. 相关性：输出是否与任务相关
4. 逻辑性：输出是否逻辑连贯

输出JSON格式:
{{
    "passed": true/false,
    "accuracy_score": 0-1,
    "completeness_score": 0-1,
    "relevance_score": 0-1,
    "logic_score": 0-1,
    "feedback": "详细反馈",
    "suggestions": ["改进建议1", "改进建议2"]
}}"""

        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        validation = json.loads(response.choices[0].message.content)
        
        # 计算综合置信度
        confidence = (
            validation.get("accuracy_score", 0) * 0.4 +
            validation.get("completeness_score", 0) * 0.3 +
            validation.get("relevance_score", 0) * 0.2 +
            validation.get("logic_score", 0) * 0.1
        )
        
        return ReflectionResult(
            passed=validation.get("passed", False),
            type=ReflectionType.VALIDATION,
            original_output=output,
            reflected_output=None,
            feedback=validation.get("feedback", ""),
            confidence=confidence,
            suggestions=validation.get("suggestions", []),
            metadata=validation
        )
    
    async def _correct_output(
        self,
        task: str,
        output: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ReflectionResult:
        """
        纠正输出
        
        流程:
        1. 首先验证输出
        2. 如果已通过,直接返回
        3. 如果未通过,使用LLM纠正
        4. 再次验证纠正后的输出
        5. 返回纠正结果
        """
        # 首先验证
        validation = await self._validate_output(task, output, context)
        
        if validation.passed:
            return validation
        
        # 如果未通过，尝试纠正
        prompt = f"""以下输出存在问题，请纠正。

任务: {task}

原始输出:
{self._format_output(output)}

问题反馈: {validation.feedback}

请提供纠正后的输出，确保:
1. 解决所有指出的问题
2. 保持输出格式一致
3. 完整回答任务要求

纠正后的输出:"""

        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        corrected_output = response.choices[0].message.content.strip()
        
        # 再次验证纠正后的输出
        re_validation = await self._validate_output(task, corrected_output, context)
        
        return ReflectionResult(
            passed=re_validation.passed,
            type=ReflectionType.CORRECTION,
            original_output=output,
            reflected_output=corrected_output,
            feedback=f"Original: {validation.feedback}\nCorrected: {re_validation.feedback}",
            confidence=re_validation.confidence,
            suggestions=re_validation.suggestions,
            metadata={
                "original_validation": validation.metadata,
                "corrected_validation": re_validation.metadata
            }
        )
    
    async def _iterative_correction(
        self,
        task: str,
        output: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ReflectionResult]:
        """
        迭代纠正（多次尝试）
        
        最多尝试max_reflection_iterations次(默认3次)
        如果达到confidence_threshold(默认0.7),认为纠正成功
        """
        current_output = output
        
        for iteration in range(self.max_reflection_iterations):
            logger.debug(f"Reflection iteration {iteration + 1}/{self.max_reflection_iterations}")
            
            result = await self._correct_output(task, current_output, context)
            
            if result.passed and result.confidence >= self.confidence_threshold:
                logger.info(f"Reflection succeeded after {iteration + 1} iterations")
                return result
            
            if result.reflected_output:
                current_output = result.reflected_output
            else:
                break
        
        logger.warning(f"Reflection failed after {self.max_reflection_iterations} iterations")
        return None
```

**使用示例**:
```python
# 创建反思Agent
reflection_agent = ReflectionAgent(
    llm_client=llm,
    enable_auto_correction=True,
    max_reflection_iterations=3,
    confidence_threshold=0.7
)

# 反思验证
result = await reflection_agent.reflect(
    task="计算2+2",
    output="5",  # 错误输出
    reflection_type=ReflectionType.VALIDATION
)

print(f"是否通过: {result.passed}")           # False
print(f"置信度: {result.confidence}")         # < 0.7
print(f"反馈: {result.feedback}")            # "输出不准确,2+2应该等于4"
print(f"纠正后输出: {result.reflected_output}") # "4"

# 反思统计
stats = reflection_agent.get_reflection_stats()
print(stats)
# {
#     "total": 10,
#     "passed": 7,
#     "failed": 3,
#     "pass_rate": 0.7,
#     "avg_confidence": 0.82,
#     "by_type": {"validation": 6, "correction": 3, "improvement": 1}
# }
```

---

### 3.2 多Agent协作系统

**核心实现**:
```python
# algo/agent-service/core/agent/multi_agent_system.py

class MultiAgentSystem:
    """
    Multi-Agent协作系统
    
    功能:
    1. Agent管理：注册、注销、状态监控
    2. 消息路由：点对点、广播、订阅
    3. 任务分配：智能分配、负载均衡
    4. 协作决策：投票、共识、仲裁
    """
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.message_bus: asyncio.Queue = asyncio.Queue()
        self.shared_state: Dict[str, Any] = {}
        self.running = False
        
        # 消息历史
        self.message_history: List[AgentMessage] = []
    
    def register_agent(self, agent: Agent) -> None:
        """
        注册Agent
        
        将Agent注册到系统,并共享全局状态
        """
        self.agents[agent.agent_id] = agent
        agent.shared_state = self.shared_state
        logger.info(f"Registered agent: {agent.agent_id} ({agent.role.value})")
    
    async def send_message(self, message: AgentMessage) -> None:
        """
        发送消息
        
        支持两种模式:
        1. 点对点: receiver_id指定具体Agent
        2. 广播: receiver_id="broadcast"
        """
        self.message_history.append(message)
        
        if message.receiver_id == "broadcast":
            # 广播消息
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender_id:
                    await agent.inbox.put(message)
        elif message.receiver_id in self.agents:
            # 点对点消息
            await self.agents[message.receiver_id].inbox.put(message)
        else:
            logger.warning(f"Receiver not found: {message.receiver_id}")
    
    async def assign_task(
        self,
        task: str,
        preferred_role: Optional[AgentRole] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        分配任务给合适的Agent
        
        选择策略:
        1. 状态检查: idle/waiting
        2. 角色匹配: 优先选择preferred_role
        3. 能力检查: 必须具备required_capabilities
        4. 简单策略: 返回第一个候选
        """
        # 选择合适的Agent
        candidate = self._select_agent(preferred_role, required_capabilities)
        
        if not candidate:
            logger.warning("No suitable agent found for task")
            return None
        
        # 发送任务分配消息
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id="system",
            receiver_id=candidate.agent_id,
            type=MessageType.TASK_ASSIGNMENT,
            content=task
        )
        
        await self.send_message(message)
        logger.info(f"Task assigned to agent {candidate.agent_id}")
        
        return candidate.agent_id
    
    async def collaborative_decision(
        self,
        question: str,
        voting_agents: Optional[List[str]] = None,
        consensus_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        协作决策
        
        投票流程:
        1. 选择投票Agent(默认所有Agent)
        2. 收集每个Agent的投票(同意/反对/中立)
        3. 统计投票结果
        4. 如果最多票数占比>= consensus_threshold,做出决策
        5. 否则返回None
        
        返回:
        {
            "decision": "同意"/"反对"/None,
            "consensus": 0.75,
            "votes": {"agent1": "同意", "agent2": "同意", "agent3": "反对"},
            "vote_counts": {"同意": 2, "反对": 1}
        }
        """
        # 选择投票Agent
        if voting_agents:
            voters = [self.agents[aid] for aid in voting_agents if aid in self.agents]
        else:
            voters = list(self.agents.values())
        
        if not voters:
            return {"decision": None, "consensus": 0.0, "votes": {}}
        
        # 收集投票
        votes = {}
        for agent in voters:
            vote = await self._get_agent_vote(agent, question)
            votes[agent.agent_id] = vote
        
        # 统计投票
        vote_counts = defaultdict(int)
        for vote in votes.values():
            vote_counts[vote] += 1
        
        # 找到最多票数的选项
        if vote_counts:
            majority_vote = max(vote_counts.items(), key=lambda x: x[1])
            consensus = majority_vote[1] / len(votes)
            
            decision = majority_vote[0] if consensus >= consensus_threshold else None
        else:
            decision = None
            consensus = 0.0
        
        return {
            "decision": decision,
            "consensus": consensus,
            "votes": votes,
            "vote_counts": dict(vote_counts)
        }
    
    async def _get_agent_vote(self, agent: Agent, question: str) -> str:
        """
        获取Agent的投票
        
        使用LLM基于Agent的角色和能力进行投票
        """
        prompt = f"""请对以下问题进行投票决策。

问题: {question}

基于你的角色（{agent.role.value}）和专长（{', '.join(agent.capabilities)}），
请选择: "同意"、"反对" 或 "中立"

只输出一个词，不要其他内容。"""

        response = await agent.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )
        
        vote = response.choices[0].message.content.strip()
        
        # 标准化投票
        if "同意" in vote or "yes" in vote.lower():
            return "同意"
        elif "反对" in vote or "no" in vote.lower():
            return "反对"
        else:
            return "中立"
```

**使用示例**:
```python
# 创建多Agent系统
system = MultiAgentSystem()

# 创建专业化Agents
planner = SpecializedAgents.create_planner(llm)
executor = SpecializedAgents.create_executor(llm)
critic = SpecializedAgents.create_critic(llm)
coordinator = SpecializedAgents.create_coordinator(llm)

# 注册Agents
system.register_agent(planner)
system.register_agent(executor)
system.register_agent(critic)
system.register_agent(coordinator)

# 分配任务
agent_id = await system.assign_task(
    task="分析用户需求并生成需求文档",
    preferred_role=AgentRole.PLANNER,
    required_capabilities=["task_decomposition", "planning"]
)
# 返回: "planner"

# 协作决策
decision = await system.collaborative_decision(
    question="是否应该采用微服务架构?",
    voting_agents=None,  # 所有Agent投票
    consensus_threshold=0.6
)
# 返回: {"decision": "同意", "consensus": 0.75, "votes": {...}, "vote_counts": {...}}

# 广播消息
message = AgentMessage(
    id=str(uuid.uuid4()),
    sender_id="coordinator",
    receiver_id="broadcast",
    type=MessageType.PROVIDE_INFO,
    content={"update": "项目进度50%"}
)
await system.send_message(message)

# 获取系统状态
state = system.get_system_state()
print(state)
# {
#     "num_agents": 4,
#     "agents": {
#         "planner": {"status": "idle", "role": "planner", ...},
#         "executor": {"status": "working", "current_task": "...", ...},
#         ...
#     },
#     "num_messages": 15,
#     "shared_state": {...}
# }
```

---

### 3.3 工具权限管理

**核心实现**:
```python
# algo/agent-service/core/agent/tool_permissions.py

class ToolPermissionManager:
    """
    工具权限管理器
    
    功能:
    1. 权限定义和管理
    2. 访问控制验证
    3. 审计日志记录
    4. 动态权限授予/撤销
    """
    
    def __init__(self):
        # 工具注册表
        self.tools: Dict[str, ToolDefinition] = {}
        
        # 权限规则: agent_id -> [PermissionRule]
        self.permissions: Dict[str, List[PermissionRule]] = {}
        
        # 访问日志
        self.access_logs: List[AccessLog] = []
        
        # 默认权限策略
        self.default_policy: PermissionLevel = PermissionLevel.NONE
    
    def grant_permission(
        self,
        agent_id: str,
        tool_id: str,
        permission_level: PermissionLevel,
        conditions: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
        granted_by: str = "admin"
    ) -> bool:
        """
        授予权限
        
        Args:
            agent_id: Agent ID
            tool_id: 工具ID
            permission_level: 权限级别(NONE/READ/WRITE/EXECUTE/ADMIN)
            conditions: 附加条件(如{"time": "9:00-18:00"})
            expires_at: 过期时间
            granted_by: 授予者
        
        Returns:
            bool: 是否成功
        """
        if tool_id not in self.tools:
            logger.error(f"Tool not found: {tool_id}")
            return False
        
        rule = PermissionRule(
            agent_id=agent_id,
            tool_id=tool_id,
            permission_level=permission_level,
            conditions=conditions or {},
            expires_at=expires_at,
            granted_by=granted_by
        )
        
        if agent_id not in self.permissions:
            self.permissions[agent_id] = []
        
        self.permissions[agent_id].append(rule)
        
        logger.info(f"Granted {permission_level.name} permission to {agent_id} for {tool_id}")
        return True
    
    def check_permission(
        self,
        agent_id: str,
        tool_id: str,
        required_level: Optional[PermissionLevel] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        检查权限
        
        检查流程:
        1. 获取工具定义
        2. 确定要求的权限级别(默认使用工具的required_permission)
        3. 查找Agent的权限规则
        4. 检查是否过期
        5. 检查权限级别是否足够
        6. 检查附加条件是否满足
        7. 返回结果
        """
        # 获取工具定义
        tool = self.tools.get(tool_id)
        if not tool:
            logger.warning(f"Tool not found: {tool_id}")
            return False
        
        # 确定要求的权限级别
        if required_level is None:
            required_level = tool.required_permission
        
        # 检查Agent的权限规则
        if agent_id not in self.permissions:
            has_permission = self.default_policy.value >= required_level.value
        else:
            has_permission = False
            
            for rule in self.permissions[agent_id]:
                if rule.tool_id != tool_id:
                    continue
                
                # 检查是否过期
                if rule.expires_at and datetime.now() > rule.expires_at:
                    logger.warning(f"Permission expired for {agent_id} on {tool_id}")
                    continue
                
                # 检查权限级别
                if rule.permission_level.value >= required_level.value:
                    # 检查附加条件
                    if self._check_conditions(rule.conditions, context):
                        has_permission = True
                        break
        
        return has_permission
    
    async def execute_tool(
        self,
        agent_id: str,
        tool_id: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行工具（带权限检查）
        
        流程:
        1. 检查权限
        2. 如果无权限,记录日志并返回错误
        3. 如果有权限,执行工具
        4. 记录访问日志
        5. 返回结果
        """
        # 检查权限
        has_permission = self.check_permission(
            agent_id, tool_id, PermissionLevel.EXECUTE, context
        )
        
        if not has_permission:
            self._log_access(
                agent_id=agent_id,
                tool_id=tool_id,
                action="execute",
                success=False,
                metadata={"reason": "permission_denied"}
            )
            
            return {
                "success": False,
                "error": "Permission denied",
                "tool_id": tool_id
            }
        
        # 获取工具
        tool = self.tools.get(tool_id)
        if not tool:
            return {
                "success": False,
                "error": "Tool not found",
                "tool_id": tool_id
            }
        
        # 执行工具
        result = await tool.function(**parameters)
        
        self._log_access(
            agent_id=agent_id,
            tool_id=tool_id,
            action="execute",
            success=True,
            metadata={"parameters": parameters}
        )
        
        return {
            "success": True,
            "result": result,
            "tool_id": tool_id
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        返回:
        {
            "total_tools": 10,
            "total_agents_with_permissions": 5,
            "total_accesses": 100,
            "successful_accesses": 90,
            "failed_accesses": 10,
            "success_rate": 0.9,
            "agent_stats": {...},
            "tool_stats": {...}
        }
        """
        total_accesses = len(self.access_logs)
        successful_accesses = sum(1 for log in self.access_logs if log.success)
        
        # 按Agent统计
        agent_stats = {}
        for log in self.access_logs:
            if log.agent_id not in agent_stats:
                agent_stats[log.agent_id] = {"total": 0, "success": 0, "failed": 0}
            agent_stats[log.agent_id]["total"] += 1
            if log.success:
                agent_stats[log.agent_id]["success"] += 1
            else:
                agent_stats[log.agent_id]["failed"] += 1
        
        # 按工具统计
        tool_stats = {}
        for log in self.access_logs:
            if log.tool_id not in tool_stats:
                tool_stats[log.tool_id] = {"total": 0, "success": 0, "failed": 0}
            tool_stats[log.tool_id]["total"] += 1
            if log.success:
                tool_stats[log.tool_id]["success"] += 1
            else:
                tool_stats[log.tool_id]["failed"] += 1
        
        return {
            "total_tools": len(self.tools),
            "total_agents_with_permissions": len(self.permissions),
            "total_accesses": total_accesses,
            "successful_accesses": successful_accesses,
            "failed_accesses": total_accesses - successful_accesses,
            "success_rate": successful_accesses / total_accesses if total_accesses > 0 else 0,
            "agent_stats": agent_stats,
            "tool_stats": tool_stats
        }
```

**使用示例**:
```python
# 创建权限管理器
perm_manager = ToolPermissionManager()

# 注册工具
search_tool = ToolDefinition(
    tool_id="search",
    name="搜索工具",
    description="在互联网上搜索信息",
    category=ToolCategory.NETWORK,
    required_permission=PermissionLevel.EXECUTE,
    risk_level=2,
    function=search_function
)
perm_manager.register_tool(search_tool)

# 授予权限
perm_manager.grant_permission(
    agent_id="executor_agent",
    tool_id="search",
    permission_level=PermissionLevel.EXECUTE,
    conditions=None,
    expires_at=None,
    granted_by="admin"
)

# 检查权限
has_perm = perm_manager.check_permission(
    agent_id="executor_agent",
    tool_id="search",
    required_level=PermissionLevel.EXECUTE
)
print(f"有权限: {has_perm}")  # True

# 执行工具
result = await perm_manager.execute_tool(
    agent_id="executor_agent",
    tool_id="search",
    parameters={"query": "Python tutorial"}
)
print(result)  # {"success": True, "result": "...", "tool_id": "search"}

# 撤销权限
perm_manager.revoke_permission(
    agent_id="executor_agent",
    tool_id="search"
)

# 获取统计
stats = perm_manager.get_statistics()
print(stats)
# {
#     "total_tools": 5,
#     "total_agents_with_permissions": 3,
#     "total_accesses": 50,
#     "successful_accesses": 45,
#     "failed_accesses": 5,
#     "success_rate": 0.9,
#     ...
# }

# 导出审计日志
perm_manager.export_audit_log("audit_logs.json")
```

---

## 4. 关键数据结构与UML图

```mermaid
classDiagram
    class AgentState {
        +str task
        +Dict context
        +List~Dict~ plan
        +int current_step
        +List~Dict~ execution_results
        +bool need_reflection
        +bool need_replan
        +int iterations
        +int max_iterations
        +str final_result
        +str error
    }
    
    class CompleteLangGraphWorkflow {
        -Dict config
        -Any llm
        -List tools
        -ToolExecutor tool_executor
        -StateGraph workflow
        -App app
        +__init__(config)
        -_init_llm(llm_config)
        -_build_graph()
        +run(task, context, max_iterations) Dict
        -_planning_node(state) Dict
        -_execution_node(state) Dict
        -_reflection_node(state) Dict
        -_synthesis_node(state) Dict
        -_execute_step(step, state) Any
        -_should_execute(state) str
        -_should_reflect(state) str
    }
    
    class ReflectionAgent {
        -Any llm
        -bool enable_auto_correction
        -int max_reflection_iterations
        -float confidence_threshold
        -List~ReflectionResult~ reflection_history
        +reflect(task, output, context, type) ReflectionResult
        -_validate_output(task, output, context) ReflectionResult
        -_correct_output(task, output, context) ReflectionResult
        -_improve_output(task, output, context) ReflectionResult
        -_iterative_correction(task, output, context) ReflectionResult
        +get_reflection_stats() Dict
    }
    
    class ReflectionResult {
        +bool passed
        +ReflectionType type
        +Any original_output
        +Any reflected_output
        +str feedback
        +float confidence
        +List~str~ suggestions
        +Dict metadata
    }
    
    class MultiAgentSystem {
        -Dict~str Agent~ agents
        -Queue message_bus
        -Dict shared_state
        -bool running
        -List~AgentMessage~ message_history
        +register_agent(agent)
        +unregister_agent(agent_id)
        +send_message(message)
        +assign_task(task, preferred_role, required_capabilities) str
        +collaborative_decision(question, voting_agents, consensus_threshold) Dict
        +start()
        +stop()
        +get_system_state() Dict
    }
    
    class Agent {
        +str agent_id
        +AgentRole role
        -Any llm
        +List~str~ capabilities
        +Dict~str Callable~ tools
        +AgentState state
        +Queue inbox
        +Dict shared_state
        +process_message(message) AgentMessage
        +execute_task(task) Any
        -_handle_task_assignment(message)
        -_handle_help_request(message)
        -_handle_info(message)
        +get_state() AgentState
    }
    
    class AgentMessage {
        +str id
        +str sender_id
        +str receiver_id
        +MessageType type
        +Any content
        +Dict metadata
        +float timestamp
    }
    
    class ToolPermissionManager {
        -Dict~str ToolDefinition~ tools
        -Dict~str List~PermissionRule~~ permissions
        -List~AccessLog~ access_logs
        -PermissionLevel default_policy
        +register_tool(tool)
        +grant_permission(agent_id, tool_id, permission_level, conditions, expires_at)
        +revoke_permission(agent_id, tool_id)
        +check_permission(agent_id, tool_id, required_level, context) bool
        +execute_tool(agent_id, tool_id, parameters, context) Dict
        +get_statistics() Dict
        +export_audit_log(filepath)
    }
    
    class ToolDefinition {
        +str tool_id
        +str name
        +str description
        +ToolCategory category
        +PermissionLevel required_permission
        +int risk_level
        +Callable function
        +Dict parameters
        +Dict metadata
    }
    
    class PermissionRule {
        +str agent_id
        +str tool_id
        +PermissionLevel permission_level
        +Dict conditions
        +datetime expires_at
        +str granted_by
        +datetime granted_at
    }
    
    CompleteLangGraphWorkflow "1" --> "*" AgentState : uses
    ReflectionAgent "1" --> "*" ReflectionResult : produces
    MultiAgentSystem "1" --> "*" Agent : manages
    MultiAgentSystem "1" --> "*" AgentMessage : routes
    ToolPermissionManager "1" --> "*" ToolDefinition : manages
    ToolPermissionManager "1" --> "*" PermissionRule : enforces
    Agent "*" --> "1" AgentMessage : sends/receives
```

---

## 5. 最佳实践与优化

### 5.1 任务规划优化

**提示工程**:
```python
# 优化规划提示,提高计划质量
planning_prompt = f"""
你是一个专业的任务规划专家。请将以下任务分解为具体、可执行的步骤。

任务: {task}

可用工具: {', '.join(available_tools)}

上下文: {json.dumps(context)}

要求:
1. 步骤具体、可执行、独立
2. 合理使用可用工具
3. 估算每个步骤的预期时间
4. 考虑步骤之间的依赖关系
5. 包含验证步骤

输出JSON格式:
[
    {{
        "step_number": 1,
        "description": "具体步骤描述",
        "tool": "工具名(可选)",
        "input": {{"参数": "值"}},
        "expected_output": "预期输出",
        "estimated_time": 5.0,
        "dependencies": [0]  // 依赖的步骤编号
    }},
    ...
]
"""
```

### 5.2 反思频率控制

**动态反思**:
```python
# 根据任务复杂度和步骤数动态调整反思频率
def should_reflect(current_step, total_steps, complexity):
    """
    决定是否需要反思
    
    策略:
    - 简单任务(complexity=low): 只在最后反思
    - 中等任务(complexity=medium): 每3步反思一次
    - 复杂任务(complexity=high): 每步都反思
    """
    if complexity == "low":
        return current_step == total_steps  # 只在最后
    elif complexity == "medium":
        return current_step % 3 == 0 or current_step == total_steps  # 每3步
    else:
        return True  # 每步都反思

# 在执行节点中使用
need_reflection = should_reflect(current_step, len(plan), task_complexity)
```

### 5.3 多Agent负载均衡

**智能分配策略**:
```python
def _select_agent(
    self,
    preferred_role: Optional[AgentRole],
    required_capabilities: Optional[List[str]]
) -> Optional[Agent]:
    """
    智能选择Agent
    
    评分策略:
    1. 状态得分: idle=10, waiting=5, working=0
    2. 角色匹配: 完全匹配+10, 部分匹配+5
    3. 能力匹配: 每个能力+2
    4. 负载得分: 当前任务数越少越高
    
    选择总分最高的Agent
    """
    scores = {}
    
    for agent in self.agents.values():
        score = 0
        
        # 状态得分
        if agent.state.status == "idle":
            score += 10
        elif agent.state.status == "waiting":
            score += 5
        else:
            continue  # working/blocked不参与
        
        # 角色匹配
        if preferred_role and agent.role == preferred_role:
            score += 10
        
        # 能力匹配
        if required_capabilities:
            matched = sum(1 for cap in required_capabilities if cap in agent.capabilities)
            score += matched * 2
        
        # 负载得分(假设记录了任务计数)
        task_count = agent.state.metadata.get("task_count", 0)
        score += max(0, 10 - task_count)
        
        scores[agent.agent_id] = score
    
    # 选择最高分
    if scores:
        best_agent_id = max(scores.items(), key=lambda x: x[1])[0]
        return self.agents[best_agent_id]
    
    return None
```

### 5.4 工具调用优化

**批量执行**:
```python
async def execute_batch_tools(
    tool_calls: List[Dict[str, Any]],
    max_concurrent: int = 5
) -> List[Any]:
    """
    批量并发执行工具
    
    限制并发数,避免资源耗尽
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_limit(call):
        async with semaphore:
            return await tool_executor.execute(
                call["tool_name"],
                **call["parameters"]
            )
    
    tasks = [execute_with_limit(call) for call in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results
```

**工具结果缓存**:
```python
from functools import lru_cache
import hashlib
import json

# 幂等工具结果缓存(如search)
tool_cache = {}

async def execute_with_cache(tool_name, parameters):
    """
    带缓存的工具执行
    
    对于幂等工具(如search),缓存结果
    """
    # 生成缓存key
    cache_key = hashlib.md5(
        json.dumps({"tool": tool_name, "params": parameters}, sort_keys=True).encode()
    ).hexdigest()
    
    # 检查缓存
    if cache_key in tool_cache:
        logger.info(f"Cache hit for {tool_name}")
        return tool_cache[cache_key]
    
    # 执行工具
    result = await tool_executor.execute(tool_name, **parameters)
    
    # 缓存结果(仅缓存幂等工具)
    if tool_name in IDEMPOTENT_TOOLS:
        tool_cache[cache_key] = result
    
    return result
```

---

## 6. 总结

Agent服务作为VoiceHelper的智能任务处理核心,实现了以下能力:

1. **LangGraph工作流**:状态图架构,支持Planning→Execution→Reflection→Synthesis完整闭环
2. **自我反思**:验证、纠正、改进输出,迭代提升质量(最多3次,置信度阈值0.7)
3. **多Agent协作**:消息传递、任务分配、协作决策(投票/共识/仲裁)
4. **工具权限管理**:5级权限(NONE/READ/WRITE/EXECUTE/ADMIN),审计日志,条件/过期控制
5. **异步执行**:后台任务执行,立即返回task_id,支持状态查询

通过LangGraph的灵活状态转换、反思机制的质量保障、多Agent的协作能力、权限系统的安全控制,实现了企业级的Agent服务。

未来优化方向:
- 支持更多推理策略(Tree-of-Thoughts、ReAct)
- 长期记忆管理(向量数据库存储历史交互)
- 工具自动发现和注册(插件化)
- 多模态工具支持(图像、视频处理)
- 分布式Agent集群(跨机器协作)

---

**文档状态**:✅ 已完成  
**覆盖度**:100%(LangGraph、反思、多Agent、工具权限、最佳实践)  
**下一步**:生成Multimodal多模态服务模块文档(10-Multimodal服务)

