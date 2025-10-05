---
title: "Open-Assistant-04-Website-概览"
date: 2025-10-05T10:45:52+08:00
draft: false
tags:
  - 架构设计
  - 概览
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - Open-Assistant-04-Website-概览"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# Open-Assistant-04-Website-概览

## 1. 模块职责

Website 是 Open-Assistant 的 Web 前端应用，提供用户交互界面。

**主要职责：**
- **用户管理**：注册、登录、个人资料、统计数据
- **任务系统**：展示任务、提交响应、评分排名
- **聊天界面**：与 Inference 服务交互，进行对话
- **排行榜**：展示贡献者排名和成就
- **管理后台**：用户管理、内容审核、系统配置
- **多语言支持**：i18n 国际化，支持 30+ 语言

**输入：**
- 用户交互（点击、输入、导航）
- Backend API 响应（任务、消息、用户数据）
- Inference API 响应（流式对话 Token）

**输出：**
- 渲染的 HTML 页面（SSR 或 CSR）
- API 请求（调用 Backend/Inference）
- 用户反馈（提示、错误信息）

## 2. 技术栈

### 2.1 核心框架

| 技术 | 版本 | 作用 |
|---|---|---|
| Next.js | 13+ | React 全栈框架，支持 SSR/SSG/ISR |
| React | 18+ | UI 组件库 |
| TypeScript | 5+ | 类型安全的 JavaScript |
| Prisma | 5+ | ORM，管理 PostgreSQL 数据库 |
| NextAuth.js | 4+ | 用户认证与会话管理 |

### 2.2 UI 库与样式

| 技术 | 作用 |
|---|---|
| Chakra-UI | 预构建的 UI 组件（Button、Modal、Toast 等） |
| TailwindCSS | 实用优先的 CSS 框架 |
| Framer Motion | 动画库 |
| React Icons | 图标库 |

### 2.3 状态管理与数据获取

| 技术 | 作用 |
|---|---|
| SWR | 数据获取与缓存（Stale-While-Revalidate） |
| React Query | 异步状态管理 |
| React Context | 全局状态（用户、主题、语言） |

### 2.4 测试与质量

| 技术 | 作用 |
|---|---|
| Cypress | E2E 测试与组件测试 |
| Jest | 单元测试 |
| React Testing Library | React 组件测试工具 |
| ESLint | 代码检查 |
| Prettier | 代码格式化 |

## 3. 架构设计

### 3.1 整体架构图

```mermaid
flowchart TB
    subgraph Browser["浏览器"]
        UI[React 组件<br/>Chakra-UI]
        State[状态管理<br/>SWR/Context]
    end
    
    subgraph Next["Next.js 服务器"]
        Pages[Pages<br/>SSR/CSR]
        API[API Routes<br/>BFF 层]
        Auth[NextAuth.js<br/>认证中间件]
        Prisma[Prisma Client<br/>ORM]
    end
    
    subgraph Storage["存储"]
        WebDB[(Web DB<br/>PostgreSQL)]
        Session[(Session Store<br/>Redis/DB)]
    end
    
    subgraph Backend["Backend API"]
        TaskAPI[Tasks API]
        MessageAPI[Messages API]
        UserAPI[Users API]
    end
    
    subgraph Inference["Inference API"]
        ChatAPI[Chats API]
        StreamAPI[SSE Stream]
    end
    
    UI --> State
    State <--> Pages
    Pages --> API
    API --> Auth
    Auth --> Session
    
    API --> Prisma
    Prisma --> WebDB
    
    API --> TaskAPI
    API --> MessageAPI
    API --> UserAPI
    
    API --> ChatAPI
    API --> StreamAPI
```

### 3.2 目录结构

```
website/src/
├── pages/                    # Next.js 页面与 API Routes
│   ├── _app.tsx             # 应用入口，全局配置
│   ├── _document.tsx        # HTML 文档结构
│   ├── index.tsx            # 首页
│   ├── dashboard.tsx        # 用户仪表板
│   ├── chat/                # 聊天相关页面
│   ├── create/              # 创建响应页面
│   ├── evaluate/            # 评估排名页面
│   ├── account/             # 账户管理
│   ├── admin/               # 管理后台
│   └── api/                 # API Routes（BFF）
│       ├── auth/            # 认证相关 API
│       ├── tasks/           # 任务相关 API
│       ├── messages/        # 消息相关 API
│       └── chat/            # 聊天相关 API
│
├── components/              # React 组件
│   ├── Chat/                # 聊天组件
│   ├── Tasks/               # 任务组件
│   ├── Messages/            # 消息组件
│   ├── Dashboard/           # 仪表板组件
│   ├── Header/              # 页头组件
│   └── ...                  # 其他公共组件
│
├── lib/                     # 工具库
│   ├── oasst_api_client.ts  # Backend API 客户端
│   ├── oasst_inference_client.ts  # Inference API 客户端
│   ├── auth.ts              # 认证逻辑
│   ├── prismadb.ts          # Prisma 客户端单例
│   └── ...
│
├── types/                   # TypeScript 类型定义
├── hooks/                   # 自定义 React Hooks
├── styles/                  # 样式文件
└── utils/                   # 工具函数
```

## 4. 核心页面与功能

### 4.1 首页与导航

**页面：** `pages/index.tsx`

**功能：**
- 项目介绍与愿景
- 注册/登录入口
- 快速开始引导

### 4.2 用户仪表板

**页面：** `pages/dashboard.tsx`

**功能：**
- 显示可用任务卡片（创建初始提示、回复助手、排名响应等）
- 用户统计（积分、排名、连续天数）
- 最近活动

**关键组件：**

```tsx
// components/Dashboard/TaskOption.tsx
function TaskOption({ task, onTaskClick }) {
  return (
    <Card onClick={() => onTaskClick(task.type)}>
      <Icon as={task.icon} />
      <Text>{task.title}</Text>
      <Text>{task.description}</Text>
    </Card>
  );
}
```

### 4.3 任务页面

**创建响应：** `pages/create/`

| 任务类型 | 页面文件 | 说明 |
|---|---|---|
| 创建初始提示 | `initial_prompt.tsx` | 用户输入对话开场白 |
| 回复助手 | `assistant_reply.tsx` | 对用户消息生成助手回复 |
| 回复用户 | `user_reply.tsx` | 对助手消息生成用户回复 |

**评估与排名：** `pages/evaluate/`

| 任务类型 | 页面文件 | 说明 |
|---|---|---|
| 排名初始提示 | `rank_initial_prompts.tsx` | 对多个提示进行排序 |
| 排名助手回复 | `rank_assistant_replies.tsx` | 对多个助手回复进行排序 |
| 标注回复 | `label_assistant_reply.tsx` | 对回复打标签（有用、有害、幽默等） |

**任务流程：**

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant P as Task Page
    participant API as /api/tasks
    participant B as Backend
    
    U->>P: 访问任务页面
    P->>API: GET /api/tasks/available
    API->>B: GET /api/v1/tasks/available
    B-->>API: {task: {...}}
    API-->>P: {task: {...}}
    P-->>U: 显示任务内容
    
    U->>P: 提交响应
    P->>API: POST /api/tasks/interaction<br/>{type, content}
    API->>B: POST /api/v1/tasks/{task_id}/interaction
    B-->>API: 200 OK
    API-->>P: {success: true}
    P-->>U: 显示成功提示<br/>+XP 奖励
```

### 4.4 聊天界面

**页面：** `pages/chat/[id].tsx`

**功能：**
- 与 Inference 服务实时对话
- 流式显示 AI 回复
- 多轮对话历史
- 配置生成参数（temperature、top_p、max_tokens）
- 插件选择（web_retriever 等）

**关键组件：**

```tsx
// components/Chat/ChatConversation.tsx
function ChatConversation({ chatId }) {
  const { messages, sendMessage, isStreaming } = useChatStream(chatId);
  
  return (
    <VStack>
      {messages.map(msg => (
        <ChatMessageEntry key={msg.id} message={msg} />
      ))}
      {isStreaming && <LoadingIndicator />}
      <ChatForm onSubmit={sendMessage} disabled={isStreaming} />
    </VStack>
  );
}
```

**流式响应处理：**

```typescript
// lib/chat_stream.ts
export async function* streamChatResponse(chatId: string, message: string) {
  const response = await fetch(`/api/chat/${chatId}/message`, {
    method: 'POST',
    body: JSON.stringify({ content: message }),
    headers: { 'Content-Type': 'application/json' }
  });
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        
        if (data.event_type === 'token') {
          yield { type: 'token', text: data.text };
        } else if (data.event_type === 'message') {
          yield { type: 'complete', message: data.message };
        }
      }
    }
  }
}
```

### 4.5 排行榜

**页面：** `pages/leaderboard.tsx`

**功能：**
- 显示贡献者排名（按总积分、每日积分、每周积分）
- 用户统计（提示数、回复数、排名数、标注数）
- 分页加载

**数据获取：**

```tsx
// hooks/useleaderboard.ts
function useLeaderboard(time_frame: 'total' | 'day' | 'week') {
  return useSWR(
    `/api/leaderboard?time_frame=${time_frame}`,
    fetcher,
    { refreshInterval: 60000 }  // 每分钟刷新
  );
}
```

### 4.6 管理后台

**页面：** `pages/admin/`

| 页面 | 功能 |
|---|---|
| `index.tsx` | 管理概览 |
| `users.tsx` | 用户管理（查看、封禁、权限） |
| `messages.tsx` | 消息管理（查看、删除、标记） |
| `status.tsx` | 系统状态（队列、统计、健康检查） |
| `trollboard.tsx` | 低质量贡献者列表 |

**权限控制：**

```typescript
// lib/auth.ts
export async function requireAdmin(req, res) {
  const session = await getSession({ req });
  
  if (!session || !session.user.isAdmin) {
    return res.status(403).json({ error: 'Forbidden' });
  }
  
  return session;
}
```

## 5. API Routes（BFF 层）

Website 使用 Next.js API Routes 作为 Backend For Frontend（BFF）层，封装 Backend 和 Inference API。

### 5.1 认证 API

**文件：** `pages/api/auth/[...nextauth].ts`

**功能：**
- Discord OAuth
- Email Magic Link
- Debug Credentials（开发用）
- 会话管理

**配置：**

```typescript
export default NextAuth({
  providers: [
    DiscordProvider({
      clientId: process.env.DISCORD_CLIENT_ID,
      clientSecret: process.env.DISCORD_CLIENT_SECRET,
    }),
    EmailProvider({
      server: process.env.EMAIL_SERVER,
      from: process.env.EMAIL_FROM,
    }),
    // 开发环境：无密码登录
    ...(process.env.DEBUG_LOGIN === 'true' ? [CredentialsProvider({
      name: 'Debug',
      credentials: { username: { label: "Username", type: "text" } },
      authorize: async (credentials) => ({ id: credentials.username, name: credentials.username })
    })] : [])
  ],
  callbacks: {
    async session({ session, token }) {
      // 从 Backend 获取用户信息
      const userData = await backendFetch(`/api/v1/frontend_users/${token.sub}`);
      session.user.isAdmin = userData.is_admin;
      session.user.score = userData.score;
      return session;
    }
  }
});
```

### 5.2 任务 API

**获取可用任务：** `GET /api/tasks/available`

```typescript
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const session = await getSession({ req });
  if (!session) return res.status(401).json({ error: 'Unauthorized' });
  
  // 调用 Backend API
  const task = await oasstApiClient.fetchTask({
    type: req.query.type as string,
    lang: req.query.lang as string,
    user_id: session.user.id,
  });
  
  res.status(200).json(task);
}
```

**提交任务交互：** `POST /api/tasks/interaction`

```typescript
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const session = await getSession({ req });
  if (!session) return res.status(401).json({ error: 'Unauthorized' });
  
  const { task_id, update_type, content } = req.body;
  
  // 调用 Backend API
  await oasstApiClient.postInteraction({
    type: update_type,
    task_id,
    payload: content,
    user_id: session.user.id,
  });
  
  res.status(200).json({ success: true });
}
```

### 5.3 聊天 API

**创建消息（流式）：** `POST /api/chat/[id]/message`

```typescript
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const session = await getSession({ req });
  if (!session) return res.status(401).json({ error: 'Unauthorized' });
  
  const { id: chatId } = req.query;
  const { content, parent_id } = req.body;
  
  // 设置 SSE 响应头
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  
  try {
    // 调用 Inference API（流式）
    const stream = inferenceClient.postAssistantMessage({
      chat_id: chatId,
      parent_id,
      model_config_name: 'llama-30b',
      // 省略：其他参数
    });
    
    for await (const event of stream) {
      // 转发 Token 给客户端
      res.write(`data: ${JSON.stringify(event)}\n\n`);
    }
    
    res.end();
  } catch (error) {
    res.write(`data: ${JSON.stringify({ event_type: 'error', message: error.message })}\n\n`);
    res.end();
  }
}
```

## 6. 数据模型（Web DB）

Website 使用独立的 PostgreSQL 数据库存储会话和本地状态。

**Prisma Schema：** `prisma/schema.prisma`

```prisma
model Account {
  id                String  @id @default(cuid())
  userId            String
  type              String
  provider          String
  providerAccountId String
  refresh_token     String?
  access_token      String?
  expires_at        Int?
  token_type        String?
  scope             String?
  id_token          String?
  session_state     String?
  
  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  @@unique([provider, providerAccountId])
}

model Session {
  id           String   @id @default(cuid())
  sessionToken String   @unique
  userId       String
  expires      DateTime
  user         User     @relation(fields: [userId], references: [id], onDelete: Cascade)
}

model User {
  id            String    @id @default(cuid())
  name          String?
  email         String?   @unique
  emailVerified DateTime?
  image         String?
  accounts      Account[]
  sessions      Session[]
}

// 其他模型：VerificationToken 等
```

## 7. 状态管理

### 7.1 全局状态（Context）

**认证状态：** NextAuth Session

```tsx
// 在任意组件中使用
import { useSession } from 'next-auth/react';

function MyComponent() {
  const { data: session, status } = useSession();
  
  if (status === 'loading') return <Loading />;
  if (status === 'unauthenticated') return <LoginPrompt />;
  
  return <div>Welcome, {session.user.name}!</div>;
}
```

**主题与语言：** React Context

```tsx
// context/ThemeContext.tsx
export const ThemeContext = createContext<{
  colorMode: 'light' | 'dark';
  toggleColorMode: () => void;
}>(null);

// 在 _app.tsx 中提供
<ThemeContext.Provider value={themeValue}>
  <Component {...pageProps} />
</ThemeContext.Provider>
```

### 7.2 数据获取（SWR）

**优势：**
- 自动缓存与重新验证
- 乐观更新
- 错误重试
- 焦点重新获取

**示例：**

```tsx
function TaskList() {
  const { data: tasks, error, mutate } = useSWR('/api/tasks/my-tasks', fetcher);
  
  if (error) return <ErrorMessage />;
  if (!tasks) return <Loading />;
  
  return (
    <VStack>
      {tasks.map(task => (
        <TaskCard key={task.id} task={task} onComplete={() => mutate()} />
      ))}
    </VStack>
  );
}
```

## 8. 国际化（i18n）

Website 支持 30+ 语言，使用 `next-i18next`。

**配置：** `next-i18next.config.js`

```javascript
module.exports = {
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh', 'es', 'fr', 'de', 'ja', ...],  // 30+ 语言
  },
  localePath: path.resolve('./public/locales'),
};
```

**翻译文件：** `public/locales/{lang}/common.json`

```json
{
  "dashboard": "Dashboard",
  "leaderboard": "Leaderboard",
  "tasks": {
    "initial_prompt": "Create Initial Prompt",
    "assistant_reply": "Reply as Assistant"
  }
}
```

**使用：**

```tsx
import { useTranslation } from 'next-i18next';

function MyComponent() {
  const { t } = useTranslation();
  
  return <Text>{t('tasks.initial_prompt')}</Text>;
}
```

## 9. 性能优化

### 9.1 SSR/SSG/ISR

**静态生成（SSG）：** 首页、关于页面

```tsx
// pages/about.tsx
export async function getStaticProps() {
  return {
    props: { teamMembers: [...] },
    revalidate: 3600,  // ISR：每小时重新生成
  };
}
```

**服务端渲染（SSR）：** 需要认证的页面

```tsx
// pages/dashboard.tsx
export async function getServerSideProps(context) {
  const session = await getSession(context);
  if (!session) {
    return { redirect: { destination: '/auth/signin' } };
  }
  return { props: { session } };
}
```

### 9.2 代码分割

**动态导入：**

```tsx
import dynamic from 'next/dynamic';

const HeavyComponent = dynamic(() => import('../components/HeavyComponent'), {
  loading: () => <Loading />,
  ssr: false,  // 仅客户端加载
});
```

### 9.3 图片优化

**Next.js Image 组件：**

```tsx
import Image from 'next/image';

<Image
  src="/logo.png"
  alt="Logo"
  width={200}
  height={50}
  loading="lazy"  // 懒加载
  placeholder="blur"  // 模糊占位符
/>
```

## 10. 部署

### 10.1 生产构建

```bash
# 安装依赖
npm ci

# 构建
npm run build

# 启动
npm run start
```

### 10.2 Docker 部署

**Dockerfile：** `docker/Dockerfile.website`

```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npx prisma generate
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

CMD ["npm", "start"]
```

### 10.3 环境变量

```bash
# 数据库
DATABASE_URL=postgresql://user:pass@localhost:5432/oasst_web

# 认证
NEXTAUTH_URL=https://open-assistant.io
NEXTAUTH_SECRET=your-secret-key

# OAuth
DISCORD_CLIENT_ID=your-discord-client-id
DISCORD_CLIENT_SECRET=your-discord-client-secret

# Backend API
BACKEND_URL=https://backend.open-assistant.io
INFERENCE_SERVER_HOST=https://inference.open-assistant.io

# 功能开关
DEBUG_LOGIN=false  # 生产环境关闭
```

## 11. 常见问题

### Q1: 如何调试 SSR 问题？

**答：** 检查服务器端日志和客户端控制台。

- 服务器端：`console.log` 输出到终端
- 客户端：浏览器控制台
- Hydration 错误：确保 SSR 和 CSR 渲染结果一致

### Q2: 如何处理认证状态同步？

**答：** NextAuth 自动管理会话。

```tsx
// 强制刷新会话
import { useSession } from 'next-auth/react';
const { data: session, update } = useSession();

// 更新会话数据
await update({ ...session, user: { ...session.user, score: newScore } });
```

### Q3: 如何优化 SWR 缓存？

**答：** 配置全局或局部缓存策略。

```tsx
// 全局配置（_app.tsx）
<SWRConfig value={{
  refreshInterval: 30000,  // 30 秒自动刷新
  dedupingInterval: 2000,  // 2 秒内去重请求
  revalidateOnFocus: true,  // 焦点时重新验证
}}>
  <Component {...pageProps} />
</SWRConfig>
```

---

**下一步：**
- 阅读 `website/README.md` 了解开发环境配置
- 查看 `pages/dashboard.tsx` 理解任务系统
- 参考 `components/Chat/` 了解聊天实现

