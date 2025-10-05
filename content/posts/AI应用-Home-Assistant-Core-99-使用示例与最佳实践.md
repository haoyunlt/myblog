---
title: "Home Assistant Core - 使用示例与最佳实践"
date: 2025-10-05T10:45:52+08:00
draft: false
tags:
  - 最佳实践
  - 实战经验
  - 源码分析
categories:
  - 技术文档
description: "源码剖析 - Home Assistant Core - 使用示例与最佳实践"
author: "源码分析"
weight: 500
ShowToc: true
TocOpen: true
---

# Home Assistant Core - 使用示例与最佳实践

## 框架使用示例

### 示例 1：创建自定义集成

本示例演示如何创建一个简单的自定义集成，包括状态更新、事件监听和服务注册。

#### 目录结构

```
custom_components/
└── my_integration/
    ├── __init__.py
    ├── manifest.json
    ├── config_flow.py
    └── sensor.py
```

#### manifest.json

```json
{
  "domain": "my_integration",
  "name": "My Custom Integration",
  "codeowners": ["@username"],
  "config_flow": true,
  "dependencies": [],
  "documentation": "https://example.com",
  "iot_class": "local_polling",
  "requirements": [],
  "version": "1.0.0"
}
```

#### __init__.py - 集成初始化

```python
"""My Custom Integration"""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.const import Platform

PLATFORMS: list[Platform] = [Platform.SENSOR]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """设置配置条目"""
    # 1. 存储集成数据
    hass.data.setdefault(entry.domain, {})
    hass.data[entry.domain][entry.entry_id] = {
        "name": entry.data["name"],
        "update_interval": entry.data.get("update_interval", 60)
    }
    
    # 2. 转发到平台设置
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    # 3. 注册服务
    async def handle_custom_service(call):
        """处理自定义服务调用"""
        entity_id = call.data.get("entity_id")
        # 执行服务逻辑
        
    hass.services.async_register(
        entry.domain,
        "custom_action",
        handle_custom_service
    )
    
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """卸载配置条目"""
    # 1. 卸载平台
    unload_ok = await hass.config_entries.async_unload_platforms(
        entry, PLATFORMS
    )
    
    if unload_ok:
        # 2. 清理数据
        hass.data[entry.domain].pop(entry.entry_id)
    
    return unload_ok
```

#### sensor.py - 传感器平台

```python
"""Sensor platform for my_integration"""
from __future__ import annotations

from datetime import timedelta
import logging

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """设置传感器平台"""
    # 1. 获取集成配置
    config = hass.data[entry.domain][entry.entry_id]
    
    # 2. 创建数据更新协调器
    async def async_update_data():
        """获取最新数据"""
        # 实现数据获取逻辑
        return {"temperature": 25.0, "humidity": 60.0}
    
    coordinator = DataUpdateCoordinator(
        hass,
        _LOGGER,
        name="my_integration",
        update_method=async_update_data,
        update_interval=timedelta(seconds=config["update_interval"]),
    )
    
    # 3. 首次数据获取
    await coordinator.async_config_entry_first_refresh()
    
    # 4. 创建实体
    async_add_entities([
        MyTemperatureSensor(coordinator, entry),
        MyHumiditySensor(coordinator, entry),
    ])

class MyTemperatureSensor(CoordinatorEntity, SensorEntity):
    """温度传感器实体"""
    
    def __init__(self, coordinator, entry):
        """初始化传感器"""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_name = f"{entry.data['name']} Temperature"
        self._attr_unique_id = f"{entry.entry_id}_temperature"
        self._attr_native_unit_of_measurement = "°C"
        self._attr_device_class = "temperature"
    
    @property
    def native_value(self):
        """返回传感器值"""
        return self.coordinator.data.get("temperature")
    
    @property
    def available(self) -> bool:
        """返回实体是否可用"""
        return self.coordinator.last_update_success

class MyHumiditySensor(CoordinatorEntity, SensorEntity):
    """湿度传感器实体"""
    
    def __init__(self, coordinator, entry):
        """初始化传感器"""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_name = f"{entry.data['name']} Humidity"
        self._attr_unique_id = f"{entry.entry_id}_humidity"
        self._attr_native_unit_of_measurement = "%"
        self._attr_device_class = "humidity"
    
    @property
    def native_value(self):
        """返回传感器值"""
        return self.coordinator.data.get("humidity")
```

#### config_flow.py - 配置流

```python
"""Config flow for my_integration"""
from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult

STEP_USER_DATA_SCHEMA = vol.Schema({
    vol.Required("name"): str,
    vol.Optional("update_interval", default=60): vol.All(
        vol.Coerce(int),
        vol.Range(min=10, max=3600)
    ),
})

class MyIntegrationConfigFlow(config_entries.ConfigFlow, domain="my_integration"):
    """处理配置流"""
    
    VERSION = 1
    
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """处理用户输入步骤"""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # 验证输入
            await self.async_set_unique_id(user_input["name"])
            self._abort_if_unique_id_configured()
            
            # 创建配置条目
            return self.async_create_entry(
                title=user_input["name"],
                data=user_input,
            )
        
        # 显示配置表单
        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )
```

#### 使用说明

1. 将集成代码放入 `custom_components/my_integration/` 目录
2. 重启 Home Assistant
3. 在 UI 界面添加集成：设置 → 设备与服务 → 添加集成
4. 搜索 "My Custom Integration" 并配置
5. 配置完成后会创建两个传感器实体

### 示例 2：监听事件并执行自动化

本示例演示如何监听状态变更事件，并根据条件执行自定义逻辑。

```python
"""事件监听示例"""
from homeassistant.core import HomeAssistant, Event, callback
from homeassistant.const import EVENT_STATE_CHANGED

async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """设置组件"""
    
    @callback
    def handle_state_change(event: Event) -> None:
        """处理状态变更事件"""
        # 1. 获取事件数据
        entity_id = event.data.get("entity_id")
        old_state = event.data.get("old_state")
        new_state = event.data.get("new_state")
        
        # 2. 过滤目标实体
        if not entity_id.startswith("sensor.temperature_"):
            return
        
        # 3. 检查状态变化
        if old_state is None or new_state is None:
            return
        
        try:
            old_temp = float(old_state.state)
            new_temp = float(new_state.state)
        except (ValueError, TypeError):
            return
        
        # 4. 执行自定义逻辑
        if new_temp > 30.0 and old_temp <= 30.0:
            # 温度超过阈值，发送通知
            hass.services.call(
                "notify",
                "persistent_notification",
                {
                    "title": "温度警告",
                    "message": f"{entity_id} 温度过高：{new_temp}°C",
                },
            )
    
    # 注册监听器
    hass.bus.async_listen(EVENT_STATE_CHANGED, handle_state_change)
    
    return True
```

### 示例 3：创建自定义服务

本示例演示如何注册自定义服务，支持参数验证和响应数据。

```python
"""自定义服务示例"""
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse
from homeassistant.helpers import config_validation as cv
import voluptuous as vol

# 服务参数 schema
SERVICE_SCHEMA = vol.Schema({
    vol.Required("entity_id"): cv.entity_ids,
    vol.Optional("value"): cv.positive_int,
    vol.Optional("mode", default="normal"): vol.In(["normal", "fast", "slow"]),
})

async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """设置组件"""
    
    async def handle_custom_service(call: ServiceCall) -> ServiceResponse:
        """处理自定义服务调用"""
        # 1. 获取参数（已验证）
        entity_ids = call.data["entity_id"]
        value = call.data.get("value", 100)
        mode = call.data["mode"]
        
        # 2. 执行业务逻辑
        results = {}
        for entity_id in entity_ids:
            # 获取实体状态
            state = hass.states.get(entity_id)
            if state is None:
                continue
            
            # 执行操作
            # （此处省略具体业务逻辑）
            results[entity_id] = {
                "success": True,
                "value": value,
                "mode": mode,
            }
        
        # 3. 返回响应数据
        return {"results": results}
    
    # 注册服务
    hass.services.async_register(
        "my_domain",
        "custom_service",
        handle_custom_service,
        schema=SERVICE_SCHEMA,
        supports_response="optional",  # 支持返回响应数据
    )
    
    return True
```

调用示例：

```yaml
# 在自动化中调用
action:
  - service: my_domain.custom_service
    data:
      entity_id:
        - sensor.temperature_1
        - sensor.temperature_2
      value: 50
      mode: fast
    response_variable: service_result

  - service: notify.persistent_notification
    data:
      message: "服务调用结果：{{ service_result }}"
```

## 实战经验与最佳实践

### 最佳实践 1：使用 DataUpdateCoordinator 管理数据更新

**问题**：直接在实体中实现数据更新逻辑会导致多个实体重复请求数据，增加网络负载和设备压力。

**解决方案**：使用 `DataUpdateCoordinator` 统一管理数据更新，所有实体共享相同的数据源。

```python
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from datetime import timedelta

# 创建协调器
coordinator = DataUpdateCoordinator(
    hass,
    logger,
    name="my_integration",
    update_method=async_fetch_data,
    update_interval=timedelta(seconds=30),
)

# 实体继承 CoordinatorEntity
class MySensor(CoordinatorEntity, SensorEntity):
    """传感器实体"""
    
    def __init__(self, coordinator):
        super().__init__(coordinator)
    
    @property
    def native_value(self):
        # 从协调器获取数据
        return self.coordinator.data.get("value")
```

**优势**：
- 减少网络请求次数
- 统一错误处理
- 自动管理更新间隔
- 支持失败重试

### 最佳实践 2：使用 HassJob 优化回调性能

**问题**：频繁创建任务和检查函数类型会增加性能开销。

**解决方案**：使用 `HassJob` 包装回调函数，缓存函数类型信息。

```python
from homeassistant.core import HassJob, callback

# 定义回调函数
@callback
def my_listener(event):
    """事件监听器"""
    # 处理事件

# 创建 HassJob
job = HassJob(my_listener)

# 使用 HassJob 注册监听器
hass.bus.async_listen("my_event", job)
```

**优势**：
- 减少类型检查开销
- 提高事件分发性能
- 支持优先级设置

### 最佳实践 3：正确处理异步操作

**问题**：在同步代码中调用异步函数，或在异步函数中执行阻塞操作。

**解决方案**：严格区分同步和异步代码，使用正确的调用方式。

```python
# ❌ 错误：在同步代码中直接调用异步函数
def sync_function(hass):
    result = hass.async_call_service(...)  # 错误！

# ✅ 正确：使用 async_create_task
def sync_function(hass):
    hass.async_create_task(
        hass.services.async_call("domain", "service", {})
    )

# ❌ 错误：在异步函数中执行阻塞操作
async def async_function():
    import time
    time.sleep(10)  # 阻塞事件循环！

# ✅ 正确：使用 async_add_executor_job
async def async_function(hass):
    await hass.async_add_executor_job(blocking_operation)
```

### 最佳实践 4：使用 Context 追踪操作链路

**问题**：难以追踪自动化触发链路和操作来源。

**解决方案**：正确传递和使用 Context 对象。

```python
from homeassistant.core import Context

# 创建带用户信息的 context
context = Context(user_id=user.id)

# 在服务调用中传递 context
await hass.services.async_call(
    "light",
    "turn_on",
    {"entity_id": "light.living_room"},
    context=context,
)

# 在事件监听器中检查 context
@callback
def handle_state_change(event):
    context = event.context
    if context.user_id:
        # 用户触发的操作
        pass
    else:
        # 系统或自动化触发的操作
        pass
```

### 最佳实践 5：实现优雅的组件卸载

**问题**：组件卸载时未清理资源，导致内存泄漏或连接残留。

**解决方案**：实现完整的 `async_unload_entry` 方法。

```python
async def async_setup_entry(hass, entry):
    """设置配置条目"""
    # 存储需要清理的资源
    entry.async_on_unload(remove_listener)
    entry.async_on_unload(connection.close)
    
    return True

async def async_unload_entry(hass, entry):
    """卸载配置条目"""
    # 1. 卸载平台
    unload_ok = await hass.config_entries.async_unload_platforms(
        entry, PLATFORMS
    )
    
    if unload_ok:
        # 2. 清理数据
        hass.data[DOMAIN].pop(entry.entry_id)
        
        # 3. 注册的清理函数会自动执行
    
    return unload_ok
```

### 最佳实践 6：使用类型提示提高代码质量

**问题**：缺少类型提示导致 IDE 无法提供准确的代码补全，容易引入类型错误。

**解决方案**：为所有函数添加完整的类型提示。

```python
from __future__ import annotations

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddEntitiesCallback

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """设置传感器平台"""
    # 类型提示帮助 IDE 提供准确的代码补全
    entities: list[SensorEntity] = [
        MySensor(entry),
    ]
    async_add_entities(entities)
```

### 最佳实践 7：使用 asyncio.Lock 保护共享资源

**问题**：多个协程并发访问共享资源导致竞态条件。

**解决方案**：使用 `asyncio.Lock` 保护临界区。

```python
import asyncio

class MyIntegration:
    """集成类"""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._shared_data = {}
    
    async def async_update_data(self, key: str, value: Any) -> None:
        """更新共享数据"""
        async with self._lock:
            # 临界区：安全更新共享数据
            self._shared_data[key] = value
            await self._async_process_data()
    
    async def async_get_data(self, key: str) -> Any:
        """获取共享数据"""
        async with self._lock:
            # 临界区：安全读取共享数据
            return self._shared_data.get(key)
```

### 最佳实践 8：使用 ConfigEntry 存储配置

**问题**：配置存储在 `hass.data` 中，重启后丢失。

**解决方案**：使用 Config Flow 和 ConfigEntry 持久化配置。

```python
# config_flow.py
class MyConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """配置流"""
    
    async def async_step_user(self, user_input=None):
        """用户配置步骤"""
        if user_input is not None:
            # 保存配置到 ConfigEntry（自动持久化）
            return self.async_create_entry(
                title=user_input["name"],
                data=user_input,
            )
        
        return self.async_show_form(
            step_id="user",
            data_schema=CONFIG_SCHEMA,
        )

# __init__.py
async def async_setup_entry(hass, entry):
    """从 ConfigEntry 读取配置"""
    name = entry.data["name"]
    api_key = entry.data["api_key"]
    
    # 使用配置初始化集成
    client = MyClient(api_key)
    hass.data[DOMAIN][entry.entry_id] = client
    
    return True
```

## 常见问题与解决方案

### 问题 1：事件循环阻塞

**症状**：UI 响应缓慢，日志出现 "Detected blocking call" 警告。

**原因**：在事件循环中执行了阻塞操作（如网络请求、文件 I/O、sleep）。

**解决方案**：
```python
# ❌ 错误
async def async_update(self):
    import time
    time.sleep(5)  # 阻塞事件循环

# ✅ 正确
async def async_update(self):
    await asyncio.sleep(5)  # 非阻塞延迟

# ✅ 正确：将阻塞操作移到执行器
async def async_update(self):
    await self.hass.async_add_executor_job(blocking_network_call)
```

### 问题 2：内存泄漏

**症状**：系统运行一段时间后内存持续增长。

**原因**：事件监听器未正确移除，或实体未正确清理。

**解决方案**：
```python
# ✅ 正确：保存移除函数并在卸载时调用
async def async_setup_entry(hass, entry):
    # 注册监听器并保存移除函数
    remove_listener = hass.bus.async_listen("my_event", handler)
    
    # 注册清理函数
    entry.async_on_unload(remove_listener)
    
    return True

async def async_unload_entry(hass, entry):
    # 清理函数会自动执行
    return True
```

### 问题 3：状态更新不及时

**症状**：传感器值变化，但 UI 显示延迟。

**原因**：未调用 `async_write_ha_state()` 或使用了过长的更新间隔。

**解决方案**：
```python
class MySensor(SensorEntity):
    """传感器实体"""
    
    async def async_update(self):
        """更新传感器值"""
        self._attr_native_value = await self._fetch_value()
        
        # 立即写入状态
        self.async_write_ha_state()
```

### 问题 4：服务调用超时

**症状**：服务调用返回超时错误。

**原因**：服务处理器执行时间过长。

**解决方案**：
```python
async def handle_service(call: ServiceCall):
    """处理服务调用"""
    # 立即返回，后台处理
    hass.async_create_background_task(
        process_service_call(call),
        "process_service_call"
    )

async def process_service_call(call: ServiceCall):
    """后台处理服务调用"""
    # 执行耗时操作
    await long_running_operation()
```

## 性能优化建议

### 优化 1：减少状态更新频率

```python
from homeassistant.helpers.event import async_track_time_interval
from datetime import timedelta

# 使用定时器而不是持续轮询
async def async_setup_entry(hass, entry):
    async def async_update(now):
        """定时更新"""
        await update_sensors()
    
    # 每 30 秒更新一次
    entry.async_on_unload(
        async_track_time_interval(hass, async_update, timedelta(seconds=30))
    )
```

### 优化 2：批量更新实体

```python
# ✅ 正确：批量添加实体
async def async_setup_entry(hass, entry, async_add_entities):
    entities = [create_entity(i) for i in range(100)]
    async_add_entities(entities)

# ❌ 错误：逐个添加实体
async def async_setup_entry(hass, entry, async_add_entities):
    for i in range(100):
        async_add_entities([create_entity(i)])  # 效率低
```

### 优化 3：使用事件过滤器

```python
# ✅ 正确：使用过滤器减少不必要的函数调用
def event_filter(event: Event) -> bool:
    """事件过滤器"""
    entity_id = event.data.get("entity_id")
    return entity_id and entity_id.startswith("sensor.")

hass.bus.async_listen(
    EVENT_STATE_CHANGED,
    handler,
    event_filter=event_filter
)
```

### 优化 4：使用 eager_start 优化任务创建

```python
# ✅ 正确：使用 eager_start 减少任务切换开销
hass.async_create_task(
    my_coroutine(),
    eager_start=True
)
```

通过遵循这些最佳实践和优化建议，可以构建高性能、可靠的 Home Assistant 集成组件。
