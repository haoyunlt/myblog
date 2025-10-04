---
title: "CPython 框架使用示例和最佳实践"
date: 2025-09-28T01:46:41+08:00
draft: false
tags: ['源码分析', 'Python', '最佳实践']
categories: ['Python']
description: "CPython 框架使用示例和最佳实践的深入技术分析文档"
keywords: ['源码分析', 'Python', '最佳实践']
author: "技术分析师"
weight: 1
---

## 概述

本章提供 CPython 框架的实际使用示例，包括嵌入应用、扩展模块开发、性能优化和生产环境最佳实践。

## 1. CPython 嵌入式应用开发

### 1.1 基础嵌入示例

```c
/* 基础嵌入示例：在 C 应用中运行 Python 代码 */
#include <Python.h>
#include <stdio.h>

int main()
{
    // 1. 基本初始化
    Py_Initialize();

    if (!Py_IsInitialized()) {
        fprintf(stderr, "Python 初始化失败\n");
        return 1;
    }

    printf("Python 版本: %s\n", Py_GetVersion());

    // 2. 执行简单 Python 代码
    PyRun_SimpleString("print('Hello from embedded Python!')");
    PyRun_SimpleString("import sys; print('搜索路径:', sys.path[:3])");

    // 3. 执行文件中的 Python 代码
    FILE *fp = fopen("script.py", "r");
    if (fp) {
        PyRun_SimpleFile(fp, "script.py");
        fclose(fp);
    }

    // 4. 清理和退出
    Py_Finalize();
    return 0;
}
```

### 1.2 高级嵌入配置

```c
/* 高级嵌入：自定义配置和错误处理 */
#include <Python.h>

typedef struct {
    int initialized;
    PyObject *main_module;
    PyObject *global_dict;
} PythonContext;

PythonContext py_ctx = {0};

/*

 * 功能: 初始化 Python 环境
 * 参数: script_path - Python 脚本搜索路径
 * 返回: 0 成功，-1 失败

 */
int init_python(const char *script_path)
{
    PyStatus status;
    PyConfig config;

    // 初始化配置
    PyConfig_InitPythonConfig(&config);

    // 设置程序名
    status = PyConfig_SetBytesString(&config, &config.program_name, "MyApp");
    if (PyStatus_Exception(status)) {
        goto fail;
    }

    // 设置 Python 主目录
    if (script_path) {
        status = PyConfig_SetBytesString(&config, &config.home, script_path);
        if (PyStatus_Exception(status)) {
            goto fail;
        }
    }

    // 禁用信号处理器（嵌入应用通常自己处理信号）
    config.install_signal_handlers = 0;

    // 设置详细模式
    config.verbose = 1;

    // 从配置初始化
    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);

    if (PyStatus_Exception(status)) {
        return -1;
    }

    // 获取主模块和全局字典
    py_ctx.main_module = PyImport_AddModule("__main__");
    if (!py_ctx.main_module) {
        PyErr_Print();
        return -1;
    }
    Py_INCREF(py_ctx.main_module);

    py_ctx.global_dict = PyModule_GetDict(py_ctx.main_module);
    if (!py_ctx.global_dict) {
        return -1;
    }
    Py_INCREF(py_ctx.global_dict);

    py_ctx.initialized = 1;
    printf("Python 环境初始化成功\n");
    return 0;

fail:
    PyConfig_Clear(&config);
    if (PyStatus_IsExit(status)) {
        return status.exitcode;
    }
    return -1;
}

/*

 * 功能: 执行 Python 代码并返回结果
 * 参数: code - Python 代码字符串
 * 返回: PyObject* - 执行结果，NULL 表示失败

 */
PyObject *execute_python(const char *code)
{
    if (!py_ctx.initialized) {
        fprintf(stderr, "Python 环境未初始化\n");
        return NULL;
    }

    PyObject *result = PyRun_String(code, Py_eval_input,
                                   py_ctx.global_dict, py_ctx.global_dict);

    if (!result) {
        if (PyErr_Occurred()) {
            printf("Python 执行错误:\n");
            PyErr_Print();
        }
    }

    return result;
}

/*

 * 功能: 调用 Python 函数
 * 参数: func_name - 函数名
 *      format - 参数格式字符串
 *      ... - 可变参数
 * 返回: PyObject* - 函数返回值

 */
PyObject *call_python_function(const char *func_name, const char *format, ...)
{
    PyObject *func = PyDict_GetItemString(py_ctx.global_dict, func_name);
    if (!func || !PyCallable_Check(func)) {
        fprintf(stderr, "函数 '%s' 不存在或不可调用\n", func_name);
        return NULL;
    }

    PyObject *args = NULL;
    if (format && *format) {
        va_list vargs;
        va_start(vargs, format);
        args = Py_VaBuildValue(format, vargs);
        va_end(vargs);

        if (!args) {
            return NULL;
        }
    } else {
        args = PyTuple_New(0);
    }

    PyObject *result = PyObject_CallObject(func, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Print();
    }

    return result;
}

/*

 * 功能: 设置 Python 变量
 * 参数: name - 变量名
 *      value - 变量值（C 字符串）
 * 返回: 0 成功，-1 失败

 */
int set_python_variable(const char *name, const char *value)
{
    PyObject *py_value = PyUnicode_FromString(value);
    if (!py_value) {
        return -1;
    }

    int result = PyDict_SetItemString(py_ctx.global_dict, name, py_value);
    Py_DECREF(py_value);

    return result;
}

/*

 * 功能: 获取 Python 变量
 * 参数: name - 变量名
 * 返回: char* - 变量值字符串，需要调用者释放

 */
char *get_python_variable(const char *name)
{
    PyObject *value = PyDict_GetItemString(py_ctx.global_dict, name);
    if (!value) {
        return NULL;
    }

    PyObject *str_value = PyObject_Str(value);
    if (!str_value) {
        return NULL;
    }

    const char *c_str = PyUnicode_AsUTF8(str_value);
    char *result = strdup(c_str);  // 复制字符串

    Py_DECREF(str_value);
    return result;
}

/*

 * 功能: 清理 Python 环境

 */
void cleanup_python()
{
    if (py_ctx.initialized) {
        Py_XDECREF(py_ctx.global_dict);
        Py_XDECREF(py_ctx.main_module);
        Py_Finalize();
        py_ctx.initialized = 0;
        printf("Python 环境清理完成\n");
    }
}

/* 使用示例 */
int main()
{
    // 初始化
    if (init_python("./python_scripts") != 0) {
        fprintf(stderr, "Python 初始化失败\n");
        return 1;
    }

    // 设置变量
    set_python_variable("app_name", "CPython 嵌入应用");
    set_python_variable("version", "1.0.0");

    // 定义 Python 函数
    const char *python_functions =
        "def greet(name):\n"
        "    return f'Hello, {name}! Welcome to {app_name} v{version}'\n"
        "\n"
        "def calculate(x, y, operation='add'):\n"
        "    if operation == 'add':\n"
        "        return x + y\n"
        "    elif operation == 'multiply':\n"
        "        return x * y\n"
        "    elif operation == 'divide':\n"
        "        return x / y if y != 0 else None\n"
        "    else:\n"
        "        return None\n"
        "\n"
        "def process_data(data_list):\n"
        "    return {\n"
        "        'count': len(data_list),\n"
        "        'sum': sum(data_list),\n"
        "        'average': sum(data_list) / len(data_list) if data_list else 0,\n"
        "        'max': max(data_list) if data_list else None,\n"
        "        'min': min(data_list) if data_list else None\n"
        "    }\n";

    PyRun_SimpleString(python_functions);

    // 调用 Python 函数
    PyObject *greeting = call_python_function("greet", "s", "用户");
    if (greeting) {
        printf("问候消息: %s\n", PyUnicode_AsUTF8(greeting));
        Py_DECREF(greeting);
    }

    PyObject *calc_result = call_python_function("calculate", "dd", 15.5, 3.2);
    if (calc_result) {
        printf("计算结果: %.2f\n", PyFloat_AsDouble(calc_result));
        Py_DECREF(calc_result);
    }

    // 处理复杂数据结构
    PyObject *data_list = PyList_New(0);
    for (int i = 1; i <= 10; i++) {
        PyList_Append(data_list, PyLong_FromLong(i * i));
    }

    PyDict_SetItemString(py_ctx.global_dict, "test_data", data_list);
    PyObject *stats = call_python_function("process_data", "O", data_list);

    if (stats && PyDict_Check(stats)) {
        printf("数据统计:\n");

        PyObject *count = PyDict_GetItemString(stats, "count");
        PyObject *sum = PyDict_GetItemString(stats, "sum");
        PyObject *avg = PyDict_GetItemString(stats, "average");

        if (count) printf("  数量: %ld\n", PyLong_AsLong(count));
        if (sum) printf("  总和: %ld\n", PyLong_AsLong(sum));
        if (avg) printf("  平均值: %.2f\n", PyFloat_AsDouble(avg));

        Py_DECREF(stats);
    }

    Py_DECREF(data_list);

    // 异常处理演示
    PyObject *error_result = call_python_function("calculate", "dds", 10.0, 0.0, "divide");
    if (error_result == Py_None) {
        printf("除零操作返回 None\n");
    }
    Py_XDECREF(error_result);

    // 清理
    cleanup_python();
    return 0;
}
```

## 2. Python 扩展模块开发

### 2.1 完整扩展模块示例

```c
/* 完整的 Python 扩展模块：数据处理工具 */
#include <Python.h>
#include <math.h>
#include <string.h>

/* 模块状态结构 */
typedef struct {
    PyObject *ProcessError;  // 自定义异常
    long process_count;      // 处理计数
} DataProcessState;

/* 获取模块状态 */
static inline DataProcessState *
get_module_state(PyObject *module)
{
    void *state = PyModule_GetState(module);
    assert(state != NULL);
    return (DataProcessState *)state;
}

/*

 * 功能: 数据统计分析
 * 参数: self - 模块对象
 *      args - 位置参数
 *      kwargs - 关键字参数
 * 返回: 统计结果字典

 */
static PyObject *
dataprocess_analyze(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *data_obj;
    int detailed = 0;  // 默认简单统计

    static char *kwlist[] = {"data", "detailed", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist,
                                    &data_obj, &detailed)) {
        return NULL;
    }

    // 检查输入是否为序列
    if (!PySequence_Check(data_obj)) {
        PyErr_SetString(PyExc_TypeError, "data must be a sequence");
        return NULL;
    }

    Py_ssize_t length = PySequence_Length(data_obj);
    if (length < 0) {
        return NULL;
    }

    if (length == 0) {
        PyErr_SetString(PyExc_ValueError, "data sequence is empty");
        return NULL;
    }

    // 提取数值数据
    double *values = (double *)malloc(length * sizeof(double));
    if (!values) {
        return PyErr_NoMemory();
    }

    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject *item = PySequence_GetItem(data_obj, i);
        if (!item) {
            free(values);
            return NULL;
        }

        double val = PyFloat_AsDouble(item);
        Py_DECREF(item);

        if (PyErr_Occurred()) {
            free(values);
            return NULL;
        }

        values[i] = val;
    }

    // 计算基本统计量
    double sum = 0.0, sum_squares = 0.0;
    double min_val = values[0], max_val = values[0];

    for (Py_ssize_t i = 0; i < length; i++) {
        double val = values[i];
        sum += val;
        sum_squares += val * val;

        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    double mean = sum / length;
    double variance = (sum_squares / length) - (mean * mean);
    double std_dev = sqrt(variance);

    // 创建结果字典
    PyObject *result = PyDict_New();
    if (!result) {
        free(values);
        return NULL;
    }

    PyDict_SetItemString(result, "count", PyLong_FromSsize_t(length));
    PyDict_SetItemString(result, "sum", PyFloat_FromDouble(sum));
    PyDict_SetItemString(result, "mean", PyFloat_FromDouble(mean));
    PyDict_SetItemString(result, "min", PyFloat_FromDouble(min_val));
    PyDict_SetItemString(result, "max", PyFloat_FromDouble(max_val));
    PyDict_SetItemString(result, "std_dev", PyFloat_FromDouble(std_dev));

    if (detailed) {
        PyDict_SetItemString(result, "variance", PyFloat_FromDouble(variance));
        PyDict_SetItemString(result, "range",
                           PyFloat_FromDouble(max_val - min_val));

        // 计算中位数
        // 复制数组并排序
        double *sorted_values = (double *)malloc(length * sizeof(double));
        if (sorted_values) {
            memcpy(sorted_values, values, length * sizeof(double));

            // 简单冒泡排序（实际应用中应使用更高效的算法）
            for (Py_ssize_t i = 0; i < length - 1; i++) {
                for (Py_ssize_t j = 0; j < length - i - 1; j++) {
                    if (sorted_values[j] > sorted_values[j + 1]) {
                        double temp = sorted_values[j];
                        sorted_values[j] = sorted_values[j + 1];
                        sorted_values[j + 1] = temp;
                    }
                }
            }

            double median;
            if (length % 2 == 0) {
                median = (sorted_values[length/2 - 1] + sorted_values[length/2]) / 2.0;
            } else {
                median = sorted_values[length/2];
            }

            PyDict_SetItemString(result, "median", PyFloat_FromDouble(median));
            free(sorted_values);
        }
    }

    free(values);

    // 更新模块状态
    DataProcessState *state = get_module_state(self);
    state->process_count++;

    return result;
}

/*

 * 功能: 数据过滤
 * 参数: self - 模块对象
 *      args - 位置参数 (data, filter_func)
 * 返回: 过滤后的数据列表

 */
static PyObject *
dataprocess_filter(PyObject *self, PyObject *args)
{
    PyObject *data_obj, *filter_func;

    if (!PyArg_ParseTuple(args, "OO", &data_obj, &filter_func)) {
        return NULL;
    }

    if (!PySequence_Check(data_obj)) {
        PyErr_SetString(PyExc_TypeError, "first argument must be a sequence");
        return NULL;
    }

    if (!PyCallable_Check(filter_func)) {
        PyErr_SetString(PyExc_TypeError, "second argument must be callable");
        return NULL;
    }

    Py_ssize_t length = PySequence_Length(data_obj);
    if (length < 0) {
        return NULL;
    }

    PyObject *result = PyList_New(0);
    if (!result) {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject *item = PySequence_GetItem(data_obj, i);
        if (!item) {
            Py_DECREF(result);
            return NULL;
        }

        // 调用过滤函数
        PyObject *args_tuple = PyTuple_Pack(1, item);
        PyObject *filter_result = PyObject_CallObject(filter_func, args_tuple);

        Py_DECREF(args_tuple);

        if (!filter_result) {
            Py_DECREF(item);
            Py_DECREF(result);
            return NULL;
        }

        // 检查过滤结果
        int should_include = PyObject_IsTrue(filter_result);
        Py_DECREF(filter_result);

        if (should_include < 0) {
            Py_DECREF(item);
            Py_DECREF(result);
            return NULL;
        }

        if (should_include) {
            PyList_Append(result, item);
        }

        Py_DECREF(item);
    }

    return result;
}

/*

 * 功能: 数据变换
 * 参数: self - 模块对象
 *      args - 位置参数 (data, transform_func)
 * 返回: 变换后的数据列表

 */
static PyObject *
dataprocess_transform(PyObject *self, PyObject *args)
{
    PyObject *data_obj, *transform_func;

    if (!PyArg_ParseTuple(args, "OO", &data_obj, &transform_func)) {
        return NULL;
    }

    if (!PySequence_Check(data_obj)) {
        PyErr_SetString(PyExc_TypeError, "first argument must be a sequence");
        return NULL;
    }

    if (!PyCallable_Check(transform_func)) {
        PyErr_SetString(PyExc_TypeError, "second argument must be callable");
        return NULL;
    }

    Py_ssize_t length = PySequence_Length(data_obj);
    if (length < 0) {
        return NULL;
    }

    PyObject *result = PyList_New(length);
    if (!result) {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject *item = PySequence_GetItem(data_obj, i);
        if (!item) {
            Py_DECREF(result);
            return NULL;
        }

        // 调用变换函数
        PyObject *args_tuple = PyTuple_Pack(1, item);
        PyObject *transformed = PyObject_CallObject(transform_func, args_tuple);

        Py_DECREF(args_tuple);
        Py_DECREF(item);

        if (!transformed) {
            Py_DECREF(result);
            return NULL;
        }

        PyList_SetItem(result, i, transformed);  // 转移引用
    }

    return result;
}

/*

 * 功能: 获取处理统计信息
 * 参数: self - 模块对象
 *      args - 参数（无）
 * 返回: 统计信息字典

 */
static PyObject *
dataprocess_get_stats(PyObject *self, PyObject *args)
{
    DataProcessState *state = get_module_state(self);

    PyObject *stats = PyDict_New();
    if (!stats) {
        return NULL;
    }

    PyDict_SetItemString(stats, "process_count",
                        PyLong_FromLong(state->process_count));

    return stats;
}

/*

 * 功能: 重置统计信息
 * 参数: self - 模块对象
 *      args - 参数（无）
 * 返回: None

 */
static PyObject *
dataprocess_reset_stats(PyObject *self, PyObject *args)
{
    DataProcessState *state = get_module_state(self);
    state->process_count = 0;

    Py_RETURN_NONE;
}

/* 模块方法表 */
static PyMethodDef DataProcessMethods[] = {
    {"analyze", (PyCFunction)dataprocess_analyze,
     METH_VARARGS | METH_KEYWORDS,
     "Analyze data and return statistics.\n\n"
     "Args:\n"
     "    data: Sequence of numeric values\n"
     "    detailed (bool): Return detailed statistics\n\n"
     "Returns:\n"
     "    dict: Statistical analysis results"},

    {"filter", dataprocess_filter, METH_VARARGS,
     "Filter data using a predicate function.\n\n"
     "Args:\n"
     "    data: Sequence to filter\n"
     "    filter_func: Callable that returns True/False\n\n"
     "Returns:\n"
     "    list: Filtered data"},

    {"transform", dataprocess_transform, METH_VARARGS,
     "Transform data using a transformation function.\n\n"
     "Args:\n"
     "    data: Sequence to transform\n"
     "    transform_func: Callable that transforms each item\n\n"
     "Returns:\n"
     "    list: Transformed data"},

    {"get_stats", dataprocess_get_stats, METH_NOARGS,
     "Get module processing statistics."},

    {"reset_stats", dataprocess_reset_stats, METH_NOARGS,
     "Reset module processing statistics."},

    {NULL, NULL, 0, NULL}
};

/* 模块状态管理 */
static int
dataprocess_traverse(PyObject *module, visitproc visit, void *arg)
{
    DataProcessState *state = get_module_state(module);
    Py_VISIT(state->ProcessError);
    return 0;
}

static int
dataprocess_clear(PyObject *module)
{
    DataProcessState *state = get_module_state(module);
    Py_CLEAR(state->ProcessError);
    return 0;
}

static void
dataprocess_free(void *module)
{
    dataprocess_clear((PyObject *)module);
}

/* 模块定义 */
static struct PyModuleDef dataprocessmodule = {
    PyModuleDef_HEAD_INIT,
    "dataprocess",              /* 模块名 */
    "Data processing utilities",  /* 模块文档 */
    sizeof(DataProcessState),    /* 模块状态大小 */
    DataProcessMethods,          /* 方法表 */
    NULL,                       /* m_reload */
    dataprocess_traverse,       /* m_traverse */
    dataprocess_clear,          /* m_clear */
    dataprocess_free            /* m_free */
};

/* 模块初始化函数 */
PyMODINIT_FUNC
PyInit_dataprocess(void)
{
    PyObject *module = PyModule_Create(&dataprocessmodule);
    if (module == NULL) {
        return NULL;
    }

    DataProcessState *state = get_module_state(module);

    // 创建自定义异常
    state->ProcessError = PyErr_NewException("dataprocess.ProcessError",
                                           NULL, NULL);
    if (state->ProcessError == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(state->ProcessError);

    if (PyModule_AddObject(module, "ProcessError", state->ProcessError) < 0) {
        Py_DECREF(state->ProcessError);
        Py_DECREF(module);
        return NULL;
    }

    // 添加常量
    PyModule_AddIntConstant(module, "DEFAULT_PRECISION", 6);
    PyModule_AddStringConstant(module, "VERSION", "1.0.0");

    // 初始化状态
    state->process_count = 0;

    return module;
}
```

### 2.2 扩展模块测试

```python
# test_dataprocess.py - 扩展模块测试
import dataprocess
import math
import random

def test_analyze():
    """测试数据分析功能"""
    print("=== 测试数据分析 ===")

    # 生成测试数据
    data = [random.gauss(100, 15) for _ in range(1000)]

    # 基本统计
    basic_stats = dataprocess.analyze(data)
    print("基本统计:")
    for key, value in basic_stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # 详细统计
    detailed_stats = dataprocess.analyze(data, detailed=True)
    print("\n详细统计:")
    for key, value in detailed_stats.items():
        if key not in basic_stats:
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

def test_filter():
    """测试数据过滤功能"""
    print("\n=== 测试数据过滤 ===")

    numbers = list(range(-10, 11))
    print(f"原始数据: {numbers}")

    # 过滤正数
    positive = dataprocess.filter(numbers, lambda x: x > 0)
    print(f"正数: {positive}")

    # 过滤偶数
    even = dataprocess.filter(numbers, lambda x: x % 2 == 0)
    print(f"偶数: {even}")

    # 过滤绝对值大于5的数
    abs_gt5 = dataprocess.filter(numbers, lambda x: abs(x) > 5)
    print(f"绝对值>5: {abs_gt5}")

def test_transform():
    """测试数据变换功能"""
    print("\n=== 测试数据变换 ===")

    numbers = [1, 2, 3, 4, 5]
    print(f"原始数据: {numbers}")

    # 平方变换
    squares = dataprocess.transform(numbers, lambda x: x**2)
    print(f"平方: {squares}")

    # 三角函数变换
    sines = dataprocess.transform(numbers, lambda x: math.sin(x))
    print(f"正弦值: {[f'{x:.4f}' for x in sines]}")

    # 字符串变换
    strings = dataprocess.transform(numbers, lambda x: f"数字_{x}")
    print(f"字符串: {strings}")

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")

    try:
        dataprocess.analyze("not a sequence")
    except TypeError as e:
        print(f"预期的类型错误: {e}")

    try:
        dataprocess.analyze([])
    except ValueError as e:
        print(f"预期的值错误: {e}")

    try:
        dataprocess.filter([1, 2, 3], "not callable")
    except TypeError as e:
        print(f"预期的类型错误: {e}")

def test_performance():
    """性能测试"""
    print("\n=== 性能测试 ===")
    import time

    # 生成大量数据
    large_data = [random.random() for _ in range(100000)]

    # 测试分析性能
    start = time.time()
    result = dataprocess.analyze(large_data, detailed=True)
    end = time.time()

    print(f"分析 {len(large_data)} 个数据点用时: {end-start:.4f}秒")
    print(f"平均值: {result['mean']:.6f}")
    print(f"标准差: {result['std_dev']:.6f}")

    # 测试过滤性能
    start = time.time()
    filtered = dataprocess.filter(large_data, lambda x: x > 0.5)
    end = time.time()

    print(f"过滤 {len(large_data)} 个数据点用时: {end-start:.4f}秒")
    print(f"过滤后数据量: {len(filtered)}")

def test_module_stats():
    """测试模块统计"""
    print("\n=== 模块统计 ===")

    # 重置统计
    dataprocess.reset_stats()

    # 执行一些操作
    data = [1, 2, 3, 4, 5]
    dataprocess.analyze(data)
    dataprocess.analyze(data, detailed=True)

    # 检查统计信息
    stats = dataprocess.get_stats()
    print(f"处理次数: {stats['process_count']}")

if __name__ == "__main__":
    print(f"DataProcess 模块版本: {dataprocess.VERSION}")
    print(f"默认精度: {dataprocess.DEFAULT_PRECISION}")

    test_analyze()
    test_filter()
    test_transform()
    test_error_handling()
    test_performance()
    test_module_stats()

    print("\n所有测试完成!")
```

## 3. 多线程编程实践

### 3.1 GIL 管理最佳实践

```c
/* 多线程 CPython 编程最佳实践 */
#include <Python.h>
#include <pthread.h>
#include <unistd.h>

typedef struct {
    PyObject *callback;
    PyObject *data;
    int thread_id;
    volatile int should_stop;
} WorkerContext;

/*

 * 功能: CPU 密集型任务的线程函数
 * 说明: 演示正确的 GIL 管理

 */
void *cpu_intensive_worker(void *arg)
{
    WorkerContext *ctx = (WorkerContext *)arg;
    PyGILState_STATE gstate;

    printf("线程 %d 开始 CPU 密集型任务\n", ctx->thread_id);

    while (!ctx->should_stop) {
        // 获取 GIL 执行 Python 操作
        gstate = PyGILState_Ensure();

        // 执行少量 Python 操作
        if (ctx->callback && PyCallable_Check(ctx->callback)) {
            PyObject *args = PyTuple_Pack(2,
                PyLong_FromLong(ctx->thread_id),
                ctx->data);

            PyObject *result = PyObject_CallObject(ctx->callback, args);

            if (result) {
                // 处理结果
                if (PyLong_Check(result)) {
                    long value = PyLong_AsLong(result);
                    if (value == 0) {
                        ctx->should_stop = 1;
                    }
                }
                Py_DECREF(result);
            } else {
                // 处理异常
                PyErr_Print();
                ctx->should_stop = 1;
            }

            Py_DECREF(args);
        }

        // 释放 GIL
        PyGILState_Release(gstate);

        // 执行 CPU 密集型计算（不需要 GIL）
        double result = 0.0;
        for (int i = 0; i < 1000000; i++) {
            result += sin(i * 0.001) * cos(i * 0.001);
        }

        // 模拟处理时间
        usleep(100000);  // 100ms
    }

    printf("线程 %d 结束\n", ctx->thread_id);
    return NULL;
}

/*

 * 功能: I/O 密集型任务的线程函数
 * 说明: 演示 I/O 操作中的 GIL 管理

 */
void *io_intensive_worker(void *arg)
{
    WorkerContext *ctx = (WorkerContext *)arg;
    PyGILState_STATE gstate;

    printf("线程 %d 开始 I/O 密集型任务\n", ctx->thread_id);

    int operation_count = 0;

    while (!ctx->should_stop && operation_count < 10) {
        // 获取 GIL
        gstate = PyGILState_Ensure();

        // 模拟文件 I/O 操作
        const char *python_io_code =
            "import time\n"
            "import os\n"
            "filename = f'/tmp/thread_{thread_id}_data.txt'\n"
            "with open(filename, 'w') as f:\n"
            "    f.write(f'Thread {thread_id} operation {operation}\\n')\n"
            "with open(filename, 'r') as f:\n"
            "    content = f.read()\n"
            "os.unlink(filename)\n"
            "time.sleep(0.1)\n";

        // 设置 Python 变量
        PyObject *main_dict = PyModule_GetDict(PyImport_AddModule("__main__"));
        PyDict_SetItemString(main_dict, "thread_id",
                           PyLong_FromLong(ctx->thread_id));
        PyDict_SetItemString(main_dict, "operation",
                           PyLong_FromLong(operation_count));

        // 执行 I/O 操作
        PyRun_SimpleString(python_io_code);

        if (PyErr_Occurred()) {
            PyErr_Print();
            ctx->should_stop = 1;
        }

        // 释放 GIL
        PyGILState_Release(gstate);

        operation_count++;

        // I/O 等待期间不需要持有 GIL
        usleep(500000);  // 500ms
    }

    printf("线程 %d I/O 任务完成\n", ctx->thread_id);
    return NULL;
}

/*

 * 功能: 创建和管理多线程

 */
static PyObject *
run_multithreaded_tasks(PyObject *self, PyObject *args)
{
    PyObject *callback = NULL;
    PyObject *data = Py_None;
    int num_cpu_threads = 2;
    int num_io_threads = 3;

    if (!PyArg_ParseTuple(args, "|OO", &callback, &data)) {
        return NULL;
    }

    printf("启动多线程任务...\n");

    // 创建线程数组
    int total_threads = num_cpu_threads + num_io_threads;
    pthread_t *threads = malloc(total_threads * sizeof(pthread_t));
    WorkerContext *contexts = malloc(total_threads * sizeof(WorkerContext));

    if (!threads || !contexts) {
        free(threads);
        free(contexts);
        return PyErr_NoMemory();
    }

    // 保存当前线程状态
    PyThreadState *save_state = PyEval_SaveThread();

    // 创建 CPU 密集型线程
    for (int i = 0; i < num_cpu_threads; i++) {
        contexts[i].callback = callback;
        contexts[i].data = data;
        contexts[i].thread_id = i + 1;
        contexts[i].should_stop = 0;

        Py_XINCREF(callback);
        Py_XINCREF(data);

        if (pthread_create(&threads[i], NULL, cpu_intensive_worker, &contexts[i]) != 0) {
            printf("创建 CPU 线程 %d 失败\n", i);
        }
    }

    // 创建 I/O 密集型线程
    for (int i = 0; i < num_io_threads; i++) {
        int idx = num_cpu_threads + i;
        contexts[idx].callback = callback;
        contexts[idx].data = data;
        contexts[idx].thread_id = idx + 1;
        contexts[idx].should_stop = 0;

        Py_XINCREF(callback);
        Py_XINCREF(data);

        if (pthread_create(&threads[idx], NULL, io_intensive_worker, &contexts[idx]) != 0) {
            printf("创建 I/O 线程 %d 失败\n", i);
        }
    }

    // 等待一段时间
    sleep(5);

    // 通知所有线程停止
    for (int i = 0; i < total_threads; i++) {
        contexts[i].should_stop = 1;
    }

    // 等待所有线程完成
    for (int i = 0; i < total_threads; i++) {
        pthread_join(threads[i], NULL);
        Py_XDECREF(contexts[i].callback);
        Py_XDECREF(contexts[i].data);
    }

    // 恢复线程状态
    PyEval_RestoreThread(save_state);

    free(threads);
    free(contexts);

    printf("所有线程任务完成\n");

    Py_RETURN_NONE;
}

/* 模块方法定义 */
static PyMethodDef ThreadingExampleMethods[] = {
    {"run_multithreaded_tasks", run_multithreaded_tasks, METH_VARARGS,
     "Run CPU and I/O intensive tasks in multiple threads"},
    {NULL, NULL, 0, NULL}
};

/* 模块定义 */
static struct PyModuleDef threadingexamplemodule = {
    PyModuleDef_HEAD_INIT,
    "threading_example",
    "Multi-threading examples with proper GIL management",
    -1,
    ThreadingExampleMethods
};

PyMODINIT_FUNC
PyInit_threading_example(void)
{
    return PyModule_Create(&threadingexamplemodule);
}
```

## 4. 性能优化实践

### 4.1 缓存和对象重用

```c
/* 性能优化：缓存和对象重用 */
#include <Python.h>

/* 字符串缓存结构 */
typedef struct {
    PyObject **cache;
    size_t size;
    size_t capacity;
} StringCache;

static StringCache g_string_cache = {NULL, 0, 0};

/*

 * 功能: 初始化字符串缓存

 */
static int init_string_cache(size_t initial_capacity)
{
    g_string_cache.cache = (PyObject **)malloc(initial_capacity * sizeof(PyObject *));
    if (!g_string_cache.cache) {
        return -1;
    }

    g_string_cache.capacity = initial_capacity;
    g_string_cache.size = 0;

    return 0;
}

/*

 * 功能: 缓存字符串对象

 */
static void cache_string(PyObject *str)
{
    if (g_string_cache.size < g_string_cache.capacity) {
        Py_INCREF(str);
        g_string_cache.cache[g_string_cache.size++] = str;
    }
}

/*

 * 功能: 查找缓存的字符串

 */
static PyObject *find_cached_string(const char *str)
{
    for (size_t i = 0; i < g_string_cache.size; i++) {
        PyObject *cached = g_string_cache.cache[i];
        const char *cached_str = PyUnicode_AsUTF8(cached);
        if (cached_str && strcmp(cached_str, str) == 0) {
            Py_INCREF(cached);
            return cached;
        }
    }
    return NULL;
}

/*

 * 功能: 创建或获取缓存的字符串

 */
static PyObject *
get_or_create_string(PyObject *self, PyObject *args)
{
    const char *str;

    if (!PyArg_ParseTuple(args, "s", &str)) {
        return NULL;
    }

    // 首先检查缓存
    PyObject *cached = find_cached_string(str);
    if (cached) {
        return cached;
    }

    // 创建新字符串并缓存
    PyObject *new_str = PyUnicode_FromString(str);
    if (new_str) {
        cache_string(new_str);
    }

    return new_str;
}

/* 快速数字转换缓存 */
static PyObject *small_int_cache[512];  // -256 到 255
static int cache_initialized = 0;

/*

 * 功能: 初始化小整数缓存

 */
static void init_small_int_cache(void)
{
    if (cache_initialized) {
        return;
    }

    for (int i = 0; i < 512; i++) {
        small_int_cache[i] = PyLong_FromLong(i - 256);
    }
    cache_initialized = 1;
}

/*

 * 功能: 快速整数创建

 */
static PyObject *
fast_int(PyObject *self, PyObject *args)
{
    long value;

    if (!PyArg_ParseTuple(args, "l", &value)) {
        return NULL;
    }

    // 使用缓存的小整数
    if (value >= -256 && value <= 255) {
        PyObject *result = small_int_cache[value + 256];
        Py_INCREF(result);
        return result;
    }

    // 创建新的整数对象
    return PyLong_FromLong(value);
}

/*

 * 功能: 批量操作优化示例

 */
static PyObject *
batch_process(PyObject *self, PyObject *args)
{
    PyObject *data_list;
    PyObject *operation;

    if (!PyArg_ParseTuple(args, "OO", &data_list, &operation)) {
        return NULL;
    }

    if (!PyList_Check(data_list)) {
        PyErr_SetString(PyExc_TypeError, "first argument must be a list");
        return NULL;
    }

    if (!PyCallable_Check(operation)) {
        PyErr_SetString(PyExc_TypeError, "second argument must be callable");
        return NULL;
    }

    Py_ssize_t length = PyList_Size(data_list);
    PyObject *result = PyList_New(length);

    if (!result) {
        return NULL;
    }

    // 批量处理，减少函数调用开销
    PyObject *args_tuple = PyTuple_New(1);
    if (!args_tuple) {
        Py_DECREF(result);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject *item = PyList_GET_ITEM(data_list, i);  // 借用引用

        // 重用参数元组
        PyTuple_SET_ITEM(args_tuple, 0, item);
        Py_INCREF(item);  // 因为 SET_ITEM 会偷取引用

        PyObject *processed = PyObject_CallObject(operation, args_tuple);

        if (!processed) {
            Py_DECREF(args_tuple);
            Py_DECREF(result);
            return NULL;
        }

        PyList_SET_ITEM(result, i, processed);  // 转移引用
    }

    Py_DECREF(args_tuple);
    return result;
}

/*

 * 功能: 内存池管理示例

 */
#define POOL_SIZE 1024
static PyObject *object_pool[POOL_SIZE];
static int pool_size = 0;

static void init_object_pool(void)
{
    memset(object_pool, 0, sizeof(object_pool));
    pool_size = 0;
}

static PyObject *get_from_pool(void)
{
    if (pool_size > 0) {
        return object_pool[--pool_size];
    }
    return PyList_New(0);  // 创建新的空列表
}

static void return_to_pool(PyObject *obj)
{
    if (pool_size < POOL_SIZE && PyList_Check(obj)) {
        PyList_SetSlice(obj, 0, PyList_Size(obj), NULL);  // 清空列表
        object_pool[pool_size++] = obj;
    } else {
        Py_DECREF(obj);
    }
}

static PyObject *
use_object_pool(PyObject *self, PyObject *args)
{
    int operations;

    if (!PyArg_ParseTuple(args, "i", &operations)) {
        return NULL;
    }

    PyObject *results = PyList_New(0);
    if (!results) {
        return NULL;
    }

    for (int i = 0; i < operations; i++) {
        PyObject *temp_list = get_from_pool();

        // 执行一些操作
        for (int j = 0; j < 10; j++) {
            PyObject *item = PyLong_FromLong(i * 10 + j);
            PyList_Append(temp_list, item);
            Py_DECREF(item);
        }

        // 计算总和
        PyObject *sum_obj = PyLong_FromLong(0);
        for (Py_ssize_t k = 0; k < PyList_Size(temp_list); k++) {
            PyObject *item = PyList_GetItem(temp_list, k);
            PyObject *new_sum = PyNumber_Add(sum_obj, item);
            Py_DECREF(sum_obj);
            sum_obj = new_sum;
        }

        PyList_Append(results, sum_obj);
        Py_DECREF(sum_obj);

        // 归还到对象池
        return_to_pool(temp_list);
    }

    return results;
}
```

### 4.2 性能测试工具

```c
/* 性能测试和分析工具 */
#include <Python.h>
#include <time.h>
#include <sys/resource.h>

typedef struct {
    clock_t start_time;
    clock_t end_time;
    struct rusage start_usage;
    struct rusage end_usage;
    size_t start_memory;
    size_t peak_memory;
} PerformanceMetrics;

static PerformanceMetrics g_metrics;

/*

 * 功能: 开始性能测量

 */
static PyObject *
start_profiling(PyObject *self, PyObject *args)
{
    g_metrics.start_time = clock();
    getrusage(RUSAGE_SELF, &g_metrics.start_usage);

    // 获取内存使用情况
    g_metrics.start_memory = g_metrics.start_usage.ru_maxrss;
    g_metrics.peak_memory = g_metrics.start_memory;

    Py_RETURN_NONE;
}

/*

 * 功能: 结束性能测量

 */
static PyObject *
stop_profiling(PyObject *self, PyObject *args)
{
    g_metrics.end_time = clock();
    getrusage(RUSAGE_SELF, &g_metrics.end_usage);

    double cpu_time = ((double)(g_metrics.end_time - g_metrics.start_time)) / CLOCKS_PER_SEC;
    double user_time = (g_metrics.end_usage.ru_utime.tv_sec - g_metrics.start_usage.ru_utime.tv_sec) +
                       (g_metrics.end_usage.ru_utime.tv_usec - g_metrics.start_usage.ru_utime.tv_usec) / 1000000.0;
    double system_time = (g_metrics.end_usage.ru_stime.tv_sec - g_metrics.start_usage.ru_stime.tv_sec) +
                         (g_metrics.end_usage.ru_stime.tv_usec - g_metrics.start_usage.ru_stime.tv_usec) / 1000000.0;

    long memory_delta = g_metrics.end_usage.ru_maxrss - g_metrics.start_memory;

    PyObject *result = PyDict_New();
    PyDict_SetItemString(result, "cpu_time", PyFloat_FromDouble(cpu_time));
    PyDict_SetItemString(result, "user_time", PyFloat_FromDouble(user_time));
    PyDict_SetItemString(result, "system_time", PyFloat_FromDouble(system_time));
    PyDict_SetItemString(result, "memory_delta", PyLong_FromLong(memory_delta));
    PyDict_SetItemString(result, "page_faults",
                        PyLong_FromLong(g_metrics.end_usage.ru_majflt - g_metrics.start_usage.ru_majflt));

    return result;
}

/*

 * 功能: 基准测试装饰器

 */
static PyObject *
benchmark_function(PyObject *self, PyObject *args)
{
    PyObject *func;
    PyObject *test_args = NULL;
    int iterations = 1;

    if (!PyArg_ParseTuple(args, "O|Oi", &func, &test_args, &iterations)) {
        return NULL;
    }

    if (!PyCallable_Check(func)) {
        PyErr_SetString(PyExc_TypeError, "first argument must be callable");
        return NULL;
    }

    if (test_args == NULL) {
        test_args = PyTuple_New(0);
    } else {
        Py_INCREF(test_args);
    }

    // 开始基准测试
    clock_t start = clock();
    struct rusage start_usage;
    getrusage(RUSAGE_SELF, &start_usage);

    PyObject *last_result = NULL;
    for (int i = 0; i < iterations; i++) {
        Py_XDECREF(last_result);
        last_result = PyObject_CallObject(func, test_args);

        if (!last_result) {
            Py_DECREF(test_args);
            return NULL;
        }
    }

    clock_t end = clock();
    struct rusage end_usage;
    getrusage(RUSAGE_SELF, &end_usage);

    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    double avg_time = total_time / iterations;

    double user_time = (end_usage.ru_utime.tv_sec - start_usage.ru_utime.tv_sec) +
                       (end_usage.ru_utime.tv_usec - start_usage.ru_utime.tv_usec) / 1000000.0;

    PyObject *benchmark_result = PyDict_New();
    PyDict_SetItemString(benchmark_result, "iterations", PyLong_FromLong(iterations));
    PyDict_SetItemString(benchmark_result, "total_time", PyFloat_FromDouble(total_time));
    PyDict_SetItemString(benchmark_result, "avg_time", PyFloat_FromDouble(avg_time));
    PyDict_SetItemString(benchmark_result, "user_time", PyFloat_FromDouble(user_time));
    PyDict_SetItemString(benchmark_result, "last_result", last_result);

    Py_DECREF(test_args);
    return benchmark_result;
}
```

## 5. 生产环境最佳实践

### 5.1 错误处理和日志

```c
/* 生产环境错误处理和日志记录 */
#include <Python.h>
#include <syslog.h>
#include <stdio.h>

/* 日志级别 */
typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_ERROR = 3,
    LOG_LEVEL_CRITICAL = 4
} LogLevel;

static LogLevel current_log_level = LOG_LEVEL_INFO;

/*

 * 功能: 设置日志级别

 */
static PyObject *
set_log_level(PyObject *self, PyObject *args)
{
    int level;

    if (!PyArg_ParseTuple(args, "i", &level)) {
        return NULL;
    }

    if (level < 0 || level > 4) {
        PyErr_SetString(PyExc_ValueError, "log level must be between 0 and 4");
        return NULL;
    }

    current_log_level = (LogLevel)level;
    Py_RETURN_NONE;
}

/*

 * 功能: 记录日志

 */
static void
log_message(LogLevel level, const char *message)
{
    if (level < current_log_level) {
        return;
    }

    const char *level_names[] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"};
    time_t now = time(NULL);
    char *time_str = ctime(&now);
    time_str[strlen(time_str) - 1] = '\0';  // 移除换行符

    // 输出到控制台
    fprintf(stderr, "[%s] %s: %s\n", time_str, level_names[level], message);

    // 输出到系统日志
    int syslog_level;
    switch (level) {
        case LOG_LEVEL_DEBUG:
            syslog_level = LOG_DEBUG;
            break;
        case LOG_LEVEL_INFO:
            syslog_level = LOG_INFO;
            break;
        case LOG_LEVEL_WARNING:
            syslog_level = LOG_WARNING;
            break;
        case LOG_LEVEL_ERROR:
            syslog_level = LOG_ERR;
            break;
        case LOG_LEVEL_CRITICAL:
            syslog_level = LOG_CRIT;
            break;
    }

    syslog(syslog_level, "%s", message);
}

/*

 * 功能: Python 日志接口

 */
static PyObject *
log_info(PyObject *self, PyObject *args)
{
    const char *message;

    if (!PyArg_ParseTuple(args, "s", &message)) {
        return NULL;
    }

    log_message(LOG_LEVEL_INFO, message);
    Py_RETURN_NONE;
}

static PyObject *
log_warning(PyObject *self, PyObject *args)
{
    const char *message;

    if (!PyArg_ParseTuple(args, "s", &message)) {
        return NULL;
    }

    log_message(LOG_LEVEL_WARNING, message);
    Py_RETURN_NONE;
}

static PyObject *
log_error(PyObject *self, PyObject *args)
{
    const char *message;

    if (!PyArg_ParseTuple(args, "s", &message)) {
        return NULL;
    }

    log_message(LOG_LEVEL_ERROR, message);
    Py_RETURN_NONE;
}

/*

 * 功能: 安全的函数执行包装器

 */
static PyObject *
safe_execute(PyObject *self, PyObject *args)
{
    PyObject *func;
    PyObject *func_args = NULL;
    PyObject *default_result = Py_None;

    if (!PyArg_ParseTuple(args, "O|OO", &func, &func_args, &default_result)) {
        return NULL;
    }

    if (!PyCallable_Check(func)) {
        PyErr_SetString(PyExc_TypeError, "first argument must be callable");
        return NULL;
    }

    if (func_args == NULL) {
        func_args = PyTuple_New(0);
    } else {
        Py_INCREF(func_args);
    }

    PyObject *result = NULL;

    // 执行函数并捕获异常
    result = PyObject_CallObject(func, func_args);

    if (result == NULL) {
        // 记录异常详情
        PyObject *exc_type, *exc_value, *exc_traceback;
        PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);

        if (exc_type) {
            PyObject *exc_str = PyObject_Str(exc_value ? exc_value : exc_type);
            if (exc_str) {
                const char *exc_message = PyUnicode_AsUTF8(exc_str);
                log_message(LOG_LEVEL_ERROR, exc_message);
                Py_DECREF(exc_str);
            }
        }

        // 清理异常
        Py_XDECREF(exc_type);
        Py_XDECREF(exc_value);
        Py_XDECREF(exc_traceback);

        // 返回默认值
        result = default_result;
        Py_INCREF(result);
    }

    Py_DECREF(func_args);
    return result;
}

/*

 * 功能: 内存使用监控

 */
static PyObject *
monitor_memory(PyObject *self, PyObject *args)
{
    PyObject *threshold_obj = NULL;
    size_t threshold = 100 * 1024 * 1024;  // 默认 100MB

    if (!PyArg_ParseTuple(args, "|O", &threshold_obj)) {
        return NULL;
    }

    if (threshold_obj && PyLong_Check(threshold_obj)) {
        threshold = PyLong_AsSize_t(threshold_obj);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    size_t current_memory = usage.ru_maxrss;

    if (current_memory > threshold) {
        char warning_msg[256];
        snprintf(warning_msg, sizeof(warning_msg),
                "Memory usage (%zu bytes) exceeds threshold (%zu bytes)",
                current_memory, threshold);
        log_message(LOG_LEVEL_WARNING, warning_msg);
    }

    PyObject *result = PyDict_New();
    PyDict_SetItemString(result, "current_memory", PyLong_FromSize_t(current_memory));
    PyDict_SetItemString(result, "threshold", PyLong_FromSize_t(threshold));
    PyDict_SetItemString(result, "warning",
                        PyBool_FromLong(current_memory > threshold));

    return result;
}

/* 模块方法表 */
static PyMethodDef ProductionMethods[] = {
    {"set_log_level", set_log_level, METH_VARARGS, "Set logging level"},
    {"log_info", log_info, METH_VARARGS, "Log info message"},
    {"log_warning", log_warning, METH_VARARGS, "Log warning message"},
    {"log_error", log_error, METH_VARARGS, "Log error message"},
    {"safe_execute", safe_execute, METH_VARARGS, "Execute function safely"},
    {"monitor_memory", monitor_memory, METH_VARARGS, "Monitor memory usage"},
    {NULL, NULL, 0, NULL}
};

/* 模块定义 */
static struct PyModuleDef productionmodule = {
    PyModuleDef_HEAD_INIT,
    "production_utils",
    "Production environment utilities",
    -1,
    ProductionMethods
};

PyMODINIT_FUNC
PyInit_production_utils(void)
{
    openlog("python_app", LOG_PID | LOG_CONS, LOG_USER);

    PyObject *module = PyModule_Create(&productionmodule);
    if (module == NULL) {
        return NULL;
    }

    // 添加日志级别常量
    PyModule_AddIntConstant(module, "DEBUG", LOG_LEVEL_DEBUG);
    PyModule_AddIntConstant(module, "INFO", LOG_LEVEL_INFO);
    PyModule_AddIntConstant(module, "WARNING", LOG_LEVEL_WARNING);
    PyModule_AddIntConstant(module, "ERROR", LOG_LEVEL_ERROR);
    PyModule_AddIntConstant(module, "CRITICAL", LOG_LEVEL_CRITICAL);

    return module;
}
```

## 6. 构建和部署

### 6.1 setup.py 配置

```python
# setup.py - 扩展模块构建配置
from setuptools import setup, Extension
import sys
import os

# 编译器选项
extra_compile_args = [
    '-O3',           # 优化级别
    '-Wall',         # 警告
    '-Wextra',       # 额外警告
    '-std=c99',      # C99 标准
]

extra_link_args = []

# 平台特定设置
if sys.platform == 'win32':
    extra_compile_args.extend(['/Ox', '/W3'])
    extra_link_args.extend(['/MANIFEST'])
elif sys.platform == 'darwin':
    extra_compile_args.extend(['-Wno-unused-parameter'])
    extra_link_args.extend(['-Wl,-rpath,@loader_path'])
else:  # Linux 和其他 Unix
    extra_compile_args.extend(['-fPIC', '-pthread'])
    extra_link_args.extend(['-pthread'])

# 扩展模块定义
extensions = [
    Extension(
        'dataprocess',
        sources=['dataprocess.c'],
        include_dirs=[],
        libraries=['m'],  # 数学库
        library_dirs=[],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        'threading_example',
        sources=['threading_example.c'],
        libraries=['m', 'pthread'],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        'production_utils',
        sources=['production_utils.c'],
        libraries=['m'],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='cpython-examples',
    version='1.0.0',
    description='CPython API examples and utilities',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='CPython Developer',
    author_email='developer@example.com',
    url='https://github.com/example/cpython-examples',
    ext_modules=extensions,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='cpython extension api performance',
)
```

### 6.2 Makefile 配置

```makefile
# Makefile - 构建配置
CC = gcc
PYTHON = python3
PYTHON_CONFIG = python3-config

# 编译选项
CFLAGS = -std=c99 -fPIC -O3 -Wall -Wextra
CFLAGS += $(shell $(PYTHON_CONFIG) --cflags)
LDFLAGS = $(shell $(PYTHON_CONFIG) --ldflags)
LIBS = -lm -lpthread

# 目标文件
SOURCES = dataprocess.c threading_example.c production_utils.c
OBJECTS = $(SOURCES:.c=.o)
TARGETS = $(SOURCES:.c=.so)

# 默认目标
all: $(TARGETS)

# 编译规则
%.o: %.c
    $(CC) $(CFLAGS) -c $< -o $@

%.so: %.o
    $(CC) -shared $(LDFLAGS) $< -o $@ $(LIBS)

# 清理
clean:
    rm -f $(OBJECTS) $(TARGETS)
    rm -rf build/ dist/ *.egg-info/
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} +

# 安装
install: $(TARGETS)
    $(PYTHON) setup.py install

# 开发安装
develop: $(TARGETS)
    $(PYTHON) setup.py develop

# 测试
test: $(TARGETS)
    $(PYTHON) -m pytest tests/

# 文档生成
docs:
    sphinx-build -b html docs docs/_build

# 代码格式化
format:
    clang-format -i $(SOURCES)
    black *.py tests/*.py

# 静态分析
analyze:
    cppcheck --enable=all $(SOURCES)
    flake8 *.py tests/

# 性能分析
profile: $(TARGETS)
    $(PYTHON) -m cProfile -o profile.stats performance_test.py
    $(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# 内存检查
memcheck: $(TARGETS)
    valgrind --tool=memcheck --leak-check=full $(PYTHON) memory_test.py

.PHONY: all clean install develop test docs format analyze profile memcheck
```

## 7. 总结

CPython 框架使用的最佳实践包括：

### 7.1 核心原则

1. **正确的引用计数管理** - 防止内存泄漏和悬空指针
2. **异常安全编程** - 确保异常情况下的资源清理
3. **合理的 GIL 管理** - 在多线程环境中正确使用 GIL
4. **性能优化意识** - 缓存、批处理、对象重用

### 7.2 开发建议

1. **使用现代 C 标准** - C99 或更新版本
2. **遵循 Python 编码规范** - PEP 7 for C code
3. **编写完整的测试** - 单元测试和集成测试
4. **文档化 API** - 清晰的函数和参数说明

### 7.3 生产部署

1. **错误处理和日志** - 完善的错误报告机制
2. **性能监控** - CPU、内存、I/O 使用情况
3. **安全考虑** - 输入验证和资源限制
4. **版本管理** - 向后兼容性和升级策略

通过遵循这些实践，可以开发出高质量、高性能且可靠的 CPython 应用程序和扩展模块。
