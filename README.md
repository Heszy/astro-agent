# AstroAgent

一个面向分子云层级结构目录的科学数据分析 Agent。模型后端使用付费 **DeepSeek API**；工具执行、状态更新、错误反馈、预算和终止由本项目自带的 Agent harness 控制。数值计算始终由本机 Pandas/SciPy 完成，不交给大模型猜测。

## 项目展示什么

- 自研 Agent harness：状态循环、动作与时间预算、错误预算、重复动作熔断、观察截断和审计轨迹
- DeepSeek tool calling，但 API 只生成动作，不执行工具
- FastAPI 工程化封装
- 科学数据筛选、bootstrap、标度关系拟合与绘图
- Agent 工具选择正确率与 API 延迟、并发、token 用量评测
- 参数白名单、失败处理与无 API 单元测试

## 系统结构

```text
用户问题
   ↓
FastAPI /chat
   ↓
DeepSeek API（只生成下一步动作）
   ↓
自研 Agent Harness（状态 / 校验 / 执行 / 观察 / 重试 / 终止）
   ↓
受控工具层（schema / query / fit / compare / plot）
   ↓
本机 Pandas + SciPy + Matplotlib（真实计算）
   ↓
带工具轨迹、终止原因、token 用量和数值证据的回答
```

## 0. 准备 DeepSeek API

在 [DeepSeek开放平台](https://platform.deepseek.com/) 创建 API Key 并充值。API Key 只保存在本机 `.env`，不要提交到 Git，也不要截图公开。

本项目默认使用：

```text
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_THINKING=false
```

工具路由任务不需要长思考，因此默认关闭 thinking，以降低延迟和费用。模型名称和价格可能变化，运行前以 [DeepSeek官方价格页](https://api-docs.deepseek.com/quick_start/pricing/) 为准。

## 1. 环境准备

### Windows 11

在 PowerShell 中安装 Python 3.12：

```powershell
winget install Python.Python.3.12
```

安装完成后重新打开 PowerShell，解压并进入项目目录：

```powershell
cd C:\你的路径\astro-agent
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

`Set-ExecutionPolicy -Scope Process Bypass` 只对当前PowerShell窗口生效，不会永久修改系统策略。初始化脚本会创建 `.venv`、安装依赖、生成示例数据、运行7项离线测试，并在不存在时创建 `.env`。

编辑密钥：

```powershell
notepad .env
```

然后启动：

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.api:app --reload
```

如果不希望激活虚拟环境，也可以直接执行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload
```

### macOS

建议使用 Python 3.11 或 3.12：

```bash
brew install python@3.12
```

解压并进入项目目录，然后执行：

```bash
bash scripts/setup_mac.sh
```

该脚本会创建 `.venv`、安装依赖、生成720行示例目录并运行不调用 API 的测试。预期看到：

```text
7 passed
```

## 2. 配置密钥

```bash
cp .env.example .env
nano .env
```

把占位符替换成真实密钥：

```text
DEEPSEEK_API_KEY=sk-你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_THINKING=false
ASTRO_DATA_PATH=data/sample_catalog.csv
ASTRO_RESULTS_DIR=results
```

保存 `nano`：按 `Ctrl-O`、回车，再按 `Ctrl-X`。

## 3. 启动服务

```bash
source .venv/bin/activate
uvicorn app.api:app --reload
```

打开 <http://127.0.0.1:8000/docs>：

1. 调用 `GET /health`。这个请求不会调用付费 API。
2. 确认 `api_key_configured` 为 `true`。
3. 调用 `POST /chat`，输入：

```json
{
  "question": "拟合半径与速度弥散的标度关系，报告斜率、样本量和95%置信区间。"
}
```

也可以在新终端调用：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"拟合半径与速度弥散的标度关系，报告斜率和95%置信区间。"}'
```

响应中的关键字段：

- `trace`：每次工具动作、参数、结果、状态和耗时
- `steps`：Agent决策步数
- `termination_reason`：正常回答、预算终止或熔断原因
- `api_usage`：API返回的 token 用量

## 4. 运行 Agent 评测

```bash
python scripts/benchmark_agent.py
```

该命令会产生真实 API 费用。它运行6个问题，主要检查模型是否选择正确工具，结果保存到：

```text
results/agent_benchmark.csv
```

## 5. 运行 DeepSeek API 性能评测

```bash
python scripts/benchmark_api.py
```

默认在并发1和2下分别调用3次，并把单次输出限制在160 token。记录：

- 总响应时间
- prompt/completion/total tokens
- 近似输出 tokens/s
- cache hit/miss tokens（若API返回）

结果保存到：

```text
results/deepseek_api_benchmark.csv
```

脚本故意不硬编码费用，因为官方价格会调整。用 CSV 中的 token 数和运行时官方价格计算实际费用。

## 6. 替换真实数据

真实 CSV 第一版需要这些列：

```text
structure_id,parent_id,cloud_id,radius_pc,velocity_dispersion_kms,
mass_msun,column_density_cm2,virial_parameter,hierarchy_level
```

复制数据：

```bash
cp /你的路径/catalog.csv data/molecular_cloud_catalog.csv
```

修改 `.env`：

```text
ASTRO_DATA_PATH=data/molecular_cloud_catalog.csv
```

如果列名不同，优先在导出时重命名。不要删除 `REQUIRED_COLUMNS` 检查；单位必须同步更新 `app/catalog.py` 中的 `DATA_DICTIONARY`。

## 7. 面试演示顺序

1. 展示 `/health`，说明数据和工具在本地，模型使用DeepSeek API。
2. 展示 `app/harness.py`，说明模型不控制循环和工具执行。
3. 提问“目录有哪些字段”，展示schema动作。
4. 提问带筛选条件的标度关系问题。
5. 展示工具参数、样本量、斜率、置信区间和生成图。
6. 展示异常动作熔断测试和两个 benchmark CSV。
7. 展示每次请求的 token 用量，说明成本意识。

## 安全与边界

- API Key 只从环境变量读取，`.env` 已加入 `.gitignore`。
- 用户问题、系统提示词和返回给模型的工具结果会发送给 DeepSeek API；不要直接处理保密数据。
- 这是目录级分析，不直接读取35亿体素原始数据。
- bootstrap置信区间不是所有科学问题的完整显著性检验。
- 当前服务没有身份验证，不应暴露到公网。
- harness 不允许模型执行任意 Python、Shell 或 SQL。

## 简历描述模板

> 自研面向科学数据分析的 Agent harness，使用 DeepSeek API 生成工具动作，在本地完成动作校验、数据查询、bootstrap 标度关系拟合和可视化；实现步数/时间/错误预算、重复动作熔断、完整审计轨迹与 token 用量统计，并通过 FastAPI 对外提供服务。建立工具选择正确率及不同并发下 API 延迟、吞吐和 token 成本评测流程，数值结果由 Pandas/SciPy 确定性计算。

把真实准确率、延迟和 token 数据补在末尾，不要填写未实际测得的指标。
