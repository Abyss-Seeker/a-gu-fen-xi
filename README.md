# 个股深度诊断报告工具（本地版）

这是一款基于 Python Flask 的个股深度诊断工具，输入股票代码即可生成包含基本面、技术面、资金面、事件面、行业面、价值面六大维度的综合诊断报告，并内置 AI 对话助手。**本版本为纯本地运行版，无需网络部署，一键启动即可使用。**

---

## 目录

- [功能特点](#功能特点)
- [准备工作（只需做一次）](#准备工作只需做一次)
- [第一步：下载项目](#第一步下载项目)
- [第二步：安装 Python](#第二步安装-python)
- [第三步：安装依赖](#第三步安装依赖)
- [第四步：配置 AI 接口（可选）](#第四步配置-ai-接口可选)
- [第五步：启动程序](#第五步启动程序)
- [使用方法](#使用方法)
- [常见问题](#常见问题)

---

## 功能特点

- **实时行情**：接入腾讯股票 API，获取实时价格、涨跌幅、成交量等数据
- **六维诊断**：基本面（25分）+ 技术面（20分）+ 资金面（15分）+ 事件面（10分）+ 行业面（15分）+ 价值面（15分），满分 100 分
- **AI 深度分析**：接入 ChatGPT/DeepSeek 等 OpenAI 兼容 API，生成自然语言分析报告
- **替代标的推荐**：自动推荐同行业可替代股票，附详细对比
- **K 线图表**：可视化展示历史走势
- **预设示例**：内置多只热门股票快速体验

---

## 准备工作（只需做一次）

在开始之前，请确认你的电脑满足以下条件：

| 条件 | 最低要求 |
|------|---------|
| 操作系统 | Windows / macOS / Linux 均可 |
| 网络连接 | 需要（从腾讯/东方财富获取行情数据） |
| 硬盘空间 | 约 200 MB |

---

## 第一步：下载项目

### 方法一：使用 Git（推荐）

1. 打开终端（Windows 按 `Win+R`，输入 `cmd` 回车；macOS 打开 `终端`）

2. 输入以下命令下载项目：

   ```bash
   git clone -b local-version https://github.com/Abyss-Seeker/a-gu-fen-xi.git
   cd a-gu-fen-xi
   ```

> 💡 如果没有安装 Git，请看方法二。

### 方法二：下载 ZIP 包

1. 打开浏览器，访问：
   ```
   https://github.com/Abyss-Seeker/a-gu-fen-xi/tree/local-version
   ```

2. 点击绿色的 **Code** 按钮 → 选择 **Download ZIP**

3. 将下载的 ZIP 文件解压到你喜欢的位置（比如桌面）

---

## 第二步：安装 Python

本项目需要 Python 3.9 或更高版本。

### 检查是否已安装 Python

打开终端，输入：

```bash
python --version
```

如果显示 `Python 3.x.x`（且版本号 >= 3.9），说明已安装，可以跳过这一步。

### 如果没有安装 Python

1. 打开浏览器访问：https://www.python.org/downloads/
2. 点击黄色按钮 **Download Python**，下载最新版
3. 运行下载的安装程序

> ⚠️ **Windows 用户注意**：安装时**务必勾选底部的 `Add Python to PATH`**，否则后续命令会报"找不到 python"

4. 安装完成后，**重新打开终端**，再次输入 `python --version` 确认安装成功。

---

## 第三步：安装依赖

打开终端，进入项目文件夹（如果还没进入的话）：

```bash
cd a-gu-fen-xi
```

然后输入以下命令安装所需库：

```bash
pip install -r requirements.txt
```

等待安装完成（约 1-2 分钟），看到 `Successfully installed` 字样即可。

> 💡 如果提示 `pip` 命令找不到，试试 `pip3 install -r requirements.txt` 或 `python -m pip install -r requirements.txt`

---

## 第四步：配置 AI 接口（可选）

> ⚠️ **如果不配置 AI 接口**：诊断报告的基础评分和图表仍然可用，但不会生成 AI 分析文字，AI 聊天面板也无法使用。建议配置以获得完整体验。

### 创建配置文件

找到项目文件夹，新建一个文件名为 `config.json`，将以下内容复制进去：

```json
{
  "ai_chat": {
    "provider": "deepseek",
    "api_key": "你的API密钥粘贴在这里",
    "api_base": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "system_prompt": "你是一位专业的股票投资分析师，擅长基本面分析、技术分析和投资策略。请用中文回答用户关于股票投资的问题。回答应专业、客观，包含风险提示。"
  },
  "data_source": {
    "preferred": "akshare",
    "cache_duration_minutes": 30
  },
  "report": {
    "enable_ai_analysis": true,
    "max_history_years": 5
  }
}
```

### 推荐 API 服务商

| 服务商 | api_key 获取地址 | 填入 api_base |
|--------|-----------------|---------------|
| DeepSeek（推荐，便宜） | https://platform.deepseek.com/api_keys | `https://api.deepseek.com/v1` |
| OpenAI | https://platform.openai.com/api-keys | `https://api.openai.com/v1` |
| 硅基流动（国内） | https://siliconflow.cn | `https://api.siliconflow.cn/v1` |

只需修改 `api_key` 为你自己的密钥，其他保持不变即可。

---

## 第五步：启动程序

在终端中，确认当前在项目文件夹下，然后输入：

```bash
python app.py
```

看到如下输出说明启动成功：

```
 * Running on http://0.0.0.0:8888
 * Debugger is active!
```

打开浏览器，在地址栏输入：

```
http://localhost:8888
```

即可开始使用！

---

## 使用方法

### 基础操作

1. 在搜索框中输入股票代码（如 `600519` 代表贵州茅台，`000001` 代表平安银行）
2. 点击「生成诊断报告」
3. 等待约 10-30 秒，即可查看完整的六维诊断报告

### 预设示例

点击搜索栏下方的快捷按钮，可以快速查看热门股票的诊断报告：
- 贵州茅台（600519）
- 平安银行（000001）
- 比亚迪（002594）
- 宁德时代（300750）

### AI 聊天

报告页面右侧有 AI 聊天面板，你可以：
- 询问任何关于该股票的问题
- 让 AI 解读各项指标的含义
- 请求投资建议（仅供参考，不构成投资建议）

---

## 常见问题

### Q: 启动时报错 `ModuleNotFoundError: No module named 'flask'`

**A:** 说明依赖没有安装成功。请重新执行第三步：
```bash
pip install -r requirements.txt
```

### Q: 打开网页后提示"API Key 未配置"

**A:** 检查 `config.json` 是否在项目文件夹下，且格式正确，可以忽略继续使用（只是 AI 功能不可用）。

### Q: 报告生成失败或加载不出数据

**A:** 请检查网络连接。本工具需要联网从腾讯和东方财富获取行情数据。如果是公司网络，可能需要关闭代理或 VPN。

### Q: Windows 提示 `python` 不是内部命令

**A:** Python 没有正确安装或没加入 PATH。请重新安装 Python，并在安装界面勾选 `Add Python to PATH`。

### Q: 如何关闭程序？

**A:** 在终端窗口按 `Ctrl+C`（macOS 也是 `Ctrl+C`），然后关闭终端窗口即可。

### Q: 我可以同时运行网页版和本地版吗？

**A:** 可以。网页版部署在 Vercel（https://a-gu-fen-xi-eight.vercel.app/），本地版运行在 `localhost:8888`，两者互不干扰。

---

## 风险提示

> 本工具生成的所有分析报告、评分和 AI 建议**仅供学习参考，不构成任何投资建议**。股市有风险，投资需谨慎。请勿依据本工具的输出做出任何投资决策。

---

## 许可证

MIT License
