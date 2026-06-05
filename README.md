# 🏛️ Public_Performance_System: 基于大语言模型的公共绩效智能治理与推演沙盒

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Public_Performance_System 是一个将**公共行政管理理论**与**计算社会科学（CSS）前沿 AI 技术**深度融合的开源实证与决策支持项目。

本项目摒弃了传统的静态指标评价模式，依托高参数量大语言模型（如 Qwen-32B / DeepSeek-V3），为基层政府（如街道办）构建了一个集“政策溯源、动态诊断、多主体仿真、经验沉淀”于一体的数字治理沙盒。它能够在政策真实落地前，有效压测公共资源分配与绩效考核中的摩擦成本与非预期后果。

---

## ✨ 核心特性 (Key Features)

### 1. 📚 基于 RAG 的垂直政策知识图谱 (Semantic Alignment)
* **痛点解决：** 消除通用大模型在行政公文生成中的“幻觉”。
* **技术实现：** 采用 LangChain 与本地 FAISS 向量数据库，对冗长的政务文件进行离线语义切片。支持秒级精准溯源，确保所有评估诊断均“有法可依”。

### 2. 📊 动态平衡计分卡 (Dynamic BSC Dashboard)
* **痛点解决：** 打破绩效评估中“唯分数论”的静态局限。
* **技术实现：** 前端集成交互式数据沙盘。当指标得分发生变动（如环保响应延迟、投诉率上升）时，系统自动捕捉异常，并驱动 LLM 生成结构化的归因分析与公文级整改报告。

### 3. 👥 基于主体的多部门政策博弈仿真 (Agent-Based Simulation)
* **痛点解决：** 提前识别“一刀切”或部门间利益冲突等政策执行阻力。
* **技术实现：** 引入大语言模型代理框架（LLM Agents），在虚拟环境中实例化不同行政主体（如市考核办、市环保局、基层街道办）。通过多轮自动化推演交互，展现真实体制内的博弈逻辑。

### 4. 🔄 制度沉淀与自进化 (AutoSkill Memory)
* **痛点解决：** 传统系统“算完就忘”，缺乏组织记忆。
* **技术实现：** 参照业内前沿的自进化机制，系统能将经过多模型交叉验证（Cross-Validation）的优质评估逻辑固化为 `.md` 技能脚本存入技能库，实现隐性专家知识的显性化与复用。

---

## 📂 仓库目录架构

```text
GovAI-Sim/
│
├── app.py                     # Streamlit 前端交互与视图分发主入口
├── rag_core.py                # LangChain 检索链、文本向量化与 LLM 通信核心
├── requirements.txt           # 项目运行基础环境依赖
├── .env.example               # 环境变量配置模板
├── README.md                  # 项目说明文档
│
├── data/                      # 静态物料与测试数据集
│   ├── policy_docs/           # 考核文件、网格化管理办法等 txt/pdf 原始语料
│   └── mock_data/             # 预置的 BSC 考核指标与市民投诉结构化种子数据
│
├── local_faiss/               # RAG 本地向量数据库（运行程序后自动生成）
└── skill_bank/                # AutoSkill 自进化技能沉淀库

```

## 🚀 快速启动 (Quick Start)

**1. 克隆仓库**

```bash
git clone [https://github.com/ChillOne123/GovAI-Sim.git](https://github.com/ChillOne123/Public_Performance_System.git)
cd GovAI-Sim

```

**2. 安装依赖**
本系统对硬件要求极低，纯 CPU 亦可流畅运行。

```bash
pip install -r requirements.txt

```

**3. 配置环境变量**
复制 `.env.example` 文件并重命名为 `.env`，填入各类兼容 OpenAI 格式的 API Key。

```env
API_KEY="your_api_key_here"
BASE_URL="[https://api.siliconflow.cn/v1](https://api.siliconflow.cn/v1)"

```

**4. 启动系统**

```bash
streamlit run app.py

```

*启动后，浏览器将自动打开 `http://localhost:8501` 即可体验。*

---

## 🛠️ 技术栈 (Tech Stack)

* **Frontend:** [Streamlit](https://streamlit.io/) + Pandas (Data Processing)
* **Backend Framework:** [LangChain](https://python.langchain.com/) (RAG & Agent Orchestration)
* **Vector Store:** [FAISS](https://faiss.ai/) (Local CPU-based vector search)
* **LLM Engine:** Qwen / DeepSeek (via SiliconFlow API)

## 📝 理论依托

本项目在业务逻辑设计上，深度参考了以下公共管理与计算社会科学前沿理论：

* 罗伯特·卡普兰的 **平衡计分卡 (BSC) 模型**
* 计算社会科学中的 **基于主体的建模 (Agent-Based Modeling)** * *ByteDance* 团队的 **MUSE-Autoskill** 大模型自进化架构

---

*Developed for advancing empirical research and intelligent decision-making in Public Administration.*
