import streamlit as st
import pandas as pd
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from rag_core import query_policy, build_vector_db

# ================= 1. 环境与模型初始化 =================
load_dotenv()
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")

# ================= 2. 页面与全局配置 =================
st.set_page_config(
    page_title="GovAI-Sim 公共绩效智能沙盒", 
    page_icon="🏛️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏 Streamlit 默认水印
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ================= 3. 侧边栏导航 =================
with st.sidebar:
    st.title("⚙️ 治理仿真系统")
    st.markdown("---")
    
    page = st.radio(
        "选择沙盒环境：",
        ["📚 政策大脑 (RAG检索)", "📊 动态考核沙盘 (数据诊断)", "👥 多主体博弈仿真 (Agent)"],
        index=1 # 默认跳到沙盘页方便你测试
    )
    
    st.markdown("---")
    st.markdown("### 🧠 交叉验证基座")
    model_choice = st.selectbox(
        "切换底层大语言模型：", 
        ["Pro/zai-org/GLM-4.7", "Qwen/Qwen3-32B", "deepseek-ai/DeepSeek-V3"]
    )
    st.caption("提示: 切换模型可进行决策鲁棒性检验 (Robustness Check)")

# 动态获取当前选中的模型（用于沙盘和Agent推演）
# 修复前：
# @st.cache_resource(show_spinner=False, dependencies=[model_choice])

# 修复后（直接复制这行替换即可）：
@st.cache_resource(show_spinner=False)
def get_llm(model_name):
    return ChatOpenAI(
        model=model_name,
        openai_api_key=SILICONFLOW_API_KEY,
        openai_api_base=SILICONFLOW_BASE_URL,
        temperature=0.4 
    )

current_llm = get_llm(model_choice)

# ================= 模块 1: 政策大脑 =================
if page == "📚 政策大脑 (RAG检索)":
    st.title("📚 垂直领域政策知识检索系统")
    st.markdown("基于本地专属知识库。杜绝 AI 幻觉，确保依据精准。")
    
    with st.expander("🛠️ 系统管理员选项 (更新底层知识库)"):
        if st.button("重建本地向量库"):
            with st.spinner("正在对政策文本进行切片与向量化处理..."):
                st.success(build_vector_db())
                
    st.markdown("---")
    user_query = st.chat_input("请输入具体的绩效考核疑问...")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("正在检索本地政策库并对齐语义..."):
                answer = query_policy(user_query)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

# ================= 模块 2: 动态考核沙盘 =================
elif page == "📊 动态考核沙盘 (数据诊断)":
    st.title("📊 动态平衡计分卡 (BSC) 诊断沙盘")
    st.markdown("模拟篡改基层运行数据，压测考核指标的韧性并自动生成公文级整改报告。")
    
    # 构造一份包含“雷点”的初始考核数据
    if "bsc_data" not in st.session_state:
        st.session_state.bsc_data = pd.DataFrame({
            "考核维度": ["热线办理", "热线办理", "网格治理", "网格治理", "红线约束"],
            "指标名称": ["热线诉求同比降幅", "市民满意率", "网格主动上报数", "漏报事部件数", "异常退单/强行反馈"],
            "基准要求": ["同比下降", "85%", "≥30件/月", "0件", "0件"],
            "当前数值": ["上升2%", "81%", "26件", "4件", "1件"], # 这里全是扣分项，方便演示诊断
            "计分规则": ["每升1%扣0.1", "低于85%每1%扣0.5", "少1件扣0.5", "每件扣0.01", "每件扣0.5"]
        })

    st.warning("💡 **操作提示：** 您可以在下方表格中直接双击修改【当前数值】列的数据（例如把满意率改成 90%），然后点击诊断按钮，系统将根据新数据重新生成评估。")
    
    # 使用 st.data_editor 让表格可编辑
    edited_df = st.data_editor(
        st.session_state.bsc_data, 
        use_container_width=True,
        hide_index=True,
        disabled=["考核维度", "指标名称", "基准要求", "计分规则"] # 冻结规则列，只允许改数据
    )
    
    if st.button("🚀 启动大模型智能归因与诊断", type="primary"):
        with st.spinner(f"正在调用 {model_choice} 进行多维数据穿透分析..."):
            # 将前端表格转换为文本喂给模型
            data_str = edited_df.to_string(index=False)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个资深的政府绩效评估专家。请根据以下提取的基层街道运行数据表，写一份严肃的《月度绩效异常诊断与整改督办单》。\n"
                           "要求：\n1. 严格对照'基准要求'和'计分规则'，计算并指出具体扣了多少分。\n"
                           "2. 语言必须高度符合中国大陆体制内公文规范（使用'经查'、'亟待整改'、'压实责任'等词汇）。\n"
                           "3. 使用 Markdown 排版，结构清晰。"),
                ("human", "当前各维度数据如下：\n{data}")
            ])
            
            chain = prompt | current_llm
            result = chain.invoke({"data": data_str})
            
            st.markdown("### 📑 智能诊断督办单")
            st.info(result.content)

# ================= 模块 3: 多主体博弈仿真 =================
elif page == "👥 多主体博弈仿真 (Agent)":
    st.title("👥 基于主体的多部门政策博弈仿真 (ABM)")
    st.markdown("实例化不同利益诉求的行政部门，在资源受限的情况下推演政策执行摩擦。")
    
    # 设定博弈议题
    issue = st.text_area("输入博弈议题 / 突发事件：", 
                         "某小区突发群租房引发的消防隐患投诉，12345工单激增。但该小区同时面临老旧改造，若强行清退群租客将引发群体上访。")
    
    col1, col2 = st.columns(2)
    with col1:
        rounds = st.slider("设定多主体推演轮数：", 1, 3, 2)
    with col2:
        start_sim = st.button("▶️ 开始高维空间推演", type="primary")

    if start_sim:
        # 定义三个阵营的系统 Prompt
        agents = {
            "市城运中心考核员": "你是市城运中心考核员。你的首要任务是盯着【12345满意率】和【按时办结率】。你极其严厉，只看数据结果，对基层找的任何客观理由都认为是推诿扯皮。",
            "职能部门(房管局/消防)": "你是市属职能部门。你的首要任务是【规避主责风险】。你倾向于认为这是基层的网格巡查不到位，或者建议把工单退回给街道兜底，强调你们缺乏执法权或人手。",
            "基层街道办书记": "你是街道办书记。你的首要任务是【维稳与资源博弈】。你对上级“既要又要”的考核感到愤怒，你会疯狂抱怨人力不足，并要求职能部门联合执法，绝不单独背锅。"
        }
        
        st.markdown("### 🔄 仿真推演进程")
        sim_history = f"【初始事件】: {issue}\n\n"
        
        # 动态创建一个占位符用于流式显示对话
        chat_container = st.container()
        
        with chat_container:
            for r in range(rounds):
                st.markdown(f"#### 🔁 第 {r+1} 轮博弈")
                for agent_name, agent_persona in agents.items():
                    with st.spinner(f"{agent_name} 正在思考对策..."):
                        # 构建博弈 Prompt，将历史发言作为上下文
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", f"{agent_persona}\n请根据当前的事件背景和前人的讨论，发表你的观点、推卸责任或提出要求。保持人设，言辞犀利。限制在150字以内。"),
                            ("human", f"当前讨论记录：\n{sim_history}\n请你发言：")
                        ])
                        
                        chain = prompt | current_llm
                        response = chain.invoke({})
                        reply = response.content
                        
                        # 记录历史并展示
                        sim_history += f"[{agent_name}]: {reply}\n"
                        st.chat_message("assistant").markdown(f"**【{agent_name}】:** {reply}")
                        time.sleep(1) # 稍微停顿，增加现场推演的真实感
                        
        st.markdown("---")
        st.success("✅ 仿真推演结束。您可以根据博弈记录，进一步提取政策摩擦的结构性节点。")