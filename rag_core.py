# 【核心后端】LangChain 检索与大模型调用逻辑封装
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# 1. 加载隐藏在 .env 中的环境变量
load_dotenv()
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")

# 2. 初始化大模型与 Embeddings 模型
# 使用硅基流动免费/廉价的向量模型
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3", 
    openai_api_key=SILICONFLOW_API_KEY,
    openai_api_base=SILICONFLOW_BASE_URL
)

# 使用 Qwen 72B 作为主脑，温度调至 0.1 确保体制内公文的严谨性，不乱发散
llm = ChatOpenAI(
    model="Qwen/Qwen3-32B",
    openai_api_key=SILICONFLOW_API_KEY,
    openai_api_base=SILICONFLOW_BASE_URL,
    temperature=0.1 
)

# 3. 核心功能 A：构建本地向量知识库 (离线处理)
def build_vector_db(docs_dir="data/policy_data", db_path="local_faiss"):
    """遍历文件夹下的 PDF，切片并生成向量库"""
    all_docs = []
    
    # 自动读取目录下所有的 PDF 文件
    for filename in os.listdir(docs_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(docs_dir, filename)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # 文本切片：每块 500 字，重叠 100 字防止上下文断层
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            all_docs.extend(docs)
            
    if not all_docs:
        return "❌ 未在指定目录下找到 PDF 文件，请检查 data/policy_data/ 目录！"
        
    # 构建并保存 FAISS 向量库到本地
    vector_store = FAISS.from_documents(all_docs, embeddings)
    vector_store.save_local(db_path)
    return f"✅ 专属知识库构建完成！共切分 {len(all_docs)} 个政策文本块。"

# 4. 核心功能 B：问答检索链 (供前端 Streamlit 调用)
def query_policy(question, db_path="local_faiss"):
    """检索本地库并生成回答"""
    # 允许危险反序列化是因为这个库是我们本地自己生成的，绝对安全
    vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    # 每次检索最相关的 3 个政策片段
    retriever = vector_store.as_retriever(search_kwargs={"k": 8}) 
    
    # 设计充满行政督查威严感的高质量 Prompt
    system_prompt = (
    "你是一个兼具'政府绩效督查实务'与'公共管理学术理论'的顶级专家。\n"
    "请严格依据以下【参考资料】的内容回答用户的问题。\n\n"
    "【执行指令】:\n"
    "1. 若用户询问具体的基层考核规则、扣分标准，请务必精确引用具体数值和条款，保持体制内公文语调。\n"
    "2. 若用户询问宏观的绩效管理理论、学术前沿或概念辨析，请结合资料中的学术文献进行深度剖析，保持严谨的学术语调。\n"
    "3. 如果参考资料中完全没有提及相关内容，请直接回复：“当前知识库中暂无相关信息”，切勿主观臆造。\n\n"
    "【参考资料】:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 组装完整的 RAG 检索链
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    # 执行大模型调用
    response = rag_chain.invoke({"input": question})
    return response["answer"]

# ================= 测试区域 =================
if __name__ == "__main__":
    print("正在测试系统后端...")
    
    # 第一次运行，取消下面这行代码的注释，让它把 PDF 变成向量库
    # print(build_vector_db())
    
    # 库建好后，注释掉上面那行，取消下面两行的注释，测试大模型问答
    answer = query_policy("如果居民区辖区内发生新增在建违法建筑，城运中心会怎么扣分？")
    print("\n🤖 AI 督查员反馈:\n", answer)