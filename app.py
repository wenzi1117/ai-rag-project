import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

st.title("AI 本地知识库")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("上传PDF文件", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = Ollama(model="qwen3:4b")

    query = st.text_input("请输入问题")

    if query:
        related_docs = vectorstore.similarity_search(query, k=4)
        context = "\n".join([doc.page_content for doc in related_docs])

        history_text = "\n".join(
            [f"用户：{h['user']}\nAI：{h['ai']}" for h in st.session_state.history]
        )

        prompt = f"""
你是一个论文分析助手。

历史对话：
{history_text}

请严格根据以下资料回答问题，不要编造：

{context}

用户问题：
{query}
"""

        response = llm.invoke(prompt)

        st.session_state.history.append({
            "user": query,
            "ai": response
        })

        st.write("AI回答：")
        st.write(response)

    if st.session_state.history:
        st.subheader("历史对话")
        for h in st.session_state.history:
            st.write("用户：", h["user"])
            st.write("AI：", h["ai"])