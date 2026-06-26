import streamlit as st #快速做網頁的套件
import requests #串聯 FastAPI 的套件

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="🦮",
    layout="wide",
)

st.title("Chat with your PDF 🦮")
st.caption(" Powered by RAG — Retrieval Augmented Generation")

with st.sidebar:

    st.header("Upload your PDF")
    uploaded_files = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload one or more PDF files to chat with them.",
        accept_multiple_files=True
        )

    if uploaded_files:
        if st.button("Ingest PDF", type="primary", use_container_width=True):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    response = requests.post(
                        f"{BACKEND_URL}/ingest",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"{uploaded_file.name}：Added {data['chunks_added']} chunks")
                    st.info(f"Total in database: {data['total_chunks']} chunks")
                else:
                    st.error(f"{uploaded_file.name}：Something went wrong. Is the backend running?")

    st.divider()
    if st.button("🗑️ 清除資料庫", use_container_width=True):
        response = requests.delete(f"{BACKEND_URL}/clear")
        if response.status_code == 200:
            st.success("資料庫已清除！")
        else:
            st.error("清除失敗")

    st.divider()
    n_results = st.slider(
        "回答精確度（越高參考越多資料）",
        min_value=1,
        max_value=20,
        value=5
    )

st.divider()

try:
    status = requests.get(f"{BACKEND_URL}/").json()
    st.metric("chunks in database", status["total_chunks"])
except:
    st.error("Backend is not running")

st.divider()

## 這裡是聊天的部分，會把訊息存在 session_state 裡面，這樣就不會因為重新整理頁面而不見了

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("View retrieved chunks", expanded=False):
                for source in msg["sources"]:
                    st.markdown(
                        f"📎 **{source['source']}** - Page {source['page']} "
                        f"*(similarity score: {source['score']})*"
                    )


#if question := st.chat_input("Ask anything about your documents"): 更簡潔的寫法

#建立聊天泡泡泡，使用者輸入的訊息會存在 st.session_state.messages 裡面，這樣就不會因為重新整理頁面而不見了
question = st.chat_input("Ask anything about your documents")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)  

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):

            try:

                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={
                        "question": question,
                        "n_results": n_results
                    }
                )

                data = response.json()

                if "error" in data:

                    answer = f"{data['error']}"
                    sources = []

                else:

                    answer = data["answer"]
                    sources = data["sources"]

            except Exception as e:

                answer = f"Could not reach backened : {e}"
                sources = []

        st.write(answer)

        if sources:
            with st.expander("View retrieved chunks", expanded=False):
                for source in sources:
                    st.markdown(
                        f"📎 **{source['source']}** - Page {source['page']} "
                        f"*(similarity score: {source['score']})*"
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

