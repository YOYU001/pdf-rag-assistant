import os #讀.env檔案
import fitz #讀pdf檔案
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File #必須要有上傳檔案的功能
from fastapi.middleware.cors import CORSMiddleware #前後端連線溝通
from pydantic import BaseModel # 定義資料格式，確保前端傳來的問題格式正確。
import tempfile #暫存檔案
from dotenv import load_dotenv
import chromadb


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源（* = 全部）
    allow_methods=["*"],   # 允許所有請求方式（GET, POST...）
    allow_headers=["*"]   # 允許所有標頭
)

chroma_client = chromadb.PersistentClient(path="./chroma_db") #建立chromadb的儲存路徑


# 先試著取得已經存在的 rag_collection
# 如果不存在（第一次跑）就建立一個新的
def get_or_create_collection():
    try:
        return chroma_client.get_collection("rag_collection")
    except:
        return chroma_client.create_collection("rag_collection")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages= [] 

    for i,page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append((i+1,text))

    doc.close()
    return pages

def chunk_text(text):
    chunks= []
    start = 0

    while start < len(text):
        chunks.append(text[start:start+CHUNK_SIZE])
        start += CHUNK_SIZE-CHUNK_OVERLAP
    return chunks

def embed_text(text):
    response = client.embeddings.create(
        input = text,
        model="text-embedding-3-small"   
    )
    return[item.embedding for item in response.data]


#這是 list comprehension，把所有 item 的 embedding 都收集起來，最後一次回傳整個 list。

# async → 非同步函數，可以同時處理多個請求不會卡住
# file: UploadFile → 接收前端上傳的 PDF 檔案
# File(...) → 這是必填欄位

@app.post("/ingest") #前端設定：BACKEND_URL = "http://localhost:8000" 所以這是進到 ingest的功能而已
async def ingest(file:UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp: #delete=False 是因為windows不能開檔案做事
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    collection = get_or_create_collection()


    pages = extract_text(tmp_path)
    all_chunks = []
    all_metadata = []
    all_ids = []

    chunk_index = collection.count()

    for page_num, page_text in pages:
        for chunk in chunk_text(page_text):
            if len(chunk.strip()) < 50:
                continue
            all_chunks.append(chunk)
            all_metadata.append({"source": file.filename, "page": page_num})
            all_ids.append(f"chunk_{chunk_index}")

            chunk_index += 1

    all_embedding = []
    BATCH_SIZE = 100
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        all_embedding.extend(embed_text(batch)) #extend 是放元素  append 是放 list

#把資料存進 chromadb裡
    collection.add(
        documents=all_chunks,
        embeddings=all_embedding,
        metadatas=all_metadata,
        ids=all_ids
    )

    os.unlink(tmp_path) #已經存在chromadb裡面了 可以刪除暫時檔了

    return {
    "message": f"Ingested {file.filename} successfully",
    "chunks_added": len(all_chunks),
    "total_chunks": collection.count(),
}

#這是要確認前端的資料格式
class AskRequest(BaseModel):
    question:str
    n_results: int = 3 #要找幾個相關的chunk   default是3 也可以自己輸入


@app.post("/ask") #當前端呼叫 /ask 時執行下面的函數。
async def ask(request:AskRequest):
    collection = get_or_create_collection()
    if collection.count == 0:
        return {"error":"No document yet"}
    question_embedding = embed_text([request.question])[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results = request.n_results,
        include=["documents","metadatas","distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []  # 收集 chunk 文字，等等合併給 GPT 看
    sources = []   # 收集來源資訊，等等顯示給使用者看

#zip 是把三個 list 同步配對：
# chunk[0] + meta[0] + dist[0]  → 第一個結果
# chunk[1] + meta[1] + dist[1]  → 第二個結果
# chunk[2] + meta[2] + dist[2]  → 第三個結果

    for chunk, meta, dist in zip(chunks, metadatas, distances):
        context_parts.append(chunk)
        sources.append({
            "source": meta["source"],
            "page": meta["page"],
            "score": round(dist, 4)
        })

# 6. 把 chunks 合併成 context
    context = "\n\n---\n\n".join(chunks)

 # 7. 用新版 OpenAI Responses API 回答
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="""
        Answer the question using ONLY the provided context.
        Always mention which page the answer came from.
        If the answer is not in the context, say "I don't know."
        """,
        
        input=f"""
        Context:
        {context}

        Question:
        {request.question}

        Answer:"""
    )

    return {
        "question": request.question,
        "answer": response.output_text,
        "sources": sources,
    }

## 這只是回報訊息
@app.get("/")
def root():
    collection = get_or_create_collection()
    return {
        "status": "running",
        "total_chunks": collection.count(),
    }


 







        










