import os #用來存取環境變數
from dotenv import load_dotenv #用來讀取.env檔案
from numpy import rint
import fitz  ## PyMuPDF 用來處理PDF檔案
import openai 
from openai import OpenAI #用來存取OpenAI API
import chromadb

load_dotenv() #讀取.env檔案

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) #從環境變數中取得API Key

PDF_PATH = r"C:\my_Purdue\my_RAG\google-Sandra_Liou_resume.pdf" # PDF檔案路徑

CHUNK_SIZE = 500 # 每個chunk的字數限制
CHUNK_OVERLAP = 50 # chunk之間的重疊字數

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text = text.strip()
        if text:
            pages.append((page_num + 1, text))
    doc.close()
    print(f"Extracted text from {len(pages)} pages")
    return pages

def chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return [item.embedding for item in response.data]


def index_pdf(pdf_path):
    pages = extract_text_from_pdf(pdf_path)

    all_chunks = []
    all_metadata = []
    all_ids = []
    chunk_index = 0

    for page_num, page_text in pages:
        chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue

            all_chunks.append(chunk)
            all_metadata.append({
                "source": os.path.basename(pdf_path),
                "page": page_num,
                "chunk": chunk_index,
            })
            all_ids.append(f"chunk_{chunk_index}")
            chunk_index += 1

    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]

        embeddings = embed_text(batch)

        all_embeddings.extend(embeddings)

        print(f"Embedded batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="pdf_collection")

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadata,
        ids=all_ids,
    )

    print(f"\nDone! Indexed {len(all_chunks)} chunks from {pdf_path}")

    return collection

 # 1. 把問題轉成 embedding
def ask(collection, question, n_results=3):
    question_embedding = embed_text([question])[0]
  
  # 2. 去 ChromaDB 找最相關的 chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

 # 3. 取出搜尋結果
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # 4. 印出問題
    print(f"\nQuestion: {question}")
    
     # 5. 印出 ChromaDB 找到的 chunks
    print("\nRetrieved chunks:")
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        print(f"  [Page {meta['page']} | score: {dist:.3f}] {chunk[:100]}...")

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
        {question}
        """
    )

    # 8. 印出答案
    print("\nAnswer:")
    print(response.output_text)


if __name__ == "__main__":
    collection = index_pdf(PDF_PATH)
#是一種防護機制。直接執行這個檔案時才會跑，別人 import 你的函數時不會自動執行。

    ask(collection, "Sandra Liou 做過哪些專案")
    ask(collection, "Sandra Liou 的專長是什麼") 

