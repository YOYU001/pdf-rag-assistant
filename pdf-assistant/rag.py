import os #用來存取環境變數
from dotenv import load_dotenv #用來讀取.env檔案
from numpy import rint
import openai 
from openai import OpenAI #用來存取OpenAI API
import chromadb

load_dotenv() #讀取.env檔案

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) #從環境變數中取得API Key

documents = [
    "The return policy allows customers to return items within 30 days of purchase.",
    "All returns must include the original receipt and packaging.",
    "Refunds are processed within 5 to 7 business days after the item is received.",
    "Items that are damaged or used cannot be returned unless they are defective.",
    "To start a return, visit our website and fill out the return request form."
]

chroma_client = chromadb.Client() 
collection = chroma_client.create_collection(name="yoyo_rag") 

print("Embedding and storing documents...")

for i, doc in enumerate(documents):
    response = client.embeddings.create(  
        model="text-embedding-3-small",
        input=doc
    ) ## 這一段是在把文字變成 embedding 向量。
    embedding = response.data[0].embedding
    collection.add(
        ids=[f"doc_{i}"],
        documents=[doc],
        embeddings=[embedding]
    )
    print(f"Document {i}: {doc[:50]}... embedded and stored.")

print(f"Stored {len(documents)} documents in ChromaDB\n")

#------------step4 ask a question and get the most similar document------------
# 拿這個問題的向量
# 去 ChromaDB 搜尋
# 找最相似的文件

question = "How long do I have to return something?"

question_embedding = client.embeddings.create(
    input=question,
    model="text-embedding-3-small"
).data[0].embedding

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

retirved_chunks = results['documents'][0] ##拿第一個問題的結果 而不是拿第一個元素的意思
print("Retrieved Chunks:")
for chunk in retirved_chunks:
    print(f" -> {chunk}")
print()

#-------------接下來就可以把這些相關的文件丟給 LLM 來回答問題了----------------

context = "\n".join(retirved_chunks) #把搜尋到的 Chunk 合併成一段文字因為等等 GPT 要看

prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:

{context}

Question: {question}

Answer:

"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=prompt
)


answer = response.output_text

print(f"Question: {question}")
print(f"Answer: {answer}")





