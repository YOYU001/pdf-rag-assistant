import os #用來存取環境變數
from dotenv import load_dotenv #用來讀取.env檔案
from numpy import rint
import openai 
from openai import OpenAI #用來存取OpenAI API
load_dotenv() #讀取.env檔案

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) #從環境變數中取得API Key


sentences = [
    "The dog ran across the park.",
    "A puppy sprinted through the garden.",  # Similar to #1
    "The quarterly earnings report is due.", # Very different
]

embeddings = []

for sentence in sentences:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=sentence
    )
    embedding = response.data[0].embedding
    embeddings.append(embedding)
    print(f"Sentence: {sentence}")
    print(f"Embedding: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")# 取前三筆資料 並且到小數點後第四位
    print(f"Length: {len(embedding)} numbers\n")

def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2)) # 其實用 cosine similarity 會更好，但這裡簡單示範 dot product

# def cosine_similarity(vec1, vec2):
#     dot = dot_product(vec1, vec2)
#     norm1 = sum(a * a for a in vec1) ** 0.5
#     norm2 = sum(b * b for b in vec2) ** 0.5
#     return dot / (norm1 * norm2)

# 先算兩個向量原本有多像，
# 再除掉它們各自的長度，
# 最後只留下「方向像不像」。

sim_1_2 = dot_product(embeddings[0], embeddings[1])
sim_1_3 = dot_product(embeddings[0], embeddings[2])

print("-" * 50)
print(f"Similarity: sentence 1 vs sentence 2 → {sim_1_2:.4f}  (similar meaning)")
print(f"Similarity: sentence 1 vs sentence 3 → {sim_1_3:.4f}  (different meaning)")
print()

print("Notice: similar sentences score HIGHER. That's embeddings working.")

