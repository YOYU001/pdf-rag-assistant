# Chat with your PDF

一個基於 RAG（Retrieval Augmented Generation）技術的 PDF 問答助手，支援中英文提問。

## 功能

- 上傳一份或多份 PDF 檔案
- 自動將 PDF 切成 chunks 並儲存至向量資料庫
- 用中文或英文提問，AI 從 PDF 內容中找出答案
- 顯示答案來源（哪一頁、相似度分數）
- 可調整參考 chunk 數量（回答精確度）
- 一鍵清除資料庫

## 技術架構

| 部分 | 技術 |
|------|------|
| 前端 | Streamlit |
| 後端 | FastAPI |
| 向量資料庫 | ChromaDB |
| Embedding 模型 | text-embedding-3-small |
| 語言模型 | GPT-4o-mini |

## 安裝與執行

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 設定 API Key

建立 `.env` 檔案：

```
OPENAI_API_KEY=你的金鑰
```

### 3. 啟動後端

```bash
uvicorn backened:app --reload
```

後端會在 `http://localhost:8000` 執行。

### 4. 啟動前端

```bash
streamlit run frontend.py
```

前端會在 `http://localhost:8501` 執行。

## 使用方式

1. 在左側側邊欄上傳 PDF 檔案（可多選）
2. 點擊 **Ingest PDF** 按鈕
3. 在下方輸入框輸入問題（中英文皆可）
4. 展開 **View retrieved chunks** 可查看參考來源

## 注意事項

- PDF 需為可選取文字的格式（掃描圖片 PDF 無法讀取）
- 重新上傳同一份 PDF 不會重複新增 chunks
- 清除資料庫後需重新 ingest PDF
