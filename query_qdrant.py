# query_qdrant_fast_bge_batch.py

import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from qdrant_client import QdrantClient
from dotenv import load_dotenv  # Để đọc biến môi trường từ file .env

# ---------------------------
# Cấu hình
# ---------------------------
load_dotenv()  # Đọc các biến môi trường từ file .env

# Tên model embedding BGE-M3
BGE_MODEL = "BAAI/bge-m3"
# Tên model reranker BGE-Reranker
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Giới hạn độ dài token của embedding và reranker
MAX_LENGTH = 512
MAX_LENGTH_RERANK = 512

# Kích thước batch cho embed và rerank
BATCH_SIZE_EMBED = 32
BATCH_SIZE_RERANK = 32

# Thông tin Qdrant
QDRANT_COLLECTION = "laws_land"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Kiểm tra biến môi trường Qdrant
if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Thiếu QDRANT_URL hoặc QDRANT_API_KEY trong biến môi trường!")

# Chọn thiết bị: GPU nếu có, ngược lại CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load models (chỉ 1 lần)
# ---------------------------
print("Loading BGE-M3 embedding model...")
tokenizer_bge = AutoTokenizer.from_pretrained(BGE_MODEL)  # Load tokenizer
model_bge = AutoModel.from_pretrained(BGE_MODEL).to(DEVICE).eval()  # Load model và set eval mode

# ---------------------------
# Hàm batch embedding
# ---------------------------
print("Warming up BGE-M3 embedding model...")
@torch.no_grad()  # Không tính gradient để tiết kiệm RAM/CPU
def embed_texts(texts):
    """
    Nhận danh sách texts -> trả về vector embedding normalized
    """
    # Nếu text rỗng thì thay bằng [PAD]
    texts = [t if t and str(t).strip() else "[PAD]" for t in texts]
    embeddings_parts = []  # Lưu embedding từng batch

    # Xử lý theo batch
    for i in range(0, len(texts), BATCH_SIZE_EMBED):
        batch = texts[i:i+BATCH_SIZE_EMBED]

        # Tokenize batch
        enc = tokenizer_bge(batch, padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
        # Chuyển tensor lên thiết bị
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        # Lấy output từ model
        out = model_bge(**enc)
        last_hidden = out.last_hidden_state  # shape: (batch, seq_len, hidden_dim)

        # Tạo mask để ignore padding
        mask = enc["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)  # Tính tổng theo token
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)  # Số token thực
        mean_pooled = summed / counts  # Lấy trung bình

        arr = mean_pooled.cpu().numpy()  # Chuyển về numpy
        # Normalize vector để cosine similarity = dot product
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        embeddings_parts.append(arr.astype("float32"))

    return np.vstack(embeddings_parts)  # Ghép tất cả batch lại

# Warm-up model embedding với 1 câu dummy
_ = embed_texts(["Warm-up query"])

# ---------------------------
# Load BGE-Reranker
# ---------------------------
print("Loading BGE-Reranker model...")
tokenizer_rerank = AutoTokenizer.from_pretrained(RERANK_MODEL)  # Load tokenizer reranker
model_rerank = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL).to(DEVICE).eval()  # Load model

# ---------------------------
# Hàm batch rerank
# ---------------------------
@torch.no_grad()
def rerank(query: str, docs: list):
    """
    Nhận query + danh sách docs -> trả về docs được sắp xếp theo score cao xuống thấp
    """
    if not docs:
        return []

    scores = []  # Lưu score từng doc
    texts = [d["text"] for d in docs]

    # Xử lý theo batch
    for i in range(0, len(texts), BATCH_SIZE_RERANK):
        batch_texts = texts[i:i+BATCH_SIZE_RERANK]

        # Tokenize batch
        inputs = tokenizer_rerank([query]*len(batch_texts),
                                   batch_texts,
                                   padding=True,
                                   truncation=True,
                                   max_length=MAX_LENGTH_RERANK,
                                   return_tensors="pt").to(DEVICE)

        # Tính logits
        out = model_rerank(**inputs)
        batch_scores = out.logits.squeeze().tolist()  # Chuyển về list
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]  # Nếu chỉ 1 phần tử
        scores.extend(batch_scores)

    # Sắp xếp docs theo score giảm dần
    ranked_docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked_docs

# Warm-up reranker với 1 câu dummy
_ = rerank("Warm-up query", [{"text": "Warm-up doc"}])

# ---------------------------
# Query Qdrant + rerank
# ---------------------------
def query_qdrant(text, top_k=10, top_k_rerank=5):
    """
    Nhận query -> tìm top_k docs từ Qdrant -> rerank -> trả về top_k_rerank docs
    """
    # Khởi tạo client Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vec = embed_texts([text])[0]  # Embed query

    # Truy vấn Qdrant
    hits = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vec,
        limit=top_k
    ).points

    # Lấy payload text + metadata
    docs = [{"text": h.payload.get("text", ""), "metadata": h.payload.get("metadata", {})} for h in hits]

    # Rerank bằng model
    reranked_docs = rerank(text, docs)
    return reranked_docs[:top_k_rerank]

# ---------------------------
# Chạy thử
# ---------------------------
if __name__ == "__main__":
    query_text = "Cho tôi văn bản pháp luật về thay đổi quyền sử dụng đất"
    results = query_qdrant(query_text, top_k=10, top_k_rerank=5)

    # In kết quả
    for i, r in enumerate(results, 1):
        print(f"Rank {i} | text={r['text'][:200]}...")  # In 200 ký tự đầu
        print("metadata:", r["metadata"])
        print("----")
