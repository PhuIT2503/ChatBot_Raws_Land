# Hướng dẫn chạy Chatbot Rasa

Cài thư viện

```bash
pip install -r requirements.txt
```

Cài hai thư viện sentence-transformers và transformers để có thể tải model embedding và xử lý NLP bằng Hugging Face
```bash
pip install -U sentence-transformers transformers
```

Bước 1: Chạy actions Server
Mở **Terminal 1** và chạy lệnh sau:
```bash
rasa run actions --port 5055
```
Bước 2:
Mở **Terminal 2** và chạy lệnh sau:
```bash
rasa run --enable-api --cors "*"
```
Bước 3: Mở giao diện Web

Mở file index.html của dự án.

Chọn Go Live (nếu dùng VS Code với Live Server) để mở giao diện chatbot trên trình duyệt.

Bước 4: Sử dụng Chatbot

Nhập tin nhắn vào ô chat trên giao diện web.