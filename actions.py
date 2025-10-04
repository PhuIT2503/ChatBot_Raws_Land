# actions.py
from rasa_sdk import Action, Tracker              # Import lớp Action và Tracker từ rasa_sdk để định nghĩa hành động tùy chỉnh
from rasa_sdk.executor import CollectingDispatcher  # Import CollectingDispatcher để gửi tin nhắn phản hồi lại cho người dùng
from query_qdrant import query_qdrant # Import hàm query_qdrant từ file query_qdrant.py (hàm này xử lý tìm kiếm Qdrant)
import google.generativeai as genai  # Import thư viện Google Generative AI (Gemini API)
import os                            # Import module os để thao tác với biến môi trường
from dotenv import load_dotenv       # Import load_dotenv để đọc file .env

# -----------------------
# Load API key từ .env
# -----------------------
load_dotenv()  # Đọc file .env (tự động nạp các biến môi trường được định nghĩa trong file này)
API_KEY = os.getenv("GEMINI_API_KEY")  # Lấy API key của Gemini từ biến môi trường GEMINI_API_KEY
if not API_KEY:  # Nếu không tìm thấy API key thì báo lỗi
    raise RuntimeError("Không tìm thấy GEMINI_API_KEY trong file .env")

MODEL_NAME = "gemini-2.0-flash"  # Đặt tên model Gemini sẽ sử dụng
genai.configure(api_key=API_KEY)  # Cấu hình thư viện google.generativeai với API key đã nạp


# -----------------------
# Build prompt cho Gemini
# -----------------------
def build_prompt(query, docs):
    """
    Hàm này tạo prompt cho Gemini từ câu hỏi của người dùng (query) và danh sách tài liệu đã tìm được (docs).
    """
    context_texts = [f"- {d['text']}" for d in docs]  # Duyệt qua docs, lấy text trong mỗi tài liệu rồi format thành gạch đầu dòng
    context_str = "\n".join(context_texts)  # Ghép tất cả đoạn văn bản thành một chuỗi (mỗi đoạn trên một dòng)
    return f"""
Bạn là **trợ lý pháp luật chuyên về đất đai**, chỉ giải thích và tóm tắt các văn bản quy phạm pháp luật liên quan đến **đất đai**.

### Quy tắc bắt buộc:
1. Câu trả lời chỉ dựa trên **"Văn bản luật đất đai tham chiếu"** dưới đây. Không suy đoán hay đưa thông tin từ luật khác.
2. Nếu người dùng hỏi về một **Điều/Khoản/Điểm cụ thể** trong luật đất đai:
   - Trả lời đúng nguyên văn nội dung đó (có thể diễn đạt lại dễ hiểu nhưng không được sai ý).
   - Luôn ghi rõ số **Điều, Khoản, Điểm** khi trích dẫn.
3. Nếu người dùng hỏi tổng quát về luật đất đai (ví dụ toàn bộ một luật hoặc một chương):
   - Tổng hợp nhiều đoạn liên quan từ dữ liệu đất đai, sắp xếp logic và dễ hiểu.
4. Nếu không tìm thấy thông tin về đất đai trong dữ liệu:
   - Trả lời: "Không tìm thấy thông tin về đất đai trong dữ liệu hiện có."
   - Có thể đưa ra các đoạn gần nhất liên quan đất đai để tham khảo.
5. Cách viết câu trả lời:
   - Ngắn gọn, mạch lạc (200–250 từ).
   - Trình bày theo gạch đầu dòng có đánh số (1, 2, 3...).
   - Có thể thêm **ví dụ minh họa thực tế** nếu phù hợp.
   - Kết thúc bằng đoạn **Tổng kết ý nghĩa thực tiễn** (nêu giá trị áp dụng).
   - Cuối cùng ghi rõ **Nguồn: Điều/Khoản/Điểm...** đã sử dụng.

---

### Văn bản luật đất đai tham chiếu:
{context_str}

### Câu hỏi:
{query}

### Trả lời:
"""


# -----------------------
# Custom Action
# -----------------------
class ActionQueryRAG(Action):
    """
    Lớp custom action để tích hợp RAG (FAISS + BGE-ReRank) và Gemini trong Rasa.
    """

    def name(self):
        # Tên action (phải khớp với file domain.yml)
        return "action_query_rag"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict):
        """
        Hàm thực thi khi action này được gọi.
        - dispatcher: dùng để gửi tin nhắn phản hồi lại cho user
        - tracker: lưu trữ trạng thái hội thoại (tin nhắn trước đó, slot, intent…)
        - domain: chứa định nghĩa intent, entity, slot, action trong Rasa
        """

        # Lấy câu hỏi cuối cùng mà người dùng vừa nhập
        query = tracker.latest_message.get("text")

        try:
            docs = query_qdrant(query, top_k=10, top_k_rerank=5)  # Gọi hàm query_qdrant để tìm các đoạn văn bản liên quan từ Qdrant
        except Exception as e:
            # Nếu lỗi khi tìm kiếm RAG -> báo lỗi cho người dùng
            dispatcher.utter_message(text=f"Lỗi RAG: {e}")
            return []

        if not docs:
            # Nếu không có tài liệu nào phù hợp -> báo không tìm thấy
            dispatcher.utter_message(text="Không tìm thấy thông tin trong dữ liệu hiện có.")
            return []

        # Xây dựng prompt cho Gemini từ query + danh sách docs
        prompt = build_prompt(query, docs)

        try:
            # Khởi tạo model Gemini
            model = genai.GenerativeModel(MODEL_NAME)
            # Gửi prompt cho Gemini để sinh câu trả lời
            response = model.generate_content(prompt, generation_config={
                "temperature": 0.5,       # Nhiệt độ: điều chỉnh độ sáng tạo (0.0 = rất chính xác, 1.0 = sáng tạo hơn)
                "max_output_tokens": 512, # Giới hạn số token đầu ra
            })
            # Lấy văn bản trả lời từ response
            answer = getattr(response, "text", None)
        except Exception as e:
            # Nếu lỗi khi gọi Gemini -> báo cho user
            dispatcher.utter_message(text=f"Lỗi gọi Gemini: {e}")
            return []

        if not answer:
            # Nếu Gemini không trả lời -> báo lỗi
            dispatcher.utter_message(text="Gemini không trả về câu trả lời.")
        else:
            # Gửi câu trả lời về cho user
            dispatcher.utter_message(text=answer)

        return []
