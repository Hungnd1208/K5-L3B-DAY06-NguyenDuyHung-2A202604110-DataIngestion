# Báo cáo cá nhân — Day 6 (Data Ingestion Engineering)

**Học viên:** Nguyễn Duy Hưng 
**MSSV:** 2A202604110  
**Lớp:** K5 - L3B  

---

## 1. Dữ liệu và pipeline

* **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:**
  * *Chủ đề:* Báo cáo lý thuyết Chuyên sâu về Mạng Nơ-ron Tích chập (Convolutional Neural Network - CNN) và Deep Learning.
  * *Nguồn & Bản quyền:* Tài liệu học tập và nghiên cứu nội bộ thuộc VinUni Lab (`starter/data/BÁO CÁO LÝ THUYẾT CNN.pdf`), được phép sử dụng cho mục đích giáo dục và nghiên cứu học thuật.
  * *Số trang & Quy tắc tính:* Tài liệu có tổng cộng 32 trang vật lý. Áp dụng quy tắc tính trang: chỉ tính các trang có nội dung kiến thức chuyên môn hữu ích (từ Trang 1 đến Trang 31, đạt **31 trang nội dung** $\ge 30$ trang theo yêu cầu, tổng cộng **46.991 ký tự**). Trang 32 chứa danh mục tài liệu tham khảo (References) được chủ động loại bỏ khỏi corpus nạp vào vector store. Chi tiết lưu tại `data/manifest.json`.

* **Chuẩn hóa, lỗi và giới hạn:**
  * Sử dụng thư viện `pypdf` để bóc tách text từng trang vật lý. Chuẩn hóa ký tự Unicode theo chuẩn **NFC** (đảm bảo tính nhất quán của dấu tiếng Việt), chuẩn hóa các ký tự xuống dòng (`\r\n` $\rightarrow$ `\n`), loại bỏ khoảng trắng dư thừa ở đầu/cuối dòng trong khi vẫn bảo toàn ranh giới các đoạn văn bản (`\n\n`).
  * *Giới hạn:* Trang 21 chứa bảng so sánh cơ chế lan truyền ngược qua Pooling; khi trích xuất text thuần qua PDF, cấu trúc dạng bảng bị mất format hàng/cột, gây suy giảm độ mạch lạc ngữ nghĩa của các ô trong bảng.

* **Embedding model/provider/version/số chiều, vector store/version, metric, lệnh chạy và chi phí:**
  * *Embedding Model:* **Google Gemini API** (`models/gemini-embedding-001`), hỗ trợ đa ngôn ngữ chất lượng cao; sử dụng kỹ thuật Matryoshka Representation Learning (MRL) để rút gọn vector về **768 dimensions**.
  * *Vector Store:* **PostgreSQL 16** với extension **pgvector 0.8.7** (triển khai qua Docker Compose cục bộ, cổng `55433`).
  * *Khoảng cách đo lường (Metric):* **Cosine Distance** (`ORDER BY embedding <=> query_vector::vector ASC`).
  * *Lệnh chạy tái lập:* 
    ```bash
    python -m src.run
    ```
  * *Chi phí:* **$0.00** (sử dụng gói Gemini Free Tier qua Google AI Studio).

* **Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn:**
  * *Metadata bắt buộc:* `document_id` (`"bao-cao-cnn"`), `source` (`"starter/data/BÁO CÁO LÝ THUYẾT CNN.pdf"`), `page` (số trang vật lý 1–31).
  * *Trường lọc bổ sung:* `chapter` (nhận các giá trị: `loi_mo_dau`, `chuong_1`, `chuong_2`, `chuong_3`, `chuong_4`, `chuong_5`, `ket_luan`). Nguồn giá trị được ánh xạ chính xác từ mục lục và tiêu đề thực tế trong tài liệu.
  * *Lý do chọn:* Tài liệu được cấu trúc phân chia rành mạch theo các chương lớn. Khi AI agent nhận câu hỏi có phạm vi cụ thể (ví dụ: "Trong Chương 4..."), trường `chapter` cho phép lọc nhanh qua mệnh đề SQL `WHERE chapter = %s`, ngăn chặn việc tìm nhầm sang các chương có từ khóa trùng lặp nhưng khác ngữ cảnh.

---

## 2. Cấu hình và benchmark

* **Ba cấu hình A / B / C:**
  * **Chiến lược A (Fixed-size):** Cửa sổ cắt cố định `size = 500` ký tự, bước nhảy `overlap = 50` ký tự. Lưu metadata cơ bản. Tìm kiếm Cosine thuần, **không dùng bộ lọc**.
  * **Chiến lược B (By Paragraph):** Cắt theo cấu trúc đoạn văn bản tự nhiên (`\n\n`), gom các đoạn nhỏ và khống chế `max_size = 700` ký tự. Cùng metadata như A, **không dùng bộ lọc**. (So sánh A – B: Đánh giá tác động thuần túy của kỹ thuật chunking).
  * **Chiến lược C (Paragraph + Filter):** Cùng cách chia chunk và vector như B. Điểm khác biệt duy nhất: **kích hoạt SQL metadata filter** (`WHERE chapter = %s`) khi câu hỏi có chứa điều kiện lọc. (So sánh B – C: Đánh giá tác động thuần túy của bộ lọc metadata).

* **Bộ benchmark 20 câu:**
  * Được thiết kế và kiểm chứng trích dẫn (`evidence_quote`) 100% khớp văn bản trước khi chạy thử nghiệm, lưu tại `benchmark/benchmark.json`.
  * **Phân chia:** **15 câu Dev** (`q01` – `q15`) và **5 câu Test holdout độc lập** (`q16` – `q20`).
  * **Câu hỏi Metadata Filter (3 câu):** Có 2 câu trong Dev (`q13` lọc theo Chương 3, `q14` lọc theo Chương 4) và 1 câu trong Test (`q17` lọc theo Chương 4).

* **Tiêu chí chọn chiến lược đã chốt trước khi xem test:**
  * Ưu tiên số 1: Tối đa hóa **Evidence Hit@3** trên tập Dev.
  * Ưu tiên số 2: Độ mạch lạc của chunk (**Chunk Coherence**) $\ge 1.5/2.0$.
  * Ưu tiên số 3: Tối ưu tài nguyên (số lượng chunk vừa phải, thời gian và chi phí thấp).

---

## 3. Kết quả

### Bảng 1: Hiệu năng định lượng tự động (Retrieval Accuracy & Resource)

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|---|---|---|---|---|
| **A** (Fixed-size 500/50) | Dev | 10/15 (66.67%) | 13/15 (86.67%) | 116 | 123.81s | $0.00 |
| | Test | 5/5 (100.0%) | 5/5 (100.0%) | | | |
| **B** (By Paragraph 700) | Dev | 11/15 (73.33%) | 14/15 (93.33%) | 109 | 43.20s | $0.00 |
| | Test | 4/5 (80.0%) | 5/5 (100.0%) | | | |
| **C** (Paragraph + Filter) | Dev | 11/15 (73.33%) | 14/15 (93.33%) | 109 | 43.20s | $0.00 |
| | Test | 4/5 (80.0%) | 5/5 (100.0%) | | | |

### Bảng 2: Đánh giá chất lượng thủ công (Manual Assessment — 5 Khía cạnh)

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|:---:|:---:|:---:|:---:|
| **A** | 1.1 / 2.0 (n=10) | 15/15 (100%) | N/A | [0: 0, 1: 0, 2: 5] $\rightarrow$ **2.0 / 2.0** |
| **B** | 2.0 / 2.0 (n=10) | 15/15 (100%) | N/A | [0: 0, 1: 0, 2: 5] $\rightarrow$ **2.0 / 2.0** |
| **C** | 2.0 / 2.0 (n=10) | 15/15 (100%) | **3/3 (100%)** | [0: 0, 1: 0, 2: 5] $\rightarrow$ **2.0 / 2.0** |

* **Đường dẫn artifacts:**
  * File trace chi tiết từng câu: `results/traces_strategy_A.json`, `results/traces_strategy_B.json`, `results/traces_strategy_C.json`.
  * Nhật ký thí nghiệm: `artifacts/experiment_log.csv`.
  * Chi tiết bảng chấm thủ công: `results/manual_audit.md`.
  * *Cách chọn mẫu:* Coherence được chấm trên 10 vị trí mở đầu của 10 trang đại diện (P02, P04, P06, P08, P09, P10, P11, P17, P18, P27). Grounding và Metadata được thẩm định trên toàn bộ 5 câu test $\times$ top-3 (15 records).

---

## 4. Phân tích và kết luận

* **So sánh A–B và B–C:**
  * **So sánh A – B (Chunking):** Strategy B vượt trội toàn diện so với A. Trên tập Dev, Hit@1 tăng từ 66.67% lên 73.33% và Hit@3 tăng từ 86.67% lên 93.33%. Về mặt chất lượng, Coherence của B đạt điểm tuyệt đối 2.0/2.0 trong khi A chỉ đạt 1.1/2.0 do cửa sổ cắt cứng thường xuyên xẻ đôi câu văn giữa chừng. Hơn nữa, B tạo ra 109 chunks (ít hơn 7 chunks so với A), giúp tiết kiệm tài nguyên lưu trữ và giảm thời gian embedding.
  * **So sánh B – C (Metadata Filtering):** Trên tập dữ liệu hẹp hiện tại, B và C có điểm Hit tương đương vì các câu hỏi có ngữ nghĩa rất đặc trưng. Tuy nhiên, Strategy C mang lại giá trị bảo đảm an toàn dữ liệu: đạt tỷ lệ tuân thủ filter tuyệt đối **3/3 (100%)**, bảo đảm kết quả trả về không bao giờ bị ô nhiễm bởi các đoạn văn ở chương khác.
  * **2 ví dụ sai điển hình (Failure Analysis):**
    1. *Câu `q04` (Trang 8) ở Strategy A:* Hỏi về hàm kích hoạt ReLU giúp tăng tốc độ huấn luyện. Strategy A bị trượt vì cửa sổ cắt cứng 500 ký tự đã chia cắt từ khóa "ReLU" và câu giải thích sang 2 chunk khác nhau, khiến vector similarity bị giảm sút. Ngược lại, Strategy B gom trọn đoạn văn nên đạt Top-1 chính xác.
    2. *Câu `q08` (Trang 21) ở Strategy B:* Hỏi về sự khác biệt tham số giữa Pooling và Conv. Cả A và B đều gặp khó khăn vì nội dung gốc nằm trong một bảng biểu (Table). Khi trích xuất văn bản thuần qua `pypdf`, các hàng của bảng bị đứt thành các dòng rời rạc, làm mô hình embedding mất đi cấu trúc đối sánh giữa 2 cột.

* **Chiến lược được chọn, đánh đổi và điều kiện áp dụng:**
  * **Chiến lược tối ưu được chọn là:** **Chiến lược C (Structure-aware Chunking + Metadata Filter)**.
  * *Đánh đổi (Trade-off):* Phải đầu tư công sức xây dựng logic tách đoạn và gán nhãn metadata chuẩn xác ở khâu tiền xử lý (ingestion); câu hỏi query cần có thêm ngữ cảnh hoặc bộ trích xuất entity để cung cấp filter.
  * *Điều kiện áp dụng:* Thích hợp nhất cho tài liệu kỹ thuật, sổ tay quy chế, tài liệu pháp lý có cấu trúc chương/mục rõ ràng.

* **Hạn chế của dữ liệu, nhãn, test nhỏ, đo thời gian/chi phí:**
  * Tập test gồm 5 câu là quy mô mẫu nhỏ, chỉ mang tính chất minh họa và kiểm tra độc lập (sanity check), chưa đủ độ lớn thống kê để khái quát hóa toàn diện.
  * Thời gian đo ở Strategy A bị kéo dài (123.81s) do gặp phải cơ chế Rate Limit (429 Too Many Requests) của gói Gemini Free Tier, phải chờ 35 giây để reset quota. Chi phí thực tế là $0.00.

* **Đề xuất cải tiến nếu làm lại hoặc mở rộng:**
  1. Xây dựng bộ parser chuyên dụng cho bảng biểu (dùng `pdfplumber` hoặc `marker`) để giữ nguyên định dạng Markdown Table, giải quyết triệt để lỗi mất ngữ cảnh như ở câu `q08`.
  2. Bổ sung kỹ thuật **Hybrid Retrieval** (kết hợp vector dense của Gemini với từ khóa thưa BM25 và bộ tái xếp hạng Reranker) để nâng cao hơn nữa độ chính xác cho các câu hỏi tra cứu thuật ngữ hiếm.
