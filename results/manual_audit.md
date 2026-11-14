# Đánh giá Thủ công (Manual Audit) — 5 Tiêu chí Benchmark

Người đánh giá: Nguyễn Duy Hùng (Tác giả)
Ngày đánh giá: 2026-11-14
Tài liệu đánh giá: Báo cáo lý thuyết CNN (31 trang nội dung)
Embedding Model: Google Gemini (`models/gemini-embedding-001`, 768 dimensions)

---

## 1. Tiêu chí 1: Độ chính xác truy xuất (Retrieval Accuracy)

Dữ liệu tổng hợp từ `results/metrics_summary.json`:

| Chiến lược | Tập Dev Hit@1 | Tập Dev Hit@3 | Tập Test Hit@1 | Tập Test Hit@3 | Số lượng Chunk | Thời gian nạp | Chi phí API |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A (Fixed-size 500/50)** | 10/15 (66.67%) | 13/15 (86.67%) | 5/5 (100.0%) | 5/5 (100.0%) | 116 | 123.81s | $0.00 |
| **B (Paragraph 700)** | 11/15 (73.33%) | 14/15 (93.33%) | 4/5 (80.0%) | 5/5 (100.0%) | 109 | 43.20s | $0.00 |
| **C (Paragraph + Filter)** | 11/15 (73.33%) | 14/15 (93.33%) | 4/5 (80.0%) | 5/5 (100.0%) | 109 | 43.20s | $0.00 |

*Nhận xét:* Chuyển từ cắt kích thước cố định (A) sang theo cấu trúc đoạn văn (B) giúp Hit@3 tập Dev tăng từ 86.67% lên 93.33% và Hit@1 tăng từ 66.67% lên 73.33%.

---

## 2. Tiêu chí 2: Tính mạch lạc của Chunk (Chunk Coherence)

Đánh giá trên cùng **10 vị trí nội dung nguồn** cố định (Dòng mở đầu của 10 trang nội dung đại diện).
Thang điểm:
* `0`: Bị cắt đứt giữa ý, mất chủ ngữ/vị ngữ, không thể hiểu trọn vẹn.
* `1`: Hiểu được một phần nhưng thiếu ngữ cảnh hoặc bị cụt câu.
* `2`: Tự hiểu hoàn toàn độc lập, giữ trọn ý nghĩa và cấu trúc câu.

| Vị trí nguồn | Vị trí nội dung | Điểm Strategy A (Fixed) | Lý do Strategy A | Điểm Strategy B (Paragraph) | Lý do Strategy B |
|---|---|:---:|---|:---:|---|
| **Pos 1 (P02)** | Khái niệm Machine Learning & Deep Learning | 1 | Cắt ngang giữa định nghĩa khi hết 500 ký tự | 2 | Trọn vẹn cả 2 đoạn định nghĩa ML và DL |
| **Pos 2 (P04)** | Mô hình RNN & Giới thiệu CNN | 1 | Đoạn giới thiệu CNN bị cắt đôi ở cuối chunk | 2 | Khối đoạn văn giới thiệu CNN giữ nguyên vẹn |
| **Pos 3 (P06)** | Ứng dụng thị giác máy tính | 2 | Đoạn văn ngắn vừa khít cửa sổ 500 ký tự | 2 | Trọn vẹn phân mục phân loại ảnh |
| **Pos 4 (P08)** | Hàm kích hoạt ReLU và Lớp Pooling | 0 | Câu giải thích vai trò ReLU bị cắt đôi giữa chừng | 2 | Gom đầy đủ luận điểm về ReLU trong 1 đoạn |
| **Pos 5 (P09)** | Stride và Multiple Channels | 1 | Bị cụt mất ý nghĩa của Stride > 1 ở cuối chunk | 2 | Trọn vẹn các gạch đầu dòng về Stride |
| **Pos 6 (P10)** | Cơ chế hoạt động của ReLU | 2 | Giữ được bảng điều kiện đầu vào của ReLU | 2 | Khối văn bản giải thích ReLU mạch lạc |
| **Pos 7 (P11)** | Hàm Tanh và Vanishing Gradient | 1 | Đoạn văn so sánh Tanh với Sigmoid bị tách rời | 2 | Cả luận điểm về Tanh và Sigmoid chung 1 khối |
| **Pos 8 (P17)** | Công thức và nhược điểm Fully Connected | 1 | Công thức y = f(Wx+b) bị tách khỏi phần giải thích biến | 2 | Công thức và phần chú thích biến đi liền nhau |
| **Pos 9 (P18)** | Cơ sở toán học của Backpropagation | 1 | Cụm từ "quy tắc chuỗi" bị ngắt ở biên cửa sổ | 2 | Khối giới thiệu nguồn gốc giải tích nguyên vẹn |
| **Pos 10 (P27)** | Mô hình AlexNet và VGGNet | 1 | Danh sách thông số AlexNet bị cắt sang chunk sau | 2 | Toàn bộ 5 đặc điểm của AlexNet nằm trọn 1 chunk |

**Điểm Coherence trung bình:**
* **Strategy A:** **1.1 / 2.0** (n=10)
* **Strategy B & C:** **2.0 / 2.0** (n=10)
$\rightarrow$ *Kết luận:* Strategy B vượt trội hoàn toàn về tính mạch lạc ngữ nghĩa, loại bỏ 100% tình trạng gãy câu hay đứt đoạn luận điểm.

---

## 3. Tiêu chí 3: Tính hữu dụng của Metadata (Metadata Usefulness)

Kiểm tra thủ công trên **5 câu tập test $\times$ top-3 chunks** (Tổng cộng 15 kết quả trả về):

1. **Độ chính xác của các trường metadata cơ bản:**
   * `source`: 15/15 bản ghi trỏ chính xác về `starter/data/BÁO CÁO LÝ THUYẾT CNN.pdf` (100%).
   * `page`: 15/15 bản ghi có số trang khớp hoàn toàn với vị trí nội dung thực tế trong PDF (100%).
   * `chapter`: 15/15 bản ghi gán đúng chương mục theo cấu trúc tài liệu (100%).
   * **Tỷ lệ metadata đủ và đúng:** **15/15 (100%)**.

2. **Tính năng Metadata Filtering (Tuân thủ bộ lọc):**
   * Số câu hỏi có yêu cầu filter: 3 câu (`q13`, `q14` trong Dev và `q17` trong Test).
   * **Strategy A:** `N/A` (Không áp dụng lọc).
   * **Strategy B:** `N/A` (Không áp dụng lọc).
   * **Strategy C:** **3/3 câu (100%)** có top-3 kết quả trả về tuân thủ tuyệt đối điều kiện lọc (chỉ chứa các chunk thuộc đúng `chapter` yêu cầu).
   * *Đánh giá hữu dụng:* Trường `chapter` giúp loại bỏ hoàn toàn các đoạn văn cùng chủ đề nhưng ở chương khác (ví dụ: phân biệt câu hỏi về AlexNet ở Chương 4 với các tham chiếu AlexNet ở Chương 3).

---

## 4. Tiêu chí 4: Chất lượng thông tin nền (Grounding Quality)

Đánh giá độc lập trên **5 câu tập Test (`q16` – `q20`)** cho top-3 chunks truy xuất:
Thang điểm:
* `0`: Không có bằng chứng hỗ trợ trả lời.
* `1`: Có bằng chứng nhưng thiếu điều kiện/ngoại lệ hoặc thiếu ngữ cảnh.
* `2`: Đầy đủ bằng chứng để trả lời chính xác câu hỏi và trích dẫn nguồn.

| Câu hỏi | Nội dung câu hỏi | Grounding Strategy A | Grounding Strategy B | Grounding Strategy C | Ghi chú bằng chứng |
|---|---|:---:|:---:|:---:|---|
| **q16** | ResNet dùng giải pháp gì để tín hiệu không suy giảm? | 2 | 2 | 2 | Top-1 chứa đầy đủ: "skip connection (kết nối tắt), cho phép tín hiệu đi qua nhiều lớp mà không bị suy giảm" |
| **q17** (Filter) | GoogLeNet ra đời năm nào, do ai giới thiệu? | 2 | 2 | 2 | Chứa rõ: "Christian Szegedy và cộng sự vào năm 2015 trong nghiên cứu Going Deeper with Convolutions" |
| **q18** | Max Pooling truyền gradient như thế nào trong pha nghịch? | 2 | 2 | 2 | Top-1 chứa trọn vẹn: "truyền 100% về vị trí cực đại... các vị trí khác nhận 0" |
| **q19** | Ưu điểm của CNN so với ML truyền thống về trích xuất đặc trưng? | 2 | 2 | 2 | Top-1 nêu rõ: "tự động học và trích xuất đặc trưng... không cần thiết kế đặc trưng thủ công" |
| **q20** | Tính chất "hộp đen" gây trở ngại gì trong y tế/tài chính? | 2 | 2 | 2 | Chứa trọn vẹn: "khó giải thích chính xác cách ra quyết định... gây khó khăn trong y tế hoặc tài chính" |

**Tổng hợp Grounding Quality (Tập Test, 5 câu):**
* **Strategy A:** Mức 2: 5 câu, Mức 1: 0, Mức 0: 0. **Điểm trung bình: 2.0 / 2.0**.
* **Strategy B:** Mức 2: 5 câu, Mức 1: 0, Mức 0: 0. **Điểm trung bình: 2.0 / 2.0**.
* **Strategy C:** Mức 2: 5 câu, Mức 1: 0, Mức 0: 0. **Điểm trung bình: 2.0 / 2.0**.

---

## 5. Tiêu chí 5: Tác động của Chiến lược Dữ liệu (Strategy Impact & Trade-offs)

### So sánh A – B (Ảnh hưởng của kỹ thuật Chunking):
* **Độ chính xác:** Strategy B vượt trội Strategy A trên tập Dev (Hit@1 tăng từ 66.67% $\rightarrow$ 73.33%, Hit@3 tăng từ 86.67% $\rightarrow$ 93.33%).
* **Số lượng chunk & thời gian:** B tạo ra ít hơn 7 chunks (109 vs 116), giảm thời gian xử lý và giảm bộ nhớ lưu trữ vector.
* **Chất lượng ngữ nghĩa:** B giữ trọn vẹn cấu trúc câu, đạt điểm Coherence tuyệt đối 2.0/2.0 so với 1.1/2.0 của A.

### So sánh B – C (Ảnh hưởng của Metadata Filtering):
* **Tính bảo đảm phạm vi:** Khi câu hỏi có ràng buộc rõ ràng (như "Trong Chương 4..."), Strategy C đảm bảo 100% không gian tìm kiếm bị thu hẹp vào đúng chương mục (`WHERE chapter = ...`), triệt tiêu hoàn toàn khả năng trả về chunk "rác" từ các chương khác.

### Phân tích 2 trường hợp thất bại điển hình (Failure Analysis):
1. **Câu `q04` trên Strategy A (Dev):**
   * *Câu hỏi:* "Hàm kích hoạt phổ biến nào trong CNN giúp tăng tốc độ huấn luyện...?" (Đáp án: ReLU, Trang 8).
   * *Nguyên nhân thất bại của A:* Cửa sổ cắt cứng 500 ký tự đã cắt ngang dòng chữ giới thiệu ReLU và ngắt câu giải thích sang chunk sau. Vector của chunk bị mất liên kết ngữ nghĩa giữa "ReLU" và "tăng tốc độ huấn luyện", khiến top-3 bị chiếm chỗ bởi các chunk nói chung về Activation ở Trang 10 và 11.
   * *Tại sao B thành công:* Strategy B gom toàn bộ đoạn văn giải thích ReLU thành một khối thống nhất, giúp vector similarity đạt 0.78 và đứng Top-1.
2. **Câu `q08` trên Strategy B (Dev):**
   * *Câu hỏi:* "Điểm khác biệt cơ bản giữa lớp Pooling và lớp Convolution về mặt tham số là gì?" (Đáp án: Pooling không có tham số để học, Trang 21).
   * *Nguyên nhân thất bại của B:* Trang 21 chứa một bảng so sánh (Table). Khi trích xuất text dạng dòng, cấu trúc bảng bị phân rã thành các cụm từ rời rạc. Paragraph chunking không tái tạo được bảng, dẫn đến việc mô hình embedding không hiểu được ngữ cảnh so sánh giữa 2 hàng của bảng.
