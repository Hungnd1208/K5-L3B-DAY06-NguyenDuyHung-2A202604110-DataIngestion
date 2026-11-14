# Benchmark Dataset — Day 06

Bộ benchmark gồm **20 câu hỏi** được xây dựng thủ công từ tài liệu `BÁO CÁO LÝ THUYẾT CNN.pdf` và đã được kiểm chứng 100% khớp với nội dung nguyên văn trong tài liệu.

## 1. Phân chia tập dữ liệu (Splits)
- **Tập Dev (15 câu: `q01` – `q15`):** Dùng để thử nghiệm, quan sát và so sánh các chiến lược.
- **Tập Test (5 câu: `q16` – `q20`):** Tập holdout độc lập, giữ cố định để đánh giá cuối cùng.

## 2. Phân loại câu hỏi
- **Trực tiếp (Direct factual):** `q01`, `q02`, `q03`, `q04`, `q05`, `q08`, `q09`, `q11`, `q15`, `q16`.
- **Diễn đạt lại (Paraphrased):** `q07`, `q10`, `q19`.
- **Điều kiện / Ngoại lệ (Conditional / Edge case):** `q06`, `q12`, `q18`, `q20`.
- **Kiểm tra Metadata Filter (Filter-augmented):**
  - Trong Dev (2 câu): `q13` (lọc theo `{"chapter": "chuong_3"}`), `q14` (lọc theo `{"chapter": "chuong_4"}`).
  - Trong Test (1 câu): `q17` (lọc theo `{"chapter": "chuong_4"}`).
  - *Tổng cộng: 3 câu có metadata filter (đáp ứng đúng tiêu chí $\ge 3$ câu có filter, $\ge 1$ câu trong test).*

## 3. Cấu trúc mỗi bản ghi trong `benchmark.json`
- `id`: Mã câu hỏi định danh duy nhất.
- `split`: `"dev"` hoặc `"test"`.
- `type`: Dạng câu hỏi.
- `question`: Câu hỏi tự nhiên.
- `expected_answer`: Câu trả lời tóm tắt.
- `gold_source_id`: `"bao-cao-cnn"`.
- `gold_page`: Trang vật lý trong tài liệu PDF (1-31).
- `evidence_quote`: Đoạn trích nguyên văn chứa bằng chứng trả lời.
- `filter`: Điều kiện lọc metadata (hoặc `null`).
