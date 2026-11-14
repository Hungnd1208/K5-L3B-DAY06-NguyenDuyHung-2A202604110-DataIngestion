"""Generate data/manifest.json for the corpus."""
import json
from pathlib import Path

pages_file = Path("starter/data/extracted_pages.json")
with open(pages_file, "r", encoding="utf-8") as f:
    pages = json.load(f)

# Page calculation rule:
# Physical pages: 1 to 32.
# Useful content pages: Page 1 to Page 31 (31 pages >= 30 pages required).
# Page 32 is References.
page_details = []
total_content_chars = 0
for p_num in range(1, 33):
    text = pages.get(str(p_num), "")
    char_len = len(text)
    is_content = (1 <= p_num <= 31)
    if is_content:
        total_content_chars += char_len
    
    # Assign chapter metadata
    if p_num == 1:
        chapter = "loi_mo_dau"
    elif 2 <= p_num <= 4:
        chapter = "chuong_1"
    elif 5 <= p_num <= 6:
        chapter = "chuong_2"
    elif 7 <= p_num <= 25:
        chapter = "chuong_3"
    elif 26 <= p_num <= 28:
        chapter = "chuong_4"
    elif 29 <= p_num <= 30:
        chapter = "chuong_5"
    elif p_num == 31:
        chapter = "ket_luan"
    else:
        chapter = "tai_lieu_tham_khao"

    page_details.append({
        "page_number": p_num,
        "char_count": char_len,
        "is_content_page": is_content,
        "chapter": chapter
    })

manifest = {
    "source_id": "bao-cao-cnn",
    "title": "Báo cáo lý thuyết Convolutional Neural Network (CNN)",
    "source_path": "starter/data/BÁO CÁO LÝ THUYẾT CNN.pdf",
    "license_or_permission": "Academic Research & Educational Use (Tài liệu học tập nội bộ VinUni Lab)",
    "accessed_at": "2026-11-14",
    "total_physical_pages": 32,
    "total_content_pages": 31,
    "content_page_range": "1-31",
    "excluded_pages": [32],
    "exclusion_reason": "Trang 32 là danh mục tài liệu tham khảo (References)",
    "total_content_characters": total_content_chars,
    "page_counting_rule": "Chỉ tính các trang có nội dung kiến thức chuyên môn lý thuyết (trang 1-31). Loại bỏ trang 32 chứa danh mục tài liệu tham khảo.",
    "pages": page_details
}

manifest_dir = Path("data")
manifest_dir.mkdir(exist_ok=True)
with open(manifest_dir / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f"Manifest written to data/manifest.json. Total content pages: 31, chars: {total_content_chars}")
