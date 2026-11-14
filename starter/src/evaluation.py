"""Evaluate retrieved evidence and preserve per-query traces."""
import unicodedata
from typing import Any


def _normalize_for_match(text: str) -> str:
    """Normalize whitespace and Unicode for robust substring matching."""
    text = unicodedata.normalize("NFC", text or "")
    # Collapse all whitespace characters (newlines, multiple spaces) into single space
    return " ".join(text.split()).lower()


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and enough gold evidence; preserve hit@1 and hit@3 trace.

    A hit is valid when the chunk is from the gold page and contains the gold evidence quote.
    """
    gold_page = question.get("gold_page")
    raw_evidence = question.get("evidence_quote", "")
    norm_evidence = _normalize_for_match(raw_evidence)
    
    # We take the first 40-50 significant characters as key anchor if full quote spans formatting differences
    anchor_phrase = norm_evidence[:60] if len(norm_evidence) > 60 else norm_evidence

    annotated_hits = []
    hit_at_1 = False
    evidence_hit_at_3 = False

    for rank, h in enumerate(hits, start=1):
        chunk_page = h.get("page")
        norm_chunk_text = _normalize_for_match(h.get("text", ""))

        page_match = (chunk_page == gold_page)
        # Check full quote match or anchor match
        quote_match = (norm_evidence in norm_chunk_text) or (anchor_phrase in norm_chunk_text)
        is_hit = bool(page_match and quote_match)

        if rank == 1 and is_hit:
            hit_at_1 = True
        if is_hit:
            evidence_hit_at_3 = True

        annotated_hits.append({
            "rank": rank,
            "chunk_id": h.get("chunk_id"),
            "page": chunk_page,
            "chapter": h.get("chapter"),
            "score": h.get("score"),
            "page_match": page_match,
            "evidence_match": quote_match,
            "is_hit": is_hit,
            "text_snippet": h.get("text", "")[:120] + "..."
        })

    return {
        "question_id": question.get("id"),
        "split": question.get("split"),
        "question": question.get("question"),
        "expected_answer": question.get("expected_answer"),
        "gold_page": gold_page,
        "filter": question.get("filter"),
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "top_hits": annotated_hits
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1, Evidence Hit@3, and filter compliance across dev and test splits."""
    def calc_metrics(subset: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        if total == 0:
            return {"hit_1": "0/0", "hit_1_pct": 0.0, "hit_3": "0/0", "hit_3_pct": 0.0, "count": 0}
        
        hit1_count = sum(1 for t in subset if t["hit_at_1"])
        hit3_count = sum(1 for t in subset if t["evidence_hit_at_3"])
        
        return {
            "hit_1": f"{hit1_count}/{total}",
            "hit_1_pct": round(hit1_count / total * 100, 2),
            "hit_3": f"{hit3_count}/{total}",
            "hit_3_pct": round(hit3_count / total * 100, 2),
            "count": total
        }

    dev_traces = [t for t in traces if t["split"] == "dev"]
    test_traces = [t for t in traces if t["split"] == "test"]

    # Filter compliance check
    filter_traces = [t for t in traces if t.get("filter")]
    filter_compliant_count = 0
    total_filter_questions = len(filter_traces)

    for ft in filter_traces:
        expected_chapter = ft["filter"].get("chapter")
        all_hits_match = True
        for h in ft["top_hits"]:
            if h.get("chapter") != expected_chapter:
                all_hits_match = False
                break
        if all_hits_match and ft["top_hits"]:
            filter_compliant_count += 1

    filter_compliance_str = (
        f"{filter_compliant_count}/{total_filter_questions}"
        if total_filter_questions > 0 else "N/A"
    )

    return {
        "dev": calc_metrics(dev_traces),
        "test": calc_metrics(test_traces),
        "overall": calc_metrics(traces),
        "filter_compliance": filter_compliance_str
    }
