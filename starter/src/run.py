"""Day 06 Orchestrator: End-to-end Data Ingestion & Benchmark Pipeline."""
import json
import logging
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from .ingestion import load_pdf
from .chunking import fixed_size, by_paragraph
from .embeddings import MyEmbedder
from .store import PgVectorStore
from .evaluation import evaluate_query, summarize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_pipeline")


def main() -> None:
    # 0. Load environment
    env_path = Path("starter/.env")
    if not env_path.exists():
        env_path = Path(".env")
    load_dotenv(dotenv_path=env_path)

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not found in .env. Exiting.")
        sys.exit(1)

    logger.info("=== STEP 1: LOAD CORPUS & BENCHMARK ===")
    pdf_path = Path("starter/data/BÁO CÁO LÝ THUYẾT CNN.pdf")
    if not pdf_path.exists():
        pdf_path = Path("data/BÁO CÁO LÝ THUYẾT CNN.pdf")

    pages = load_pdf(pdf_path, document_id="bao-cao-cnn", source="starter/data/BÁO CÁO LÝ THUYẾT CNN.pdf")
    logger.info("Loaded %d content pages (Validation: >= 30 content pages requirement MET)", len(pages))
    if len(pages) < 30:
        raise ValueError(f"Corpus must have at least 30 content pages, got {len(pages)}")

    bench_path = Path("benchmark/benchmark.json")
    if not bench_path.exists():
        bench_path = Path("../benchmark/benchmark.json")
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    logger.info("Loaded %d benchmark questions (Split: %d dev / %d test)",
                len(benchmark),
                sum(1 for q in benchmark if q["split"] == "dev"),
                sum(1 for q in benchmark if q["split"] == "test"))

    logger.info("=== STEP 2: BUILD CHUNKING STRATEGIES ===")
    # Strategy A: Fixed-size (size=500, overlap=50)
    chunks_a = []
    for p in pages:
        chunks_a.extend(fixed_size(p, size=500, overlap=50))
    logger.info("Strategy A (Fixed-size): created %d chunks", len(chunks_a))

    # Strategy B & C: Structure-aware / by paragraph (max_size=700)
    chunks_b = []
    for p in pages:
        chunks_b.extend(by_paragraph(p, max_size=700))
    logger.info("Strategy B & C (By paragraph): created %d chunks", len(chunks_b))

    logger.info("=== STEP 3: EMBEDDING & PERSISTENCE IN PGVECTOR ===")
    embedder = MyEmbedder()
    store = PgVectorStore(database_url=database_url, dimensions=embedder.dimensions)
    store.create_schema()

    # Ingest Strategy A
    t0 = time.time()
    texts_a = [c.text for c in chunks_a]
    vectors_a = embedder.embed_documents(texts_a)
    store.upsert("A", chunks_a, vectors_a)
    time_a = time.time() - t0
    logger.info("Strategy A ingestion finished in %.2fs (Cost: $0.00 local model)", time_a)

    # Ingest Strategy B & C
    t0 = time.time()
    texts_b = [c.text for c in chunks_b]
    vectors_b = embedder.embed_documents(texts_b)
    store.upsert("B", chunks_b, vectors_b)
    # Strategy C uses identical chunk structure and vectors as B, stored under 'C' index
    store.upsert("C", chunks_b, vectors_b)
    time_bc = time.time() - t0
    logger.info("Strategy B & C ingestion finished in %.2fs (Cost: $0.00 local model)", time_bc)

    logger.info("=== STEP 4: RUN BENCHMARK EVALUATION (DEV + TEST) ===")
    results_dirs = [Path("results")]
    if Path("../results").exists():
        results_dirs.append(Path("../results"))
    if Path("starter/results").exists():
        results_dirs.append(Path("starter/results"))
    # Deduplicate paths
    unique_dirs = []
    for d in results_dirs:
        d.mkdir(parents=True, exist_ok=True)
        resolved = d.resolve()
        if resolved not in unique_dirs:
            unique_dirs.append(resolved)

    strategy_stats = {}
    for strat in ["A", "B", "C"]:
        logger.info("Evaluating Strategy %s ...", strat)
        traces = []
        for q in benchmark:
            q_vec = embedder.embed_query(q["question"])
            # Strategy C activates metadata filter when query has filter
            filters = q.get("filter") if strat == "C" else None
            hits = store.search(strategy=strat, query_vector=q_vec, top_k=3, filters=filters)
            trace = evaluate_query(q, hits)
            traces.append(trace)

        # Save strategy traces to all target results directories
        for u_dir in unique_dirs:
            trace_file = u_dir / f"traces_strategy_{strat}.json"
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(traces, f, ensure_ascii=False, indent=2)

        summary = summarize(traces)
        strategy_stats[strat] = {
            "summary": summary,
            "chunk_count": len(chunks_a) if strat == "A" else len(chunks_b),
            "ingest_time_seconds": round(time_a if strat == "A" else (time_bc / 2), 2),
            "cost_usd": 0.0,
            "traces_file": f"results/traces_strategy_{strat}.json"
        }

    # Save overall summary metrics to all target results directories
    for u_dir in unique_dirs:
        summary_file = u_dir / "metrics_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(strategy_stats, f, ensure_ascii=False, indent=2)

    logger.info("=== STEP 5: FINAL BENCHMARK RESULTS SUMMARY ===")
    print("\n" + "=" * 80)
    print("BENCHMARK RETRIEVAL ACCURACY REPORT (Day 06)")
    print("=" * 80)
    print(f"{'Strategy':<10} | {'Split':<6} | {'Hit@1 (n/N)':<12} | {'Hit@3 (n/N)':<12} | {'Chunks':<8} | {'Time (s)':<10} | {'Filter Compliance'}")
    print("-" * 80)
    for strat in ["A", "B", "C"]:
        s_data = strategy_stats[strat]
        summ = s_data["summary"]
        chunks = s_data["chunk_count"]
        itime = s_data["ingest_time_seconds"]
        comp = summ.get("filter_compliance", "N/A")

        # Dev row
        dev_h1 = f"{summ['dev']['hit_1']} ({summ['dev']['hit_1_pct']}%)"
        dev_h3 = f"{summ['dev']['hit_3']} ({summ['dev']['hit_3_pct']}%)"
        print(f"{strat:<10} | {'dev':<6} | {dev_h1:<12} | {dev_h3:<12} | {chunks:<8} | {itime:<10} | {comp if strat == 'C' else 'N/A'}")

        # Test row
        test_h1 = f"{summ['test']['hit_1']} ({summ['test']['hit_1_pct']}%)"
        test_h3 = f"{summ['test']['hit_3']} ({summ['test']['hit_3_pct']}%)"
        print(f"{'':<10} | {'test':<6} | {test_h1:<12} | {test_h3:<12} | {'':<8} | {'':<10} | {comp if strat == 'C' else 'N/A'}")
        print("-" * 80)

    print("\nAll traces and metrics successfully exported to results/")


if __name__ == "__main__":
    main()
