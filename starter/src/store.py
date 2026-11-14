"""Store and search vectors using PostgreSQL + pgvector."""
import json
import logging
from typing import Any
import psycopg
from psycopg.rows import dict_row
from .models import Chunk

logger = logging.getLogger(__name__)


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        self.database_url = database_url
        self.dimensions = dimensions
        logger.info("Initializing PgVectorStore with dimensions=%d", dimensions)

    def _get_connection(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def create_schema(self) -> None:
        """Create extension and schema table with vector column matching dimensions."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Ensure pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Check existing table dimensions if it exists
                cur.execute("""
                SELECT atttypmod 
                FROM pg_attribute 
                WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding';
                """)
                row = cur.fetchone()
                if row and row["atttypmod"] != self.dimensions:
                    logger.warning(
                        "Schema dimension mismatch (existing=%s, required=%d). Recreating table...",
                        row["atttypmod"], self.dimensions
                    )
                    cur.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")

                # Table schema for chunks across strategies
                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    strategy VARCHAR(16) NOT NULL,
                    chunk_id VARCHAR(128) NOT NULL,
                    document_id VARCHAR(128) NOT NULL,
                    source VARCHAR(256) NOT NULL,
                    page INT NOT NULL,
                    chapter VARCHAR(64),
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                );
                """)
                
                # Index for metadata filtering
                cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_strategy_chapter 
                ON document_chunks(strategy, chapter);
                """)
                
                conn.commit()
        logger.info("Database schema and indexes initialized successfully")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Idempotently insert or update chunks with vectors.

        Ensures old chunks for this strategy & document are cleaned up or updated without duplicates.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunks count ({len(chunks)}) does not match vectors count ({len(vectors)})")

        if not chunks:
            return

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Clear existing chunks for this strategy to ensure clean idempotent ingestion
                cur.execute(
                    "DELETE FROM document_chunks WHERE strategy = %s AND document_id = %s;",
                    (strategy, chunks[0].document_id)
                )

                insert_sql = """
                INSERT INTO document_chunks (
                    strategy, chunk_id, document_id, source, page, chapter, text, metadata, embedding
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s::vector
                )
                ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    chapter = EXCLUDED.chapter,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
                """
                
                batch_data = []
                for chunk, vector in zip(chunks, vectors):
                    if len(vector) != self.dimensions:
                        raise ValueError(f"Vector dimension {len(vector)} != expected {self.dimensions}")
                    
                    vector_str = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"
                    chapter = chunk.metadata.get("chapter", "unknown")
                    batch_data.append((
                        strategy,
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.source,
                        chunk.page,
                        chapter,
                        chunk.text,
                        json.dumps(chunk.metadata),
                        vector_str
                    ))

                cur.executemany(insert_sql, batch_data)
                conn.commit()

        logger.info("Upserted %d chunks for strategy '%s'", len(chunks), strategy)

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Filter metadata in SQL, rank by cosine distance, and return top-k."""
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Query vector dimension {len(query_vector)} != expected {self.dimensions}")

        vector_str = "[" + ",".join(f"{v:.6f}" for v in query_vector) + "]"
        
        where_clauses = ["strategy = %s"]
        where_params: list[Any] = [strategy]

        if filters:
            for k, v in filters.items():
                if k == "chapter":
                    where_clauses.append("chapter = %s")
                    where_params.append(v)
                else:
                    where_clauses.append("metadata->>%s = %s")
                    where_params.extend([k, str(v)])

        where_sql = " AND ".join(where_clauses)

        sql = f"""
        SELECT 
            chunk_id,
            document_id,
            source,
            page,
            chapter,
            text,
            metadata,
            1.0 - (embedding <=> %s::vector) AS similarity_score,
            (embedding <=> %s::vector) AS cosine_distance
        FROM document_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
        """

        query_params = [vector_str, vector_str] + where_params + [vector_str, top_k]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, query_params)
                rows = cur.fetchall()

        results = []
        for r in rows:
            results.append({
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "source": r["source"],
                "page": r["page"],
                "chapter": r["chapter"],
                "text": r["text"],
                "metadata": r["metadata"],
                "score": float(r["similarity_score"]),
                "distance": float(r["cosine_distance"])
            })

        return results
