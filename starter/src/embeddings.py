"""Embedding provider implementation supporting Gemini API (gemini-embedding-001) and local fallback."""
import logging
import os
import time
from typing import Protocol
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Production embedder using Google Gemini Embedding API with high-quality multilingual support.

    Model: gemini-embedding-001 with output_dimensionality=768 (Matryoshka representation).
    Falls back to local SentenceTransformer if no API key is provided.
    """

    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv("starter/.env")
        load_dotenv(".env")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

        if self.gemini_key and len(self.gemini_key) > 10:
            from google import genai
            from google.genai import types
            self.model_name = model_name or "models/gemini-embedding-001"
            self.dimensions = 768
            self.client = genai.Client(api_key=self.gemini_key)
            self._types = types
            self.is_gemini = True
            logger.info("Initialized Gemini Embedder with model: %s (dim=%d)", self.model_name, self.dimensions)
        else:
            from sentence_transformers import SentenceTransformer
            self.model_name = model_name or "all-MiniLM-L6-v2"
            self.local_model = SentenceTransformer(self.model_name)
            self.dimensions = self.local_model.get_embedding_dimension()
            self.is_gemini = False
            logger.info("Initialized local SentenceTransformer with model: %s (dim=%d)", self.model_name, self.dimensions)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch encode document texts into normalized dense vectors."""
        if not texts:
            return []

        if not self.is_gemini:
            embeddings = self.local_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings.tolist()

        # Batch encoding via Gemini API
        all_embeddings: list[list[float]] = []
        batch_size = 16  # Safe batch size for Gemini API rate limits

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                        config=self._types.EmbedContentConfig(output_dimensionality=self.dimensions)
                    )
                    batch_vectors = [emb.values for emb in response.embeddings]
                    all_embeddings.extend(batch_vectors)
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait_sec = 35
                        logger.warning("Gemini 429 Rate Limit encountered. Sleeping %ds to reset quota (attempt %d/%d)...", wait_sec, attempt + 1, max_retries)
                        time.sleep(wait_sec)
                    elif attempt < max_retries - 1:
                        logger.warning("Gemini API error (attempt %d): %s. Retrying in 5s...", attempt + 1, e)
                        time.sleep(5)
                    else:
                        raise e

            time.sleep(1.0)  # Safe delay between batches

        return all_embeddings

    def embed_query(self, question: str) -> list[float]:
        """Encode a single question string into normalized dense vector."""
        if not question or not question.strip():
            raise ValueError("Query string cannot be empty")

        if not self.is_gemini:
            embedding = self.local_model.encode(
                question,
                normalize_embeddings=True
            )
            return embedding.tolist()

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=question,
                    config=self._types.EmbedContentConfig(output_dimensionality=self.dimensions)
                )
                return response.embeddings[0].values
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_sec = 30
                    logger.warning("Gemini 429 on query. Sleeping %ds...", wait_sec)
                    time.sleep(wait_sec)
                elif attempt < max_retries - 1:
                    logger.warning("Gemini API query error (attempt %d): %s. Retrying in 3s...", attempt + 1, e)
                    time.sleep(3)
                else:
                    raise e
