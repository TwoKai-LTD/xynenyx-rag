"""Content chunking using LlamaIndex."""
from typing import List, Dict, Any
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from app.config import settings


class Chunker:
    """Chunker for splitting documents into chunks."""

    def __init__(self):
        """Initialize chunker with LlamaIndex SentenceSplitter."""
        self.splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            paragraph_separator="\n\n",
        )

    def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Split a document into chunks.

        Args:
            text: Document text to chunk
            metadata: Metadata to preserve per chunk

        Returns:
            List of chunk dictionaries with:
                - content: Chunk text
                - metadata: Preserved metadata
                - token_count: Approximate token count
        """
        # Create LlamaIndex Document
        doc = Document(text=text, metadata=metadata)

        # Split into nodes (chunks)
        nodes = self.splitter.get_nodes_from_documents([doc])

        # Convert to chunk dictionaries
        chunks = []
        for idx, node in enumerate(nodes):
            # Approximate token count (1 token ≈ 4 characters)
            token_count = len(node.text) // 4

            chunks.append(
                {
                    "content": node.text,
                    "metadata": {**metadata, **node.metadata},
                    "token_count": token_count,
                    "chunk_index": idx,
                }
            )

        return chunks

