"""API routes for document management."""
from fastapi import APIRouter, HTTPException, Header, Query
from typing import Optional
from uuid import UUID
from app.schemas.documents import DocumentResponse, DocumentListResponse
from app.clients.supabase import SupabaseClient

router = APIRouter(prefix="/documents", tags=["documents"])

_supabase_client = SupabaseClient()


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    List documents with optional filters.

    Args:
        status: Filter by status (pending, processing, ready, error)
        limit: Maximum number of results
        offset: Offset for pagination
        x_user_id: User ID from header

    Returns:
        List of documents
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required")

    try:
        documents = await _supabase_client.list_documents(
            user_id=x_user_id,
            status=status,
            limit=limit,
            offset=offset,
        )

        document_responses = [
            DocumentResponse(
                id=doc["id"],
                name=doc["name"],
                status=doc["status"],
                chunk_count=doc.get("chunk_count", 0),
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
                metadata=doc.get("metadata", {}),
            )
            for doc in documents
        ]

        return DocumentListResponse(documents=document_responses, count=len(document_responses))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}") from e


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Get document details.

    Args:
        document_id: Document ID
        x_user_id: User ID from header

    Returns:
        Document information
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required")

    try:
        document = await _supabase_client.get_document(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check user ownership (RLS should handle this, but verify)
        if document.get("user_id") != x_user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return DocumentResponse(
            id=document["id"],
            name=document["name"],
            status=document["status"],
            chunk_count=document.get("chunk_count", 0),
            created_at=document["created_at"],
            updated_at=document["updated_at"],
            metadata=document.get("metadata", {}),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}") from e

