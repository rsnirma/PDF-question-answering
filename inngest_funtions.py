# inngest_functions.py
import logging
import inngest

from rag import load_pdf_pages, get_embedding
from storage import QdrantStore

# Used by UI to show progress/errors
doc_status: dict[str, str] = {}
doc_error: dict[str, str] = {}
qdrant_store = QdrantStore(dim=1536)

# Inngest client
inngest_client = inngest.Inngest(
    app_id="pdf_rag_app",
    logger=logging.getLogger("uvicorn"),
)

@inngest_client.create_function(
    fn_id="process_pdf_to_qdrant",
    trigger=inngest.TriggerEvent(event="app/pdf.process"),
)
async def process_pdf_to_qdrant(ctx: inngest.Context) -> dict:
    """
    Takes:
      ctx.event.data["doc_id"]
      ctx.event.data["pdf_path"]

    Loads PDF pages -> embeds each page -> upserts into Qdrant
    Payload includes doc_id so we can filter per PDF during search.
    """
    doc_id = ctx.event.data["doc_id"]
    pdf_path = ctx.event.data["pdf_path"]

    try:
        doc_status[doc_id] = "processing"

        pages = load_pdf_pages(pdf_path)

        # Store each page as a chunk
        for i, text in enumerate(pages):
            if not text or not text.strip():
                continue
            emb = get_embedding(text)
            qdrant_store.add(doc_id=doc_id, text=text, embedding=emb, page=i)

        doc_status[doc_id] = "ready"
        return {"doc_id": doc_id, "pages": len(pages), "status": "ready"}

    except Exception as e:
        doc_status[doc_id] = "failed"
        doc_error[doc_id] = str(e)
        raise
