import uuid
from pathlib import Path
import gradio as gr
import inngest

from rag import get_embedding, answer_with_context
from inngest_functions import inngest_client, doc_status, doc_error, qdrant_store

DOC_DIR = Path("./docs")
DOC_DIR.mkdir(parents=True, exist_ok=True)

def queue_pdf_processing(pdf_file):
    if pdf_file is None:
        return None, "Please upload a PDF first."

    doc_id = str(uuid.uuid4())
    out_path = str(DOC_DIR / f"{doc_id}.pdf")

    src_path = getattr(pdf_file, "name", None) or str(pdf_file)
    Path(out_path).write_bytes(Path(src_path).read_bytes())

    doc_status[doc_id] = "queued"

    inngest_client.send_sync(
        inngest.Event(
            name="app/pdf.process",
            data={"doc_id": doc_id, "pdf_path": out_path},
        )
    )

    return doc_id, f"✅ Queued. doc_id={doc_id} (status: queued)"

def get_status(doc_id: str):
    if not doc_id:
        return "No doc yet."
    s = doc_status.get(doc_id, "unknown")
    if s == "failed":
        return f"❌ failed: {doc_error.get(doc_id, '(no details)')}"
    return f"Status: {s}"

def answer_question(doc_id, question):
    if not doc_id:
        return "Upload and process a PDF first."

    s = doc_status.get(doc_id, "unknown")
    if s != "ready":
        return f"PDF not ready yet (status: {s})."

    q_emb = get_embedding(question)
    context = "\n\n".join(
        qdrant_store.search(doc_id=doc_id, query_embedding=q_emb, top_k=3)
    )

    return answer_with_context(context, question)


def build_ui():
    with gr.Blocks() as demo:
        doc_state = gr.State(value="")

        with gr.Row():
            pdf_input = gr.File(label="Upload PDF")
            upload_btn = gr.Button("Process PDF (Background)")
            status = gr.Textbox(label="Status")

        upload_btn.click(queue_pdf_processing, inputs=[pdf_input], outputs=[doc_state, status])

        with gr.Row():
            status_btn = gr.Button("Refresh Status")
            status_btn.click(get_status, inputs=[doc_state], outputs=[status])

        question = gr.Textbox(label="Ask a question about the PDF")
        answer = gr.Textbox(label="Answer")

        question.submit(answer_question, inputs=[doc_state, question], outputs=[answer])

    return demo
