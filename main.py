from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
import inngest.fast_api
import gradio as gr

from ui import build_ui
from inngest_functions import inngest_client, process_pdf_to_qdrant

demo = build_ui()

app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [process_pdf_to_qdrant])
app = gr.mount_gradio_app(app, demo, path="/")
