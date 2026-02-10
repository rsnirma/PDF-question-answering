import os
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_pdf_pages(pdf_path: str) -> list[str]:
    reader = PyPDF2.PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return pages

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def answer_with_context(context: str, question: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"{context}\n\nQuestion: {question}\nAnswer:"
        }],
        temperature=0,
    )
    return resp.choices[0].message.content
