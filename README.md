PDF RAG Service

Event-driven RAG service for PDF document question answering.

This project implements a backend service that allows users to upload PDF documents, process them asynchronously, and ask natural language questions over the document content using embeddings and vector search.
The system follows a retrieval-augmented generation (RAG) pattern with background processing and a simple web interface.

Features

  Upload and process PDF documents
  Asynchronous background processing using event-driven workflows
  Text chunking and embedding generation
  Vector-based similarity search with Qdrant
  Question answering using retrieved document context
  Simple web UI for interaction

Architecture Overview
  PDF is uploaded via the web UI
  A background event is triggered to process the document
  PDF pages are embedded and stored in a vector database
  User questions are embedded and matched against stored vectors
  Relevant content is retrieved and used to generate an answer

Tech Stack

  Python
  FastAPI – backend API
  Gradio – web interface
  Inngest – event-driven background processing

Qdrant – vector database

OpenAI Embeddings – text embeddings
