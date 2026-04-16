# 📄 PDF QA Assistant

> An AI-powered Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF or text document and have an intelligent conversation with it — powered by Google Gemini and ChromaDB.

---

## 🚀 Live Demo

> Upload a PDF → Ask questions → Get context-grounded answers with source citations.

Built with a Chainlit chat interface, deployable via Docker or run locally.

---

## 🧠 What It Does

PDF QA Assistant is a full-stack RAG (Retrieval-Augmented Generation) pipeline that:

1. **Accepts** a PDF or `.txt` file upload through a conversational UI
2. **Parses** document content page-by-page using PyMuPDF
3. **Chunks** the content intelligently using a custom recursive character text splitter
4. **Embeds** the chunks using Google Gemini's `text-embedding-004` model
5. **Stores** embeddings in a local ChromaDB vector store
6. **Retrieves** the most semantically relevant chunks for each user query (top-k=4)
7. **Generates** a streamed, grounded answer using `gemini-1.5-flash`
8. **Cites** the source chunks inline so users can validate every response

---

## 🏗️ Architecture & Workflow

```
User Uploads PDF/TXT
        │
        ▼
 ┌─────────────────┐
 │  PyMuPDF Parser │  ← Extracts text per page with metadata (source, page number)
 └────────┬────────┘
          │
          ▼
 ┌──────────────────────────┐
 │ RecursiveCharacterText   │  ← Custom splitter: chunk_size=1000, overlap=100
 │       Splitter           │     Tries \n\n → \n → space → char fallback
 └────────┬─────────────────┘
          │
          ▼
 ┌──────────────────────────┐
 │  GeminiEmbeddingFunction │  ← google/text-embedding-004 via google-generativeai
 │  + ChromaDB (Persistent) │     Stored in ./chroma_db for session reuse
 └────────┬─────────────────┘
          │
     User Asks Question
          │
          ▼
 ┌─────────────────────────────┐
 │  VectorDatabase.search_by   │  ← Top-4 cosine-similar chunks retrieved
 │       text(query, k=4)      │
 └────────┬────────────────────┘
          │
          ▼
 ┌──────────────────────────────────────┐
 │  RetrievalAugmentedQAPipeline        │
 │  ┌───────────────────────────────┐   │
 │  │  System Prompt + Context      │   │  ← Context injected into user prompt
 │  │  → ChatGemini.astream()       │   │  ← Async streaming via Gemini 1.5 Flash
 │  └───────────────────────────────┘   │
 └────────┬─────────────────────────────┘
          │
          ▼
   Streamed Answer + Source Citations
        (Chainlit UI)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI / Chat Framework** | [Chainlit](https://chainlit.io) | Conversational web UI with file upload |
| **LLM** | Google Gemini 1.5 Flash | Streaming answer generation |
| **Embeddings** | Google `text-embedding-004` | Semantic vector representation |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Persistent local embedding storage |
| **PDF Parsing** | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) | Fast, accurate PDF text extraction |
| **Text Splitting** | Custom `RecursiveCharacterTextSplitter` | Intelligent chunk creation with overlap |
| **Runtime** | Python 3.13 + [uv](https://github.com/astral-sh/uv) | Fast dependency management |
| **Containerization** | Docker | Easy deployment to any cloud platform |

---

## 📁 Project Structure

```
PDFqaAssistant-main/
├── app.py                          # Main Chainlit app — pipeline orchestration & UI logic
├── aimakerspace/
│   ├── text_utils.py               # PDFLoader, TextFileLoader, RecursiveCharacterTextSplitter
│   ├── vectordatabase.py           # ChromaDB integration + GeminiEmbeddingFunction
│   ├── google_utils/
│   │   └── chatmodel.py            # ChatGemini — LLM wrapper with async streaming
│   └── openai_utils/
│       ├── chatmodel.py            # ChatOpenAI (fallback)
│       └── prompts.py              # SystemRolePrompt, UserRolePrompt, AssistantRolePrompt
├── Dockerfile                      # Production-ready Docker config (port 7860)
├── pyproject.toml                  # Dependencies via uv
├── chroma_db/                      # Auto-generated persistent vector store
└── .env                            # API key configuration (not committed)
```

---

## ⚙️ Key Implementation Details

### Custom Text Splitter
Rather than using LangChain's splitter out of the box, this project implements `RecursiveCharacterTextSplitter` from scratch. It tries a hierarchy of separators (`\n\n` → `\n` → `space` → `char`) and includes overlap-aware chunking for cross-boundary context preservation.

### Dual LLM / Embedding Support
The app detects your available API key at runtime:
- **Google API Key** → Uses `ChatGemini` + `GeminiEmbeddingFunction` (primary)
- **OpenAI API Key** → Falls back to `ChatOpenAI` + OpenAI embeddings
- **Neither** → Uses ChromaDB's default sentence-transformer embeddings

### Async Streaming
The entire answer pipeline (`arun_pipeline`) is async. Gemini responses are streamed token-by-token directly into the Chainlit UI using `generate_content_async(..., stream=True)`.

### Source Citations
After every response, the top-4 retrieved document chunks are surfaced as inline `cl.Text` elements, allowing users to see exactly which part of the document grounded the answer.

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv) package manager
- A Google AI API key (get one at [aistudio.google.com](https://aistudio.google.com))

### 1. Clone & Install

```bash
git clone https://github.com/your-username/PDFqaAssistant.git
cd PDFqaAssistant
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
# Add your key:
echo "GOOGLE_API_KEY=your_key_here" >> .env
```

### 3. Run Locally

```bash
uv run chainlit run app.py
```

Open `http://localhost:8000` in your browser.

### 4. Run with Docker

```bash
docker build -t pdf-qa-assistant .
docker run -p 7860:7860 -e GOOGLE_API_KEY=your_key_here pdf-qa-assistant
```

Open `http://localhost:7860`.

---

## 🌐 Deployment

The Dockerfile exposes port `7860` and is compatible with:
- **Hugging Face Spaces** (native Chainlit support)
- **Render / Railway** (Docker deploy)
- **Google Cloud Run**

---

## 📦 Dependencies

```toml
chainlit >= 2.0.4         # Chat UI framework
google-generativeai >= 0.3.0  # Gemini LLM + Embeddings
chromadb >= 0.6.3         # Vector store
pymupdf >= 1.25.1         # PDF parsing
numpy >= 2.2.2            # Numerical ops
tiktoken >= 0.8.0         # Token counting
pydantic == 2.10.1        # Data validation
python-dotenv >= 1.0.0    # Env management
websockets >= 14.2        # Chainlit async support
```

---

## 🔭 Future Improvements

- [ ] Multi-document session support
- [ ] Support for DOCX, HTML, and Markdown files
- [ ] Re-ranking retrieved chunks before generation  
- [ ] Persistent user sessions with conversation memory
- [ ] Hybrid search (BM25 + vector similarity)

---

## 👤 Author

**Bhuvanesh**  
[Portfolio](#) · [LinkedIn](#) · [GitHub](#)

> *Built as part of an AI Engineering exploration into production-grade RAG systems.*
