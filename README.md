# Study Companion — Backend

FastAPI backend for the Study Companion app: authentication, file/PDF ingestion
into a Chroma vector store, an AI tutor chatbot (LangGraph + OpenAI, streamed
to the client over SSE), quiz generation/grading, flashcards (SM-2 spaced
repetition), and usage stats.

## Prerequisites

- Python 3.11+
- PostgreSQL (running locally, with a database created for this app)
- An OpenAI API key

## Setup

```bash
# From this directory (Study Companion/)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create your local config:

```bash
cp .env.example .env
```

Then fill in `.env`. At minimum you need:

| Variable | Required | Notes |
|---|---|---|
| `DATABASE_URL` | Yes | `postgresql://user:password@localhost:5432/study_companion` — the database must already exist; tables are created automatically on first run (no migrations to run) |
| `JWT_SECRET_KEY` | Yes | Generate with `openssl rand -hex 32` |
| `OPENAI_API_KEY` | Yes | Powers the chatbot, quiz generation, and flashcard generation |
| `ENVIRONMENT` | Recommended | `development` surfaces real error details in API responses; leave unset/`production` otherwise |
| `GOOGLE_CLIENT_ID` | Optional | Only needed for "Sign in with Google" |
| `SMTP_*` | Optional | Only needed for email verification / password reset emails |
| `HF_TOKEN`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `EMBEDDING_DEVICE` | Optional | Hugging Face model download/caching for the local cross-encoder and embedding models |
| `CONFIG_DIR`, `LOG_DIR`, `VDB_DIR`, `IMAGE_DIR` | Optional | Default to `configs/`, `logs/`, `vectorstore/`, `images/` if unset |

See `.env.example` for the full list with defaults.

## Running

```bash
uvicorn main:app --reload
```

(or `python main.py`, which does the same thing). The API serves on
`http://localhost:8000` by default — the frontend (`study-companion-ui`)
expects it there unless you change `VITE_BACKEND_URL` on that side too.

## Testing

```bash
pytest
```

Tests live in `tests/` and mock out the LangGraph agent and database, so
running them makes no real OpenAI calls and doesn't touch your Postgres data.

## Project layout

- `main.py` — FastAPI app and all HTTP routes
- `models/` — the LangGraph agents (`chatbot.py` for the tutor, `quiz_generator.py` for quizzes)
- `utils/` — auth, database, file/PDF handling, vector store, prompts, SRS scheduling, etc.
- `tests/` — pytest suite
