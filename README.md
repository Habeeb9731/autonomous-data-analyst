# Autonomous Data Analyst

A Vite + React analytics workspace with a deterministic FastAPI profiling engine.

## Local development

```bash
npm install
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/uvicorn backend.main:app --reload --port 8000
npm run dev
```

The Vite dev server proxies `/api` to the local FastAPI service.

## Production deployment

The frontend builds with `npm run build` and deploys to Cloudflare Pages from `dist/`.
The Python API is a separate service and must be hosted separately for uploaded-file profiling to work in production; the Pages static deployment does not execute the local FastAPI process.
