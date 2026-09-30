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
The production API is deployed as a Cloudflare Python Worker from `cloudflare-api/`; local development continues to use the FastAPI process and Vite proxy.
