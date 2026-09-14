# FORESIGHT Deployment Guide

## 1. GitHub

Create a public repository and push the complete `FORESIGHT` folder.

Do not commit secrets, tokens or credentials.

## 2. Streamlit Community Cloud

- Create a new app from the GitHub repository.
- Main file: `app/app.py`
- Python dependencies: `requirements.txt`
- Confirm the app opens in a private/incognito window.

## 3. Render — FastAPI scoring service

The included `render.yaml` defines the service.

If configuring manually:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn service.api:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

After deployment test:
- `/health`
- `/docs`
- `/forecast/SKU0001`

## 4. Submission

Use the public URLs for:
1. Source code repository
2. Live dashboard
3. Scoring API
4. Demo video
5. Feedback video
6. Project report

Open every public URL in an incognito window before submitting.
