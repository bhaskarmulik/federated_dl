from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get('/')
def index():
    return HTMLResponse('<h3>FL Dashboard (MVP)</h3><p>Wire up WebSocket metrics here.</p>')
