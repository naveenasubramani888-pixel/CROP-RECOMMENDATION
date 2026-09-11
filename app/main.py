"""
AgriSense AI FastAPI Application Entry Point.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path

from app.config import settings
from app.database import engine, Base
from app.routes import prediction, dashboard, crops

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_TAGLINE,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Frontend Static Directory if exists
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Include API Routers
app.include_router(prediction.router, prefix=settings.API_PREFIX)
app.include_router(dashboard.router, prefix=settings.API_PREFIX)
app.include_router(crops.router, prefix=settings.API_PREFIX)

@app.get("/health", tags=["Health Check"])
def health_check():
    """Returns application health status."""
    return {
        "status": "healthy",
        "app_name": settings.PROJECT_NAME,
        "version": settings.VERSION
    }

# Frontend Page Routes
@app.get("/", response_class=HTMLResponse, tags=["Web Dashboard Pages"])
def serve_home_page():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/recommendation.html", response_class=HTMLResponse, tags=["Web Dashboard Pages"])
def serve_recommendation_page():
    return FileResponse(str(FRONTEND_DIR / "recommendation.html"))

@app.get("/dashboard.html", response_class=HTMLResponse, tags=["Web Dashboard Pages"])
def serve_dashboard_page():
    return FileResponse(str(FRONTEND_DIR / "dashboard.html"))

@app.get("/history.html", response_class=HTMLResponse, tags=["Web Dashboard Pages"])
def serve_history_page():
    return FileResponse(str(FRONTEND_DIR / "history.html"))

@app.get("/crops.html", response_class=HTMLResponse, tags=["Web Dashboard Pages"])
def serve_crops_page():
    return FileResponse(str(FRONTEND_DIR / "crops.html"))

@app.get("/insights.html", response_class=HTMLResponse, tags=["Web Dashboard Pages"])
def serve_insights_page():
    return FileResponse(str(FRONTEND_DIR / "insights.html"))

@app.get("/about.html", response_class=HTMLResponse, tags=["Web Dashboard Pages"])
def serve_about_page():
    return FileResponse(str(FRONTEND_DIR / "about.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
