#This makes the api folder a Python package and can expose key components.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import health, query, upload
import uvicorn

app = FastAPI(
    title="AURA API",
    description="Multi-Agent AI System Backend",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allow Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(query.router, prefix="/api/query", tags=["Query"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload & Analyze"])

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
