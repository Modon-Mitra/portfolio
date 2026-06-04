from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from database import init_db
from routes import visitors, comments, clicks, ml_routes

app = FastAPI(title="Abhijeet Portfolio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()

app.include_router(visitors.router, prefix="/api/visitors", tags=["Visitors"])
app.include_router(comments.router, prefix="/api/comments", tags=["Comments"])
app.include_router(clicks.router, prefix="/api/clicks", tags=["Clicks"])
app.include_router(ml_routes.router, prefix="/api/ml", tags=["ML"])

@app.get("/")
def root():
    return {"status": "Abhijeet Portfolio API running ✅"}