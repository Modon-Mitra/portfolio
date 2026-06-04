from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from database import init_db
from routes import visitors, comments, clicks, ml_routes

app = FastAPI(title="Abhijeet Portfolio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aboutmyselfinapage.netlify.app",
        "http://localhost:5500",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
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