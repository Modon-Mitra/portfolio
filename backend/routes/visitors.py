from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, Visitor

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/log")
def log_visitor(request: Request, db: Session = Depends(get_db)):
    visitor = Visitor(
        ip_address=request.client.host,
        device=request.headers.get("user-agent", "unknown")[:200],
    )
    db.add(visitor)
    db.commit()
    return {"status": "logged"}

@router.get("/count")
def get_visitor_count(db: Session = Depends(get_db)):
    count = db.query(Visitor).count()
    return {"total_visitors": count}

@router.get("/recent")
def get_recent_visitors(db: Session = Depends(get_db)):
    visitors = db.query(Visitor).order_by(
        Visitor.visited_at.desc()
    ).limit(20).all()
    return visitors