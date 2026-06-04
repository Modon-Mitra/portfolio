from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, ClickEvent
from pydantic import BaseModel

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ClickIn(BaseModel):
    section: str
    dwell_seconds: float

@router.post("/log")
def log_click(click: ClickIn, request: Request, db: Session = Depends(get_db)):
    event = ClickEvent(
        visitor_ip=request.client.host,
        section=click.section,
        dwell_seconds=click.dwell_seconds
    )
    db.add(event)
    db.commit()
    return {"status": "click logged"}

@router.get("/stats")
def click_stats(db: Session = Depends(get_db)):
    from sqlalchemy import func
    stats = db.query(
        ClickEvent.section,
        func.count(ClickEvent.id).label("visits"),
        func.avg(ClickEvent.dwell_seconds).label("avg_dwell")
    ).group_by(ClickEvent.section).all()
    return [{"section": s, "visits": v, "avg_dwell": round(d, 2)} for s, v, d in stats]