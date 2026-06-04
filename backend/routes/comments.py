from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, Comment
from pydantic import BaseModel
from typing import Optional
import pickle, os

router = APIRouter()

BASE = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

def load_sentiment():
    path = os.path.join(BASE, 'sentiment_model.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

sentiment_model = load_sentiment()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class CommentIn(BaseModel):
    name: str
    email: Optional[str] = ""
    message: str

@router.post("/add")
def add_comment(comment: CommentIn, db: Session = Depends(get_db)):
    sentiment = "Neutral"
    score = 0.5
    if sentiment_model:
        sentiment = sentiment_model.predict([comment.message])[0]
        score = float(max(sentiment_model.predict_proba([comment.message])[0]))

    new_comment = Comment(
        name=comment.name,
        email=comment.email,
        message=comment.message,
        sentiment=sentiment,
        sentiment_score=round(score, 3)
    )
    db.add(new_comment)
    db.commit()
    db.refresh(new_comment)
    return {"status": "Comment saved", "id": new_comment.id, "sentiment": sentiment}

@router.get("/all")
def get_comments(db: Session = Depends(get_db)):
    comments = db.query(Comment).order_by(Comment.created_at.desc()).all()
    return comments

@router.delete("/delete/{comment_id}")
def delete_comment(comment_id: int, db: Session = Depends(get_db)):
    comment = db.query(Comment).filter(Comment.id == comment_id).first()
    if comment:
        db.delete(comment)
        db.commit()
        return {"status": "deleted"}
    return {"status": "not found"}