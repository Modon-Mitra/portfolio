from fastapi import APIRouter
from pydantic import BaseModel
import pickle, os
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter()

# ── Load models ──
BASE = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

def load(name):
    path = os.path.join(BASE, name)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

sentiment_model  = load('sentiment_model.pkl')
skill_model      = load('skill_matcher.pkl')
chatbot_model    = load('chatbot.pkl')

# ── Schemas ──
class SkillQuery(BaseModel):
    job_role: str

class SentimentQuery(BaseModel):
    text: str

class ChatQuery(BaseModel):
    question: str

# ── SKILL MATCH ──
@router.post("/skill-match")
def skill_match(query: SkillQuery):
    if not skill_model:
        return {"result": "Model not loaded yet."}
    vec = skill_model['vectorizer'].transform([query.job_role])
    sims = cosine_similarity(vec, skill_model['skill_vectors'])[0]
    ranked = sorted(
        zip(skill_model['skill_names'], sims),
        key=lambda x: x[1], reverse=True
    )[:6]
    matches = [
        {"skill": name, "score": min(round(score * 120), 98)}
        for name, score in ranked if score > 0.03
    ]
    return {"matches": matches, "role": query.job_role}

# ── SENTIMENT ──
@router.post("/sentiment")
def analyze_sentiment(query: SentimentQuery):
    if not sentiment_model:
        return {"sentiment": "Neutral", "score": 0.5}
    pred = sentiment_model.predict([query.text])[0]
    proba = sentiment_model.predict_proba([query.text])[0]
    score = float(max(proba))
    return {"sentiment": pred, "score": round(score, 3)}

# ── CHATBOT ──
@router.post("/chat")
def chatbot(query: ChatQuery):
    if not chatbot_model:
        return {"answer": "Chatbot model not loaded yet!"}
    vec = chatbot_model['vectorizer'].transform([query.question.lower()])
    sims = cosine_similarity(vec, chatbot_model['question_vectors'])[0]
    best = int(sims.argmax())
    if sims[best] < 0.12:
        return {"answer": "I'm not sure about that! You can ask Abhijeet directly at abhijeetsengupta30@gmail.com 📧"}
    return {"answer": chatbot_model['answers'][best]}

# ── INTEREST PREDICTOR ──
@router.post("/predict-interest")
def predict_interest(data: dict):
    clicks = data.get("clicks", {})
    if not clicks:
        return {"predicted_interest": "General"}
    top = max(clicks, key=clicks.get)
    mapping = {
        "projects": "Project Collaboration",
        "ml-tools": "AI / ML Integration",
        "certificates": "Academic Background",
        "about": "General Profile",
        "comments": "Networking"
    }
    return {"predicted_interest": mapping.get(top, "General Interest")}