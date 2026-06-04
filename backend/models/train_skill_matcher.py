from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle, os

# Abhijeet's skills with descriptions (the richer the description, the better the matching)
skills = {
    "Multi-Agent AI & LangGraph": "multi agent AI systems LangGraph orchestration autonomous agents decision making workflow automation intelligent coordination gemini API",
    "Quantum Cryptography (QKD/PQC)": "quantum key distribution post quantum cryptography Qiskit quantum computing security encryption hybrid cryptographic algorithm CRIQ simulation",
    "Android Development (Java)": "Android Java mobile app development SDK XML UI jetpack components WorkManager view binding data binding mobile application",
    "Python Programming": "Python scripting backend development data processing automation machine learning AI development Flask FastAPI",
    "TensorFlow Lite & ML": "TensorFlow Lite machine learning model training inference mobile ML neural networks classification threat detection",
    "Firebase & Cloud Services": "Firebase authentication Firestore storage cloud backend real time database Supabase cloud services serverless",
    "Web Development": "HTML CSS JavaScript Tailwind CSS frontend web development REST API integration Leaflet.js web application",
    "Database Management": "MySQL PostgreSQL database SQL queries schema design data management SQLAlchemy ORM",
    "Cybersecurity & Cryptography": "cybersecurity encryption secure communication data security authentication access control permission based security threat assessment",
    "API Integration": "REST API Gemini API Claude API Groq API Google Maps API third party integration HTTP requests JSON",
    "FastAPI & Backend": "FastAPI Python backend server API development CORS middleware routing Uvicorn server side",
    "Location & Mapping": "Google Maps Platform GPS tracking location services live tracking geolocation Leaflet mapping spatial",
}

# Job roles → what skills they typically need
job_role_keywords = {
    "ai engineer": "artificial intelligence machine learning deep learning neural network model training automation agents",
    "ml engineer": "machine learning model training tensorflow pytorch scikit learn data pipeline deployment inference",
    "android developer": "android java kotlin mobile development SDK UI design app development jetpack",
    "backend developer": "backend server API REST database python fastapi flask node SQL authentication",
    "cybersecurity analyst": "security encryption threat detection vulnerability penetration testing cryptography authentication firewall",
    "quantum computing researcher": "quantum computing qiskit quantum algorithms cryptography simulation quantum key distribution",
    "full stack developer": "frontend backend HTML CSS JavaScript database API integration web development server",
    "software engineer": "programming software development algorithms data structures problem solving backend frontend",
    "data scientist": "data analysis machine learning python statistics visualization pandas numpy model training",
    "cloud engineer": "cloud services firebase supabase deployment serverless backend infrastructure API",
}

skill_names = list(skills.keys())
skill_descriptions = list(skills.values())

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
vectorizer.fit(skill_descriptions + list(job_role_keywords.values()))
skill_vectors = vectorizer.transform(skill_descriptions)

os.makedirs('saved_models', exist_ok=True)
with open('saved_models/skill_matcher.pkl', 'wb') as f:
    pickle.dump({
        'vectorizer': vectorizer,
        'skill_vectors': skill_vectors,
        'skill_names': skill_names,
        'skill_descriptions': skill_descriptions,
    }, f)

print("✅ Skill matcher model saved!")

# Test it
def match_skills(job_role, top_n=5):
    vec = vectorizer.transform([job_role])
    sims = cosine_similarity(vec, skill_vectors)[0]
    ranked = sorted(zip(skill_names, sims), key=lambda x: x[1], reverse=True)[:top_n]
    return [(name, round(score*100)) for name, score in ranked if score > 0.05]

tests = ["AI Engineer", "Cybersecurity Analyst", "Android Developer"]
for t in tests:
    print(f"\n  Role: {t}")
    for skill, score in match_skills(t):
        print(f"    {skill}: {score}%")