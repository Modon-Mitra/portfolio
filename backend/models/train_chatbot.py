from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle, os

# Q&A knowledge base about Abhijeet
qa_pairs = [
    # Identity
    ("who are you", "I'm Abhijeet Sengupta — an MCA student at BP Poddar Institute of Management and Technology, Kolkata. I specialize in AI systems, quantum cryptography, and Android development."),
    ("what is your name", "My name is Abhijeet Sengupta. I'm an MCA student and AI/security developer based in Kolkata, India."),
    ("where are you from", "I'm from Sodepur, Kolkata, West Bengal, India."),
    ("tell me about yourself", "I'm Abhijeet Sengupta, an MCA student passionate about building intelligent and secure systems. I've created a UK-registered cryptographic algorithm and an AI-powered women safety app called SurakshaAI."),

    # Education
    ("what are you studying", "I'm currently pursuing a Master of Computer Applications (MCA) at BP Poddar Institute of Management and Technology, Kolkata."),
    ("what is your educational background", "I completed my BCA from Narula Institute of Technology with an SGPA of 8.10 (2021-2024), and I'm currently pursuing MCA at BP Poddar Institute of Management and Technology."),
    ("what college do you go to", "I study at BP Poddar Institute of Management and Technology, Kolkata for my MCA degree."),

    # Projects
    ("what projects have you built", "I've built two major projects: SurakshaAI — an AI-powered women safety app with multi-agent architecture, and a Hybrid QKD-PQC Cryptographic Algorithm registered as UK Design No: 6505034. I also built a Unified AI Energy Management System using LangGraph."),
    ("tell me about surakshaai", "SurakshaAI is an AI-powered women safety and emergency response system I built for Android. It uses multi-agent AI architecture to detect threats, trigger smart SOS, share live location, collect evidence, and provide emotional support — all automatically with minimal user interaction. It uses Java, TensorFlow Lite, Firebase, Claude API, and Google Maps."),
    ("what is the quantum cryptography project", "It's a Hybrid QKD-PQC cryptographic algorithm that combines Quantum Key Distribution and Post-Quantum Cryptography to secure communication against quantum-era threats. It was simulated using Qiskit and is officially registered as UK Design No: 6505034."),
    ("what is the energy management project", "The Unified AI Energy Management System is a web platform with 6 specialized AI agents built using LangGraph and Gemini API. The agents handle energy generation, storage, distribution, consumption, maintenance, and pricing, with real-time weather integration."),

    # Skills
    ("what are your technical skills", "My key skills include Python, JavaScript, Java for Android, Multi-Agent AI with LangGraph, Quantum Cryptography with Qiskit, TensorFlow Lite, Firebase, FastAPI, PostgreSQL, MySQL, and REST API integration."),
    ("do you know python", "Yes! Python is one of my strongest languages. I use it for AI development, backend systems with FastAPI, ML model training, and quantum computing simulations with Qiskit."),
    ("do you know android development", "Yes, I've built SurakshaAI — a full-featured Android app in Java using the Android SDK, Jetpack components, TensorFlow Lite, Firebase, and Google Maps Platform."),
    ("what ai technologies do you know", "I work with Multi-Agent AI using LangGraph, TensorFlow Lite for mobile ML, Claude API, Groq API, Gemini API, and I have academic exposure to fundamental ML concepts."),
    ("do you know quantum computing", "Yes! I've implemented a Hybrid QKD-PQC cryptographic algorithm simulated on Qiskit. My work has been registered as a UK Intellectual Property Design — it's one of my most unique achievements."),

    # Achievements
    ("what are your achievements", "My top achievements include: 1) A UK Registered Design for my Hybrid QKD-PQC cryptographic algorithm (Design No: 6505034), 2) Building SurakshaAI — a production-ready AI safety platform, 3) Completing BCA with an SGPA of 8.10."),
    ("do you have any patents or registrations", "Yes! My Hybrid QKD-PQC Cryptographic Algorithm is registered as UK Design No: 6505034 with the UK Intellectual Property Office. This is one of my proudest achievements."),

    # Contact
    ("how can i contact you", "You can reach me at abhijeetsengupta30@gmail.com or call +91 8583879579. You can also connect with me on LinkedIn: linkedin.com/in/aviijit-sengupta-3714b125b/"),
    ("what is your email", "My email is abhijeetsengupta30@gmail.com — feel free to reach out!"),
    ("what is your phone number", "You can reach me at +91 8583879579."),
    ("are you available for work", "Yes! I'm actively looking for entry-level opportunities in software development, AI systems, or cybersecurity. Feel free to get in touch at abhijeetsengupta30@gmail.com"),

    # Interests
    ("what are your hobbies", "Outside of coding, I enjoy driving cars and nature photography. I find that photography trains my eye for detail, which helps in UI design too!"),
    ("what languages do you speak", "I speak Bengali (native), Hindi, and English fluently."),
]

questions = [p[0] for p in qa_pairs]
answers = [p[1] for p in qa_pairs]

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
question_vectors = vectorizer.fit_transform(questions)

os.makedirs('saved_models', exist_ok=True)
with open('saved_models/chatbot.pkl', 'wb') as f:
    pickle.dump({
        'vectorizer': vectorizer,
        'question_vectors': question_vectors,
        'answers': answers,
        'questions': questions,
    }, f)

print("✅ Chatbot model saved!")

# Test
def ask(question, threshold=0.15):
    vec = vectorizer.transform([question.lower()])
    sims = cosine_similarity(vec, question_vectors)[0]
    best_idx = sims.argmax()
    if sims[best_idx] < threshold:
        return "I'm not sure about that yet! You can ask Abhijeet directly at abhijeetsengupta30@gmail.com"
    return answers[best_idx]

tests = ["Tell me about SurakshaAI", "What skills do you have?", "How can I hire you?"]
for t in tests:
    print(f"\n  Q: {t}\n  A: {ask(t)[:80]}...")