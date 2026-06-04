from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pickle, os

# Training data — feel free to add more examples!
data = [
    # Positive
    ("Amazing work! Very impressive portfolio.", "Positive"),
    ("Great projects, especially the quantum cryptography one!", "Positive"),
    ("Excellent skills and well presented.", "Positive"),
    ("Really loved the SurakshaAI concept, brilliant idea.", "Positive"),
    ("Outstanding achievement for a student!", "Positive"),
    ("Incredible work on the UK registered design.", "Positive"),
    ("Very talented developer, would love to collaborate.", "Positive"),
    ("This is genuinely impressive for an MCA student.", "Positive"),
    ("Loved the multi-agent architecture explanation.", "Positive"),
    ("Fantastic portfolio, best I have seen today.", "Positive"),
    ("Great job, keep it up!", "Positive"),
    ("Superb work, very professional.", "Positive"),
    ("Loved the design and the projects.", "Positive"),
    ("Brilliant mind behind these projects.", "Positive"),
    ("Top notch work, very well done.", "Positive"),

    # Neutral
    ("Interesting projects, would like to know more.", "Neutral"),
    ("Good portfolio, needs some more details.", "Neutral"),
    ("Nice work, looking forward to seeing more projects.", "Neutral"),
    ("Decent work overall.", "Neutral"),
    ("The portfolio is okay, has room for improvement.", "Neutral"),
    ("Some projects are interesting.", "Neutral"),
    ("Average presentation but good content.", "Neutral"),
    ("Not bad, could be better.", "Neutral"),
    ("Reasonable skills for the level of study.", "Neutral"),
    ("Seems like a promising student.", "Neutral"),

    # Negative
    ("The portfolio needs a lot of improvement.", "Negative"),
    ("Not very impressive honestly.", "Negative"),
    ("Could have done better on the presentation.", "Negative"),
    ("The projects lack depth and documentation.", "Negative"),
    ("Disappointing for an MCA student.", "Negative"),
    ("Very basic work, nothing special.", "Negative"),
    ("The design looks outdated.", "Negative"),
    ("Skills seem limited for the roles being targeted.", "Negative"),
    ("Not convinced by the project quality.", "Negative"),
    ("Needs significant improvement overall.", "Negative"),
]

texts = [d[0] for d in data]
labels = [d[1] for d in data]

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000, C=5.0))
])

pipeline.fit(texts, labels)

# Save model
os.makedirs('saved_models', exist_ok=True)
with open('saved_models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Sentiment model trained and saved!")

# Quick test
tests = [
    "This is amazing work!",
    "Not very impressive",
    "Interesting projects overall"
]
for t in tests:
    pred = pipeline.predict([t])[0]
    prob = max(pipeline.predict_proba([t])[0])
    print(f"  '{t}' → {pred} ({round(prob*100)}%)")