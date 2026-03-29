from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi.responses import FileResponse 

nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

@app.get("/")
def serve_homepage():
    return FileResponse("index.html")

# Allow frontend to talk to backend
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Allow frontend to see the graphs in the 'static' folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load AI
model = joblib.load('fake_news_model.pkl') 
vectorizer = joblib.load('tfidf_vectorizer.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class NewsArticle(BaseModel):
    text: str

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

@app.post("/predict")
def predict_news(article: NewsArticle):
    cleaned_text = preprocess_text(article.text)
    vectorized_text = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    confidence = max(probabilities) * 100 
    
    return {
        "result": "Real News" if prediction == 1 else "Fake News",
        "confidence": round(confidence, 2)
    }