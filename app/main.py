
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize FastAPI app
app = FastAPI()

# Load the trained model pipeline
pipeline = joblib.load('model/sentiment_pipeline.pkl')

# --- Pydantic Schemas ---
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class BatchSentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

# --- Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# --- API Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    # Preprocess the input text
    preprocessed_text = preprocess_text(request.text)

    # Predict sentiment and confidence
    probabilities = pipeline.predict_proba([preprocessed_text])[0]
    prediction = pipeline.predict([preprocessed_text])[0]

    sentiment = "positive" if prediction == 1 else "negative"
    confidence = float(max(probabilities))

    return SentimentResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=confidence
    )

@app.post("/predict/batch", response_model=list[BatchSentimentResponse])
def predict_batch(requests: list[SentimentRequest]):
    responses = []
    texts = [r.text for r in requests]
    
    # Preprocess all texts
    preprocessed_texts = [preprocess_text(text) for text in texts]

    # Get predictions and probabilities
    predictions = pipeline.predict(preprocessed_texts)
    probabilities = pipeline.predict_proba(preprocessed_texts)

    for i, text in enumerate(texts):
        sentiment = "positive" if predictions[i] == 1 else "negative"
        confidence = float(max(probabilities[i]))
        responses.append(
            BatchSentimentResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence
            )
        )
    return responses
