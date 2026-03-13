from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
import os
from schemas import SentimentRequest, SentimentResponse


load_dotenv()
IP_ADDRESS = os.getenv("IP_ADDRESS", "[IP_ADDRESS]")
APP_PORT = os.getenv("APP_PORT", "8000")


app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: SentimentRequest):
    text = request.text
    # Dummy sentiment analysis logic
    if "good" in text.lower():
        sentiment = "positive"
    elif "bad" in text.lower():
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return SentimentResponse(sentiment=sentiment)

@app.post("/predict/batch")
def predict_batch(requests: list[SentimentRequest]):
    responses = []
    for request in requests:
        text = request.text
        if "good" in text.lower():
            sentiment = "positive"
        elif "bad" in text.lower():
            sentiment = "negative"
        else:
            sentiment = "neutral"
        responses.append(SentimentResponse(sentiment=sentiment))
    return responses


if __name__ == "__main__":
    uvicorn.run(app, host=IP_ADDRESS, port=int(APP_PORT))