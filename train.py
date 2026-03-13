
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from bs4 import BeautifulSoup

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('dataset/IMDB Dataset.csv')

# --- Text Preprocessing ---
print("Preprocessing text data...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove non-letters and convert to lower case
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    # Tokenize
    words = text.split()
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['review'] = df['review'].apply(preprocess_text)

# --- Model Training ---
print("Training model...")

# Convert labels to binary
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split data
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(C=1.0, solver='liblinear'))
])

# Train the model
pipeline.fit(X_train, y_train)

# --- Save Model ---
print("Saving model and vectorizer...")
joblib.dump(pipeline, 'model/sentiment_pipeline.pkl')

# --- Evaluation ---
print("Evaluating model...")
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

print("Training complete and model saved.")
