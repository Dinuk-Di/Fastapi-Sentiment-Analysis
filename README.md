
# Sentiment Analysis API

This project is a FastAPI-based web service that provides sentiment analysis for text. It uses a machine learning model trained on the IMDB Movie Reviews dataset to classify text as either "positive" or "negative" and provides a confidence score for the prediction.

## Project Structure

```
sentiment-analysis/
├── app/
│   ├── main.py           # FastAPI application logic
│   └── ...
├── dataset/
│   └── IMDB Dataset.csv  # The training data
├── model/
│   └── ...               # Saved model files will be generated here
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
├── train.py              # Script to train the sentiment analysis model
└── README.md             # This file
```

## Setup and Installation

### Prerequisites

*   Python 3.10+
*   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd sentiment-analysis
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Training the Model

To train the sentiment analysis model, run the `train.py` script. This script will load the `IMDB Dataset.csv`, preprocess the text data, train a Logistic Regression model, and save the trained pipeline (vectorizer and model) to the `model/` directory.

```bash
python train.py
```
This will create a file named `model/sentiment_pipeline.pkl`.

## Running the API

### Running Locally

To run the FastAPI server locally, use `uvicorn`. The `--reload` flag enables hot-reloading for development.

```bash
uvicorn app.main:app --reload
```
The API will be accessible at `http://127.0.0.1:8000`.

### Running with Docker

You can also build and run the application using Docker.

1.  **Build the Docker image:**
    ```bash
    docker build -t sentiment-analysis .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 sentiment-analysis
    ```
    The API will be accessible at `http://localhost:8000`.

## API Endpoints

### Health Check

Verifies that the service is running.

*   **URL:** `/health`
*   **Method:** `GET`
*   **Response:**
    ```json
    {
      "status": "ok"
    }
    ```

### Predict Sentiment (Single)

Analyzes a single piece of text.

*   **URL:** `/predict`
*   **Method:** `POST`
*   **Request Body:**
    ```json
    {
      "text": "I absolutely love this movie, it was fantastic!"
    }
    ```
*   **`curl` Example:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text": "I absolutely love this movie, it was fantastic!"}'
    ```
*   **Response:**
    ```json
    {
      "text": "I absolutely love this movie, it was fantastic!",
      "sentiment": "positive",
      "confidence": 0.98
    }
    ```

### Predict Sentiment (Batch)

Analyzes a batch of texts in a single request.

*   **URL:** `/predict/batch`
*   **Method:** `POST`
*   **Request Body:**
    ```json
    [
      { "text": "This was a great film." },
      { "text": "I would not recommend this to anyone." }
    ]
    ```
*   **`curl` Example:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict/batch" -H "Content-Type: application/json" -d '[{"text": "This was a great film."}, {"text": "I would not recommend this to anyone."}]'
    ```
*   **Response:**
    ```json
    [
      {
        "text": "This was a great film.",
        "sentiment": "positive",
        "confidence": 0.95
      },
      {
        "text": "I would not recommend this to anyone.",
        "sentiment": "negative",
        "confidence": 0.99
      }
    ]
    ```

## Model and Dataset

*   **Model:** The model is a **Logistic Regression** classifier. It is combined with a **TF-IDF Vectorizer** into a scikit-learn pipeline. This approach was chosen as it provides a strong and interpretable baseline for text classification tasks and is computationally efficient.

*   **Dataset:** The model was trained on the **IMDB Movie Reviews dataset**, which consists of 50,000 labeled movie reviews (25,000 positive, 25,000 negative). It is a standard benchmark dataset for sentiment analysis.

## Approach and Future Work

### Approach

The text preprocessing pipeline involves removing HTML tags, converting text to lowercase, removing non-alphabetic characters and stopwords, and lemmatizing words to their root form. The cleaned text is then converted into numerical features using TF-IDF, which reflects the importance of a word in a document relative to the entire corpus. These features are then used to train the Logistic Regression model.

### Future Work

With more time, the following improvements could be made:

*   **Advanced Models:** Experiment with more sophisticated models like Gradient Boosting (e.g., LightGBM, XGBoost) or deep learning models (e.g., LSTM, GRU, or a fine-tuned Transformer like BERT) for potentially higher accuracy.
*   **Hyperparameter Tuning:** Use techniques like `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for the TF-IDF vectorizer and the classifier.
*   **CI/CD Pipeline:** Set up a continuous integration and deployment pipeline to automate testing and deployment.
