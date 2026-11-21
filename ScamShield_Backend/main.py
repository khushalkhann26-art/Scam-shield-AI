from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import requests

app = FastAPI(title="ScamShield AI Backend")

# Load trained ML model and TF-IDF vectorizer
model = joblib.load("scamshield_model.joblib")
vectorizer = joblib.load("scamshield_tfidf_vectorizer.joblib")

# Gemini key will be added later in Stage 2
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class MessageRequest(BaseModel):
    message: str


def find_warning_words(message: str):
    text = message.lower()

    keywords = [
        "otp",
        "cnic",
        "bisp",
        "benazir",
        "ehsaas",
        "password",
        "bank details",
        "verify",
        "blocked",
        "click",
        "urgent",
        "payment",
        "registration fee",
        "prize",
        "winner",
        "claim",
        "tax refund",
        "account suspended",
        "parcel",
        "delivery fee",
        "investment",
        "guaranteed"
    ]

    found_words = []

    for word in keywords:
        if word in text:
            found_words.append(word)

    return found_words


def local_explanation(message: str, prediction: str, risk: str, warning_words: list):
    if prediction == "scam":
        if warning_words:
            return (
                "This message looks suspicious because it contains warning signs such as "
                + ", ".join(warning_words)
                + ". Do not click unknown links or share CNIC, OTP, passwords, bank details, or card information."
            )
        return (
            "This message looks suspicious. Do not click unknown links or share private information. "
            "Verify the message from the official source."
        )

    return (
        "This message looks safe based on the current model prediction. "
        "However, still be careful with unknown senders and never share OTP, password, CNIC, or bank details."
    )


def gemini_explanation(message: str, prediction: str, risk: str, warning_words: list):
    """
    Stage 2 function:
    If Gemini API key exists, use Gemini.
    If not, use local explanation.
    """

    if not GEMINI_API_KEY:
        return local_explanation(message, prediction, risk, warning_words)

    prompt = f"""
You are ScamShield AI, a mobile cyber safety assistant.

A user received this message:
"{message}"

Machine Learning prediction: {prediction}
Risk level: {risk}
Warning words found: {warning_words}

Explain in simple mobile-user language:
1. Why this message may be risky or safe
2. The main warning signs
3. What the user should do next

Keep it short, clear, and practical.
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    except Exception:
        return local_explanation(message, prediction, risk, warning_words)


@app.get("/")
def home():
    return {
        "message": "ScamShield AI Backend is running"
    }


@app.post("/predict")
def predict(request: MessageRequest):
    message = request.message.strip()

    if not message:
        return {
            "prediction": "unknown",
            "risk": "Unknown",
            "confidence": 0,
            "message": message,
            "warning_words": [],
            "explanation": "No message was provided."
        }

    # Find risky words
    warning_words = find_warning_words(message)

    # ML prediction
    message_vector = vectorizer.transform([message])
    ml_prediction = model.predict(message_vector)[0]

    # Confidence score
    probabilities = model.predict_proba(message_vector)[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(ml_prediction)] * 100)

    final_prediction = ml_prediction

    # Extra safety layer for strong scam indicators
    strong_scam_words = [
        "otp",
        "cnic",
        "bisp",
        "benazir",
        "ehsaas",
        "password",
        "bank details"
    ]

    for word in strong_scam_words:
        if word in warning_words:
            final_prediction = "scam"

    # Risk level
    if final_prediction == "scam":
        if confidence >= 75 or len(warning_words) >= 2:
            risk = "High"
        else:
            risk = "Medium"
    else:
        risk = "Low"

    # Explanation
    explanation = gemini_explanation(message, final_prediction, risk, warning_words)

    return {
        "prediction": final_prediction,
        "risk": risk,
        "confidence": round(confidence, 2),
        "message": message,
        "warning_words": warning_words,
        "explanation": explanation
    }