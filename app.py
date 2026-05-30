from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    transformed_news = vectorizer.transform([news])

    prediction = model.predict(transformed_news)[0]

    print("Prediction =", prediction)
    print("Probabilities =", model.predict_proba(transformed_news))

    confidence = max(model.predict_proba(transformed_news)[0]) * 100

    if prediction == 0:
        result = "🚨 FAKE NEWS"
    else:
        result = "✅ REAL NEWS"

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)