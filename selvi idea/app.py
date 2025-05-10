from flask import Flask, request, render_template_string
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(_name_)

TEMPLATE = """
<!doctype html>
<title>Sentiment Analyzer</title>
<h2>Enter a sentence to analyze sentiment</h2>
<form method="POST">
  <textarea name="text" rows="4" cols="50"></textarea><br><br>
  <input type="submit" value="Analyze">
</form>
{% if result %}
  <h3>Sentiment: {{ result }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        input_text = request.form["text"]
        transformed = vectorizer.transform([input_text])
        prediction = model.predict(transformed)[0]
        result = "Positive" if prediction == 1 else "Negative"
    return render_template_string(TEMPLATE, result=result)

if _name_ == "_main_":
    app.run()