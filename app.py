# app.py
from flask import Flask, render_template, request, jsonify
from main import IrisModel

app = Flask(__name__)
iris_model = IrisModel()
class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# Serve HTML page
@app.route("/")
def index():
    return render_template("index.html")

# Handle prediction POST requests
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    try:
        sl = float(data["sl"])
        sw = float(data["sw"])
        pl = float(data["pl"])
        pw = float(data["pw"])
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    prediction = iris_model.predict([sl, sw, pl, pw])
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
