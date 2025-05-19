from flask import Flask, request, jsonify, render_template,  send_from_directory
import joblib
import os
import pandas as pd
from model.model_train import train_model_from_s3
from model.predictor import score_and_bucket  # ✅ ADD THIS LINE

app = Flask(__name__)


app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
MODEL_DIR = "G:/MVP/mnt/AGENTICAI_MODELS/loan_prioritization/model"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "x_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        data = request.json
        result = train_model_from_s3(data["access_key"], data["secret_key"], data["s3_uri"])
        return jsonify(result)  # 👈 Always JSON

    except Exception as e:
        import traceback
        print("❌ Exception:", traceback.format_exc())  # Show real error in terminal
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/score", methods=["POST"])
def score():
    try:
        input_data = request.get_json()

        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)
        x_columns = joblib.load(COLUMNS_PATH)

        for col in x_columns:
            if col not in input_data:
                input_data[col] = "unknown" if col in encoders else 0

        df = pd.DataFrame([input_data])

        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))

        df = df[x_columns]

        prob = model.predict_proba(df)[0][1]
        bucket = "High" if prob > 0.8 else "Medium" if prob > 0.5 else "Low"

        return jsonify({"probability": round(prob * 100, 2), "bucket": bucket})
    except Exception as e:
        return jsonify({"error": str(e), "probability": 0.0, "bucket": "undefined"}), 500
    
@app.route("/predict_trail", methods=["POST"])
def predict_trail():
    try:
        file = request.files["trail_file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"status": "error", "message": "Only .csv files are accepted"}), 400

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        csv_path, json_path, html_path = score_and_bucket(filepath)

        return jsonify({
            "status": "success",
            "csv_path": f"/download/{os.path.basename(csv_path)}",
            "json_path": f"/download/{os.path.basename(json_path)}",
            "html_path": f"/download/{os.path.basename(html_path)}"
        })

    except Exception as e:
        import traceback
        print("❌ Error:\n", traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory("downloads", filename, as_attachment=False)
if __name__ == "__main__":
    app.run(debug=True, port=7000)
