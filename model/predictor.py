import pandas as pd
import joblib
from config import MODEL_PATH
import os

ENCODER_PATH = "/home/lead_priority/model/encoders.pkl"
COLUMNS_PATH = "/home/lead_priority/model/x_columns.pkl"

def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    x_columns = joblib.load(COLUMNS_PATH)
    return model, encoders, x_columns

def preprocess(df, encoders, x_columns):
    for col in x_columns:
        if col not in df.columns:
            df[col] = "unknown" if col in encoders else 0

    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))

    df = df[x_columns]
    return df

def score_and_bucket(trail_file_path, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    model, encoders, x_columns = load_model_and_metadata()

    df_raw = pd.read_csv(trail_file_path)
    df_processed = preprocess(df_raw.copy(), encoders, x_columns)

    probabilities = model.predict_proba(df_processed)[:, 1]
    df_raw["probability"] = probabilities
    df_raw["bucket"] = pd.cut(probabilities, bins=[0, 0.5, 0.8, 1], labels=["Low", "Medium", "High"])

    # Save outputs
    output_csv = os.path.join(output_dir, "trail_predictions.csv")
    output_json = os.path.join(output_dir, "trail_predictions.json")
    df_raw.to_csv(output_csv, index=False)
    df_raw.to_json(output_json, orient="records", lines=True)

    # Save top 10 HTML
    top10_df = df_raw.sort_values("probability", ascending=False).head(10)
    output_html = os.path.join(output_dir, "top10_leads.html")
    top10_df.to_html(output_html, index=False)

    return output_csv, output_json, output_html
