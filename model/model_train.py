import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import os
import boto3
from io import StringIO
import matplotlib.pyplot as plt
import base64
from io import BytesIO

MODEL_DIR = "/home/lead_priority/model"
DOWNLOADS_DIR = "downloads"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "x_columns.pkl")


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def train_model_from_s3(access_key, secret_key, s3_uri):
    try:
        assert s3_uri.startswith("s3://")
        bucket_name, object_key = s3_uri.replace("s3://", "").split("/", 1)
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        s3 = session.client("s3")
        obj = s3.get_object(Bucket=bucket_name, Key=object_key)
        df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

        # Encode categoricals
        encoders = {}
        for col in ["drop_off_step", "event_recency_bucket"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        joblib.dump(encoders, ENCODER_PATH)

        # Drop irrelevant fields
        drop_cols = ['first_name', 'last_name', 'address', 'mobile', 'drivers_license', 'dob', 'employer']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        X = df.drop(columns=["converted"])
        y = df["converted"]
        joblib.dump(list(X.columns), COLUMNS_PATH)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

        # Evaluation metrics
        y_scores = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_scores)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Lift calculation
        def compute_lift(y_true, y_scores, percentile=0.1):
            df_lift = pd.DataFrame({'y_true': y_true, 'score': y_scores})
            df_sorted = df_lift.sort_values('score', ascending=False)
            cutoff = int(len(df_lift) * percentile)
            top_k = df_sorted.iloc[:cutoff]
            baseline = df_lift['y_true'].mean()
            lift = top_k['y_true'].mean() / baseline
            return lift, top_k

        lift_score, top_10_df = compute_lift(y_test.values, y_scores)

        importances = model.feature_importances_
        feature_names = X.columns

        # Bucketing
        buckets = pd.cut(y_scores, bins=[0, 0.5, 0.8, 1], labels=['Low', 'Medium', 'High'])
        bucket_counts = buckets.value_counts().sort_index()

        top_10_df['bucket'] = pd.cut(top_10_df['score'], bins=[0, 0.5, 0.8, 1], labels=['Low', 'Medium', 'High'])
        top_10_df.to_json(os.path.join(DOWNLOADS_DIR, "top_10_leads.json"), orient="records", lines=True)
        top_10_df.to_excel(os.path.join(DOWNLOADS_DIR, "top_10_leads.xlsx"), index=False)

        top_10_df[top_10_df['bucket'] == 'High'].to_excel(os.path.join(DOWNLOADS_DIR, "high_bucket_leads.xlsx"), index=False)
        top_10_df[top_10_df['bucket'] == 'Medium'].to_excel(os.path.join(DOWNLOADS_DIR, "medium_bucket_leads.xlsx"), index=False)
        top_10_df[top_10_df['bucket'] == 'Low'].to_excel(os.path.join(DOWNLOADS_DIR, "low_bucket_leads.xlsx"), index=False)

        # ROC Curve
        fig1, ax1 = plt.subplots()
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        ax1.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        ax1.set_title("ROC Curve")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend()
        roc_img = fig_to_base64(fig1)

        # Lift Curve
        fig2, ax2 = plt.subplots()
        ax2.bar(['Lift@10%'], [lift_score])
        ax2.set_title("Lift Curve")
        ax2.set_ylabel("Lift")
        lift_img = fig_to_base64(fig2)

        # Feature Importance
        fig3, ax3 = plt.subplots()
        ax3.barh(feature_names, importances)
        ax3.set_title("Feature Importance")
        ax3.set_xlabel("Importance")
        plt.tight_layout()
        importance_img = fig_to_base64(fig3)

        # Bucket Distribution
        fig4, ax4 = plt.subplots()
        bucket_counts.plot(kind='bar', ax=ax4)
        ax4.set_title("Bucket Distribution")
        ax4.set_ylabel("Count")
        ax4.set_xlabel("Bucket")
        bucket_img = fig_to_base64(fig4)

        return {
            "status": "success",
            "AUC": round(auc_score, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "plots": {
                "roc_curve": roc_img,
                "lift_curve": lift_img,
                "feature_importance": importance_img,
                "bucket_distribution": bucket_img
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Example manual test
    result = train_model_from_s3("your-access-key", "your-secret-key", "s3://your-bucket/payday_loan_leads.csv")
    print(result)
