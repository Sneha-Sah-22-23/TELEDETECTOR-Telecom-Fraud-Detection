import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load data 
df = pd.read_csv("data/telecom_fraud.csv")
df.dropna(inplace=True)

# Features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

print(f"Dataset shape: {X.shape}")
print(f"Fraud rate: {y.mean():.2%}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify keeps fraud ratio equal in both splits
)
unseen_data = X_test.copy()
unseen_data['is_fraud'] = y_test
unseen_data.to_csv("data/unseen_test_data.csv", index=False)

X_train = X_train.drop(columns=["phone_number"])
X_test = X_test.drop(columns=["phone_number"])

# MLflow experiment
mlflow.set_experiment("Telecom Fraud Detection")

with mlflow.start_run():

    # Train — class_weight='balanced' handles the 70:30 imbalance
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_proba)

    # Log params
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("class_weight", "balanced")

    # Log metrics
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("roc_auc",   auc)

    # Save model artifact
    mlflow.sklearn.log_model(model, "fraud_model")
    run_id = mlflow.active_run().info.run_id

    print(f"\n── Results ──────────────────────")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Run ID    : {run_id}")

# Register model in MLflow Model Registry 
client = MlflowClient()

# Add description
client.update_registered_model(
    name="TelecomFraudDetector",
    description="Two-layer telecom fraud detection system. Layer 1: ML-based behaviour analysis using Random Forest. Layer 2: SIM farm detection flagging owners with 10+ SIMs across operators. Built with MLflow for DevOps AI deployment."
)

# Add tags
client.set_registered_model_tag("TelecomFraudDetector", "model_type", "RandomForestClassifier")
client.set_registered_model_tag("TelecomFraudDetector", "framework", "MLflow")
client.set_registered_model_tag("TelecomFraudDetector", "domain", "Telecom Fraud Detection")
client.set_registered_model_tag("TelecomFraudDetector", "layers", "Behaviour + SIM Farm")

print("\nModel registered successfully as 'TelecomFraudDetector'!")