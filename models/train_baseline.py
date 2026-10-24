"""
Training Script for Driver State Classification.
Generates comprehensive benchmark telemetry datasets for normal, drowsy, yawning,
and distracted states. Compares multiple ML models (Logistic Regression, Random Forest, SVM, MLP)
and serializes the highest performing model pipeline to disk.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def generate_synthetic_driver_dataset(n_samples_per_class: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic benchmark dataset modeled on physiological ranges and empirical
    distributions from UTA-RLDD and Yawn-Eye datasets.
    Classes:
      0: ALERT (Normal driving)
      1: DROWSY (Microsleep, prolonged eye closure)
      2: YAWNING (Mouth wide open, fatigue)
      3: DISTRACTED (Looking away, checking phone down)
    """
    np.random.seed(random_seed)
    records = []

    # Class 0: ALERT
    for _ in range(n_samples_per_class):
        ear = np.random.normal(0.32, 0.03)
        mar = np.random.normal(0.24, 0.04)
        perclos = np.clip(np.random.beta(1.5, 12.0) * 0.25, 0.0, 0.20)
        blink_rate = np.random.normal(16.0, 3.5)
        yaw = np.random.normal(0.0, 6.0)
        pitch = np.random.normal(0.0, 5.0)
        roll = np.random.normal(0.0, 3.0)
        records.append([ear, mar, perclos, blink_rate, yaw, pitch, roll, 0])

    # Class 1: DROWSY
    for _ in range(n_samples_per_class):
        ear = np.clip(np.random.normal(0.14, 0.035), 0.05, 0.22)
        mar = np.random.normal(0.30, 0.06)
        perclos = np.clip(np.random.normal(0.55, 0.12), 0.35, 0.95)
        blink_rate = np.random.normal(28.0, 6.0)  # High or erratic blinks
        # Driver head starts nodding downwards or drifting slightly
        yaw = np.random.normal(0.0, 8.0)
        pitch = np.random.normal(-12.0, 6.0)
        roll = np.random.normal(2.0, 5.0)
        records.append([ear, mar, perclos, blink_rate, yaw, pitch, roll, 1])

    # Class 2: YAWNING
    for _ in range(n_samples_per_class):
        # Eyes slightly squinted during deep yawn
        ear = np.random.normal(0.22, 0.04)
        mar = np.clip(np.random.normal(0.72, 0.08), 0.58, 0.95)
        perclos = np.clip(np.random.normal(0.25, 0.08), 0.10, 0.40)
        blink_rate = np.random.normal(18.0, 4.0)
        yaw = np.random.normal(0.0, 7.0)
        pitch = np.random.normal(4.0, 6.0)
        roll = np.random.normal(0.0, 4.0)
        records.append([ear, mar, perclos, blink_rate, yaw, pitch, roll, 2])

    # Class 3: DISTRACTED
    for _ in range(n_samples_per_class):
        ear = np.random.normal(0.31, 0.035)
        mar = np.random.normal(0.22, 0.04)
        perclos = np.clip(np.random.beta(1.5, 10.0) * 0.25, 0.0, 0.22)
        blink_rate = np.random.normal(15.0, 3.5)

        # 50% looking away left/right, 50% looking down at phone/lap
        if np.random.rand() > 0.5:
            # Looking left/right
            yaw_sign = 1 if np.random.rand() > 0.5 else -1
            yaw = yaw_sign * np.random.normal(35.0, 8.0)
            pitch = np.random.normal(0.0, 7.0)
        else:
            # Looking down (phone)
            yaw = np.random.normal(0.0, 10.0)
            pitch = np.random.normal(-26.0, 6.0)

        roll = np.random.normal(0.0, 5.0)
        records.append([ear, mar, perclos, blink_rate, yaw, pitch, roll, 3])

    columns = ["ear", "mar", "perclos", "blink_rate", "yaw", "pitch", "roll", "label"]
    df = pd.DataFrame(records, columns=columns)

    # Derived feature: ear_mar_ratio
    df["ear_mar_ratio"] = df["ear"] / (df["mar"] + 1e-5)

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df

def train_and_evaluate_models():
    """Trains and compares multiple classifiers on the driver state dataset."""
    print("=" * 60)
    print("Driver Drowsiness & Distraction Detection - ML Model Training")
    print("=" * 60)

    # 1. Generate Dataset
    print("\n[1/4] Generating driver telemetry dataset (6,000 samples across 4 classes)...")
    df = generate_synthetic_driver_dataset(n_samples_per_class=1500)

    feature_cols = ["ear", "mar", "perclos", "blink_rate", "yaw", "pitch", "roll", "ear_mar_ratio"]
    X = df[feature_cols].values
    y = df["label"].values

    print(f"Features: {feature_cols}")
    print(f"Class distribution:\n{df['label'].value_counts().to_dict()}")

    # 2. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[2/4] Split data: Train samples = {len(X_train)}, Test samples = {len(X_test)}")

    # 3. Model Definition
    models = {
        "Logistic Regression (Linear)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Support Vector Classifier (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "Random Forest Classifier (Ensemble)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42))
        ]),
        "Multi-Layer Perceptron (MLP Neural Net)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
        ])
    }

    print("\n[3/4] Evaluating models using 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = 0.0
    best_model_name = None
    best_pipeline = None

    results = {}
    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
        mean_f1 = scores.mean()
        std_f1 = scores.std()
        results[name] = mean_f1
        print(f"  > {name:38s}: Macro F1 = {mean_f1:.4f} (+/- {std_f1:.4f})")

        if mean_f1 > best_score:
            best_score = mean_f1
            best_model_name = name
            best_pipeline = pipeline

    # 4. Train Best Model and Evaluate on Test Set
    print(f"\n[4/4] Fitting best model: '{best_model_name}' (CV Macro F1 = {best_score:.4f})")
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")

    class_names = ["ALERT", "DROWSY", "YAWNING", "DISTRACTED"]
    print(f"\nTest Set Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Set Macro F1 : {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importance for Random Forest
    rf_clf = best_pipeline.named_steps.get("clf")
    if hasattr(rf_clf, "feature_importances_"):
        print("\nFeature Importances (Gini Importance):")
        importances = rf_clf.feature_importances_
        for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
            print(f"  - {feat:16s}: {imp:.4f}")

    # Serialize best model
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, str(config.MODEL_PATH))
    print(f"\n[OK] Successfully saved model pipeline to: {config.MODEL_PATH}")

    # Also save benchmark dataset CSV for notebooks
    csv_path = config.BASE_DIR / "models" / "benchmark_driver_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved dataset for Jupyter exploration: {csv_path}")

    return best_pipeline

if __name__ == "__main__":
    train_and_evaluate_models()
