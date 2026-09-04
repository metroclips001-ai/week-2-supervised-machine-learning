# Week 2 Task: Supervised Machine Learning Models
# Models: Linear Regression, Logistic Regression, Decision Tree,
# Random Forest, and K-Nearest Neighbors (KNN).

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score,
    recall_score, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_linear_regression():
    """Train and evaluate Linear Regression."""
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("LINEAR REGRESSION")
    print("=" * 60)
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    }).to_csv(OUTPUT_DIR / "linear_regression_predictions.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression: Actual vs Predicted")
    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())
    plt.plot([low, high], [low, high])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "linear_regression_actual_vs_predicted.png", dpi=200)
    plt.close()

    return {
        "Model": "Linear Regression",
        "Task": "Regression",
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    }


def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a classification model."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    ).to_csv(
        OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.csv"
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{name} - Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png",
        dpi=200
    )
    plt.close()

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


def main():
    run_linear_regression()

    # Classification dataset
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    results = []

    for name, model in models.items():
        results.append(
            evaluate_classifier(
                name, model, X_train, X_test, y_train, y_test
            )
        )

    results_df = pd.DataFrame(results).sort_values(
        "F1 Score", ascending=False
    )
    results_df.to_csv(
        OUTPUT_DIR / "classification_model_comparison.csv", index=False
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.bar(results_df["Model"], results_df["F1 Score"])
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.title("Classification Model Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classification_model_comparison.png", dpi=200)
    plt.close()

    print("\nAll models were trained and evaluated successfully.")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
