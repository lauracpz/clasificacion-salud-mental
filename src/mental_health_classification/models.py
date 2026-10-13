"""Preparación, entrenamiento y evaluación de los modelos."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Depression"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
DECISION_THRESHOLD = 0.50

BASELINE_FEATURES = [
    "Have you ever had suicidal thoughts ?",
    "Sleep Duration",
    "Financial Stress",
]
ALL_FEATURES = [
    "Gender",
    "Age",
    "Working Professional or Student",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
    "Sleep Duration",
    "Work/Study Pressure",
    "Work/Study Satisfaction",
]
FEATURE_SETS = {
    "Base (3 variables)": BASELINE_FEATURES,
    "Completo (10 variables)": ALL_FEATURES,
}
REQUIRED_COLUMNS = [
    "Gender",
    "Age",
    "Working Professional or Student",
    "Academic Pressure",
    "Work Pressure",
    "Study Satisfaction",
    "Job Satisfaction",
    "Sleep Duration",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
]
CATEGORY_MAPPINGS = {
    "Gender": {"female": 0, "male": 1},
    "Working Professional or Student": {
        "student": 0,
        "working professional": 1,
    },
    "Have you ever had suicidal thoughts ?": {"no": 0, "yes": 1},
    "Family History of Mental Illness": {"no": 0, "yes": 1},
}


def map_sleep_duration(value: object) -> float:
    """Convierte rangos de sueño a horas aproximadas; lo desconocido pasa a NaN."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().lower()
    patterns = (
        (r"more than\s+(\d+)\s+hours", lambda hours: hours + 1),
        (r"less than\s+(\d+)\s+hours", lambda hours: hours - 1),
        (r"(\d+)\s*-\s*(\d+)\s+hours", lambda hours: hours),
    )
    for pattern, convert in patterns:
        match = re.search(pattern, text)
        if match:
            return float(convert(int(match.group(1))))
    return np.nan


def prepare_dataset(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Valida y transforma los datos para entrenar los modelos."""
    required = set(REQUIRED_COLUMNS + [TARGET])
    missing = sorted(required.difference(raw_data.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {missing}")

    data = raw_data.copy()
    data["Work/Study Pressure"] = pd.to_numeric(
        data["Academic Pressure"], errors="coerce"
    ).combine_first(pd.to_numeric(data["Work Pressure"], errors="coerce"))
    data["Work/Study Satisfaction"] = pd.to_numeric(
        data["Study Satisfaction"], errors="coerce"
    ).combine_first(pd.to_numeric(data["Job Satisfaction"], errors="coerce"))

    for column, mapping in CATEGORY_MAPPINGS.items():
        normalized = data[column].astype("string").str.strip().str.lower()
        data[column] = normalized.map(mapping)

    data["Sleep Duration"] = data["Sleep Duration"].apply(map_sleep_duration)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data.loc[~data[TARGET].isin([0, 1]), TARGET] = np.nan

    for column in ALL_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    cleaned = data[ALL_FEATURES + [TARGET]].dropna().copy()
    if cleaned.empty:
        raise ValueError("No quedan filas completas después de preparar los datos.")
    return cleaned


def build_model() -> Pipeline:
    """Crea un escalado y una regresión logística reproducible."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
    )


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    feature_columns: list[str],
) -> tuple[dict[str, float], np.ndarray]:
    """Calcula métricas y matriz de confusión para un modelo."""
    probabilities = model.predict_proba(features[feature_columns])[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(target, predictions),
        "balanced_accuracy": balanced_accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "roc_auc": roc_auc_score(target, probabilities),
    }
    return metrics, matrix


def save_confusion_matrices(
    matrices: dict[str, np.ndarray], output_dir: Path
) -> None:
    """Guarda las matrices de confusión de los modelos comparados."""
    figure, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 5))
    axes = np.atleast_1d(axes)
    for axis, (name, matrix) in zip(axes, matrices.items()):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No depresión", "Depresión"],
            yticklabels=["No depresión", "Depresión"],
            ax=axis,
        )
        axis.set_title(name)
        axis.set_xlabel("Predicción")
        axis.set_ylabel("Valor real")

    figure.suptitle(f"Matrices de confusión (umbral = {DECISION_THRESHOLD:.2f})")
    figure.tight_layout()
    figure.savefig(output_dir / "matrices_confusion.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_experiment(dataset: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Entrena, compara y guarda resultados para ambos conjuntos de variables."""
    features = dataset[ALL_FEATURES]
    target = dataset[TARGET].astype(int)
    class_counts = target.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("Se necesitan al menos dos ejemplos de cada clase.")

    x_train, x_validation, y_train, y_validation = train_test_split(
        features,
        target,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    if y_validation.nunique() != 2:
        raise ValueError(
            "La partición de validación debe contener ambas clases; "
            "se necesitan más datos."
        )

    metric_rows = []
    matrices = {}
    fitted_models = {}
    for model_name, feature_columns in FEATURE_SETS.items():
        model = build_model()
        model.fit(x_train[feature_columns], y_train)
        metrics, matrix = evaluate_model(
            model,
            x_validation,
            y_validation,
            feature_columns,
        )
        metric_rows.append(
            {
                "modelo": model_name,
                "variables": len(feature_columns),
                **metrics,
            }
        )
        matrices[model_name] = matrix
        fitted_models[model_name] = model

    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False)
    results.to_csv(output_dir / "metricas.csv", index=False)
    save_confusion_matrices(matrices, output_dir)

    complete_model = fitted_models["Completo (10 variables)"]
    coefficients = complete_model.named_steps["logisticregression"].coef_[0]
    coefficient_table = pd.DataFrame(
        {
            "variable": ALL_FEATURES,
            "coeficiente_estandarizado": coefficients,
        }
    )
    coefficient_table["magnitud"] = coefficient_table[
        "coeficiente_estandarizado"
    ].abs()
    coefficient_table.sort_values("magnitud", ascending=False).drop(
        columns="magnitud"
    ).to_csv(output_dir / "coeficientes.csv", index=False)

    print(f"Filas utilizadas: {len(dataset):,}")
    print(f"Filas de entrenamiento: {len(x_train):,}")
    print(f"Filas de validación: {len(x_validation):,}")
    print(f"Umbral de clasificación: {DECISION_THRESHOLD:.2f}")
    print("\nMétricas de validación:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResultados guardados en: {output_dir}")
    return results
