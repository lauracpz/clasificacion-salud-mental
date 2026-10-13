"""Interfaz de línea de comandos del proyecto."""

import argparse
from pathlib import Path

import pandas as pd

from .models import TARGET, prepare_dataset, run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    """Define las rutas configurables desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Compara modelos de regresión logística sobre datos de salud mental."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "train.csv",
        help="Ruta al archivo train.csv descargado desde Kaggle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Carpeta donde se guardan métricas, coeficientes y gráficos.",
    )
    return parser.parse_args()


def main() -> None:
    """Carga los datos, informa de la limpieza y ejecuta la comparación."""
    arguments = parse_arguments()
    if not arguments.data.is_file():
        raise FileNotFoundError(
            f"No se encontró {arguments.data}. Descarga train.csv y colócalo en data/raw/."
        )

    raw_data = pd.read_csv(arguments.data)
    dataset = prepare_dataset(raw_data)
    removed_rows = len(raw_data) - len(dataset)
    print(f"Filas originales: {len(raw_data):,}")
    print(f"Filas descartadas por datos ausentes o no válidos: {removed_rows:,}")
    print("Distribución de Depression (0 = no, 1 = sí):")
    print((dataset[TARGET].value_counts(normalize=True).sort_index() * 100).round(2))
    run_experiment(dataset, arguments.output_dir)
