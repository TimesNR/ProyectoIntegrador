# RB
"""
Comparativa de baselines para serie de tiempo:
  - Naive (último valor observado)
  - Media móvil con ventana deslizante
  - Suavizado exponencial simple
Calcula MAE, MSE y sMAPE sobre el conjunto de prueba y grafica los resultados.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    """
    eps = 1e-9
    num = np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred) + eps
    return 100. * np.mean(2. * num / den)

def split_series(series: np.ndarray, train_ratio: float):
    """
    Separa la serie en train y test según proporción.
    """
    n = len(series)
    train_size = int(train_ratio * n)
    return series[:train_size], series[train_size:], train_size

def baseline_naive(y_train: np.ndarray, n_forecast: int) -> np.ndarray:
    """
    Forecast naive: repite el último valor de entrenamiento.
    """
    last = y_train[-1]
    return np.full(n_forecast, last)

def baseline_moving_average(y_train: np.ndarray, n_forecast: int, window: int) -> np.ndarray:
    """
    Forecast con media móvil:
    Para cada paso de prueba, promedia los últimos 'window' valores vistos.
    """
    history = list(y_train)
    preds = []
    for _ in range(n_forecast):
        window_vals = history[-window:]
        preds.append(np.mean(window_vals))
        history.append(np.mean(window_vals))
    return np.array(preds)

def baseline_exp_smoothing(y_train: np.ndarray, n_forecast: int, alpha: float) -> np.ndarray:
    """
    Suavizado exponencial simple:
    Ajusta nivel en muestra y pronostica futuros como último nivel.
    """
    level = y_train[0]
    for t in range(1, len(y_train)):
        level = alpha * y_train[t] + (1 - alpha) * level
    return np.full(n_forecast, level)

def evaluate_forecast(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula MAE, MSE y sMAPE entre test y predicción.
    """
    return {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "smape": smape(y_test, y_pred)
    }

def plot_baselines(y_train, y_test, forecasts: dict, train_size: int):
    """
    Grafica la serie real y los forecasts de cada baseline.
    """
    idx_train = np.arange(train_size)
    idx_test  = np.arange(train_size, train_size + len(y_test))

    plt.figure()
    plt.plot(idx_train, y_train, label="Real - Entrenamiento")
    plt.plot(idx_test,  y_test,  label="Real - Prueba", color="black")

    for name, y_pred in forecasts.items():
        plt.plot(idx_test, y_pred, linestyle="--", label=f"{name}")

    plt.title("Comparativa de Baselines")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Valor de la serie")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # -------- Parámetros --------
    # Generar ruta absoluta al CSV de la serie de tiempo
    ruta_csv = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "BaseDeDatos",
            "DATOSLIMPIOSRAPPI.csv"
        )
    )
    columna      = "Entregas Black"
    train_ratio  = 0.75

    # Parámetros de los baselines
    window_ma    = 6     # ventana para media móvil
    alpha_smooth = 0.3   # coeficiente para suavizado exponencial
    # ---------------------------

    # Cargar y preparar serie
    df     = pd.read_csv(ruta_csv, encoding="latin1")
    serie  = df[columna].dropna().astype(float).values

    # Separar train/test
    y_train, y_test, train_size = split_series(serie, train_ratio)
    n_forecast = len(y_test)

    # Generar forecasts de cada baseline
    preds = {
        "Naive": baseline_naive(y_train, n_forecast),
        f"MovAvg_w={window_ma}": baseline_moving_average(y_train, n_forecast, window_ma),
        f"ExpSmooth_α={alpha_smooth}": baseline_exp_smoothing(y_train, n_forecast, alpha_smooth)
    }

    # Calcular métricas
    rows = []
    for name, y_pred in preds.items():
        mets = evaluate_forecast(y_test, y_pred)
        rows.append({"baseline": name, **mets})
    df_mets = pd.DataFrame(rows)

    # Mostrar métricas en consola
    print("\n=== Métricas de baselines ===")
    print(df_mets.to_string(index=False))

    # Graficar comparativa
    plot_baselines(y_train, y_test, preds, train_size)

if __name__ == "__main__":
    main()
