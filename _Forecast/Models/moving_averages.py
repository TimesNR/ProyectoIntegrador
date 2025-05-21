import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

def mase(y_train: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error (MASE).
    """
    n = y_train.shape[0]
    d = np.mean(np.abs(np.diff(y_train)))
    errors = np.abs(y_test - y_pred)
    return np.mean(errors) / (d + 1e-9)


def split_series(series: np.ndarray, train_ratio: float):
    n = len(series)
    train_size = int(train_ratio * n)
    return series[:train_size], series[train_size:], train_size

def baseline_moving_average(y_train: np.ndarray, n_forecast: int, window: int) -> np.ndarray:
    history = list(y_train)
    preds = []
    for _ in range(n_forecast):
        window_vals = history[-window:]
        preds.append(np.mean(window_vals))
        history.append(np.mean(window_vals))
    return np.array(preds)

def evaluate_forecast(y_train: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mase": mase(y_train, y_test, y_pred)
    }

def plot_forecast(y_train, y_test, forecast, train_size, label):
    idx_train = np.arange(train_size)
    idx_test  = np.arange(train_size, train_size + len(y_test))

    plt.figure()
    plt.plot(idx_train, y_train, label="Entrenamiento")
    plt.plot(idx_test,  y_test,  label="Prueba", color="black")
    plt.plot(idx_test, forecast, linestyle="--", label=f"MovAvg")

    plt.title(f"Forecast usando media móvil: {label}")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.show()

def residual_diagnostics(y_test, y_pred):
    residuals = y_test - y_pred
    print(f"Mean of residuals: {np.mean(residuals)}")
    print(f"Variance of residuals: {np.var(residuals)}")
    plt.figure()
    plt.plot(residuals, label="Residuals")
    plt.title("Residual Analysis")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Residual")
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Ruta y archivo
    ruta_csv = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "BaseDeDatos",
            "DATOSLIMPIOSRAPPI.csv"
        )
    )

    columnas = ["Entregas Black", "Entregas Ocean Plastic", "Entregas Cardjolote Black", "Demanda Total Tarjetas"]  
    train_ratio  = 0.75
    window_ma    = 6

    df = pd.read_csv(ruta_csv, encoding="latin1")

    for col in columnas:
        print(f"\n=== Columna: {col} ===")
        serie = df[col].dropna().astype(float).values
        y_train, y_test, train_size = split_series(serie, train_ratio)
        n_forecast = len(y_test)

        y_pred = baseline_moving_average(y_train, n_forecast, window_ma)
        metrics = evaluate_forecast(y_train, y_test, y_pred)
        print(f"Métricas: {metrics}")

        plot_forecast(y_train, y_test, y_pred, train_size, col)
        residual_diagnostics(y_test, y_pred)

if __name__ == "__main__":
    main()