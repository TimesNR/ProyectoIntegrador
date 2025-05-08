# RB
"""
Pipeline multiserie de forecasting polinomial:
  - Takens embedding
  - Grid search (grado, alpha)
  - Entrenamiento + evaluación (MAE, MSE, sMAPE)
  - Gráfica de desempeño por serie
  - Acumulación de métricas en DataFrame para exportar
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (en %)."""
    eps = 1e-9
    num = np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred) + eps
    return 100. * np.mean(2. * num / den)

def takens_embedding(serie: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """
    Genera embedding de Takens:
      - serie: array 1D [x₀, x₁, …]
      - dimension d, delay τ
    Devuelve array shape=(N–(d–1)τ, d).
    """
    n = len(serie)
    if n < (dimension - 1) * delay + 1:
        raise ValueError("Serie demasiado corta para embedding solicitado.")
    emb = []
    for i in range((dimension - 1) * delay, n):
        window = [serie[i - j * delay] for j in range(dimension)]
        emb.append(window[::-1])
    return np.array(emb, dtype=float)

def train_evaluate(X_full: np.ndarray, train_ratio: float, degree: int, alpha: float):
    """
    Entrena Ridge sobre features polinomiales y evalúa en train + rolling test.
    Retorna (y_tr, y_tr_pred, y_te, y_te_pred, metrics_dict, train_size).
    """
    n = len(X_full)
    train_size = int(train_ratio * n)
    X_tr, X_te = X_full[:train_size], X_full[train_size:]
    X_tr_feats, y_tr = X_tr[:, :-1], X_tr[:, -1]
    y_te = X_te[:, -1]

    # Preprocesado
    scaler = StandardScaler().fit(X_tr_feats)
    X_tr_s = scaler.transform(X_tr_feats)
    poly   = PolynomialFeatures(degree).fit(X_tr_s)
    X_tr_p = poly.transform(X_tr_s)

    # Entrenamiento
    model = Ridge(alpha=alpha).fit(X_tr_p, y_tr)
    y_tr_pred = model.predict(X_tr_p)

    # Rolling prediction en test
    emb_ext = X_tr.copy()
    preds = []
    for _ in range(len(X_te)):
        last = emb_ext[-1]
        feats_s = scaler.transform(last[:-1].reshape(1, -1))
        feats_p = poly.transform(feats_s)
        p = model.predict(feats_p)[0]
        preds.append(p)
        new_row = np.roll(last, -1)
        new_row[-1] = p
        emb_ext = np.vstack([emb_ext, new_row])
    y_te_pred = np.array(preds)

    # Métricas
    mets = {
        "mse":   mean_squared_error(y_te, y_te_pred),
        "mae":   mean_absolute_error(y_te, y_te_pred),
        "smape": smape(y_te, y_te_pred)
    }
    return y_tr, y_tr_pred, y_te, y_te_pred, mets, train_size

def grid_search(X_full: np.ndarray, train_ratio: float, degrees: list, alphas: list):
    """
    Prueba combinaciones (degree, alpha), devuelve DataFrame de resultados
    y el mejor (degree, alpha) según sMAPE mínimo.
    """
    results = []
    for d in degrees:
        for a in alphas:
            _, _, _, _, mets, _ = train_evaluate(X_full, train_ratio, d, a)
            results.append({"degree": d, "alpha": a, **mets})
    df = pd.DataFrame(results)
    best = df.loc[df["smape"].idxmin()]
    return df, int(best["degree"]), float(best["alpha"])

def plot_performance(series_name, y_tr, y_tr_pred, y_te, y_te_pred, train_size):
    """Grafica real vs. predicción en train y test para una serie."""
    idx_tr = np.arange(train_size)
    idx_te = np.arange(train_size, train_size + len(y_te))

    plt.figure()
    plt.plot(idx_tr, y_tr,      label="Real - Entrenamiento")
    plt.plot(idx_tr, y_tr_pred, linestyle="--", label="Predicción - Entrenamiento")
    plt.plot(idx_te,  y_te,      label="Real - Prueba")
    plt.plot(idx_te,  y_te_pred, linestyle="--", label="Predicción - Prueba")
    plt.title(f"Desempeño: {series_name}")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.show()

def forecast_n_steps_full(
    serie: np.ndarray,
    dimension: int,
    delay: int,
    degree: int,
    alpha: float,
    n_steps: int
) -> np.ndarray:
    """
    Entrena un modelo Ridge polinomial (con Takens embedding) usando
    TODA la serie como entrenamiento y genera un forecast de n_steps.

    Parámetros:
    - serie: array 1D con la serie de tiempo original.
    - dimension, delay: parámetros de embedding de Takens.
    - degree, alpha: hiperparámetros óptimos para PolynomialFeatures y Ridge.
    - n_steps: número de pasos a pronosticar hacia adelante.

    Retorna:
    - preds: array con las n_steps predicciones sucesivas.
    """
    # 1) Embedding completo
    X_full = takens_embedding(serie, dimension, delay)

    # 2) Prepara features y target
    X_feats = X_full[:, :-1]
    y        = X_full[:,  -1]

    # 3) Escalado + polinomio
    scaler = StandardScaler().fit(X_feats)
    X_s    = scaler.transform(X_feats)
    poly   = PolynomialFeatures(degree).fit(X_s)
    X_p    = poly.transform(X_s)

    # 4) Entrenamiento final sobre TODO el embedding
    model = Ridge(alpha=alpha).fit(X_p, y)

    # 5) Forecast recursivo de n_steps
    emb_ext = X_full.copy()
    preds   = []
    for _ in range(n_steps):
        last    = emb_ext[-1]
        f_s     = scaler.transform(last[:-1].reshape(1, -1))
        f_p     = poly.transform(f_s)
        p       = model.predict(f_p)[0]
        preds.append(p)
        # desplaza el window y añade la nueva predicción
        new_row = np.roll(last, -1)
        new_row[-1] = p
        emb_ext = np.vstack([emb_ext, new_row])

    return np.array(preds)

def process_series(
    series_name: str,
    df: pd.DataFrame,
    col: str,
    embedding_dir: str,
    dimension: int,
    delay: int,
    train_ratio: float,
    degrees: list,
    alphas: list
) -> dict:
    """
    Ejecuta todo el pipeline para una columna:
      - Embedding → CSV + np.array
      - Grid search → mejor grado/alpha
      - Entrenamiento + evaluación final
      - Plot
    Devuelve diccionario con métricas.
    """
    # 1) Embedding
    emb_path = os.path.join(embedding_dir, f"emb_{series_name}.csv")
    serie = df[col].dropna().astype(float).values
    X_full = takens_embedding(serie, dimension, delay)
    os.makedirs(embedding_dir, exist_ok=True)
    pd.DataFrame(X_full, columns=[f"x_{i}" for i in range(X_full.shape[1])])\
      .to_csv(emb_path, index=False)
    print(f"[INFO] {series_name}: embedding guardado en {emb_path}")

    # 2) Grid search
    df_grid, best_d, best_a = grid_search(X_full, train_ratio, degrees, alphas)
    print(f"[INFO] {series_name}: mejor grado={best_d}, alpha={best_a}")

    # 3) Entrena y evalúa final
    y_tr, y_tr_pred, y_te, y_te_pred, mets, train_size = train_evaluate(
        X_full, train_ratio, best_d, best_a
    )

    # 4) Gráfica
    plot_performance(series_name, y_tr, y_tr_pred, y_te, y_te_pred, train_size)

    # 5) Retorna métricas
    return {
        "series": series_name,
        "degree": best_d,
        "alpha":  best_a,
        **mets
    }

def main():
    # ---------------- Configuración general ----------------
    path_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "..", "..", "BaseDeDatos", "DATOSLIMPIOSRAPPI.csv")
    )
    cols = [
        "Entregas Black",
        "Entregas Ocean Plastic",
        "Entregas Cardjolote Black",
        "Demanda Total Tarjetas",
        "# de Usuarios"
        # … otras columnas …
    ]
    embedding_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "..", "..", "BaseDeDatos", "embeddings")
    )
    dimension   = 3
    delay       = 1
    train_ratio = 0.7
    degrees     = [1, 2, 3, 4]
    alphas      = [0.01, 0.1, 1.0, 10.0, 100.0]
    n_meses     = 6  # número de pasos a forecastear
    # -------------------------------------------------------

    df = pd.read_csv(path_csv, encoding="latin1")
    rows_metrics = []
    rows_forecasts = []

    for col in cols:
        # 1) Procesa la serie (embedding, búsqueda, train/test, plot)
        series_name = col.replace(" ", "_").lower()
        metrics = process_series(
            series_name=series_name,
            df=df,
            col=col,
            embedding_dir=embedding_dir,
            dimension=dimension,
            delay=delay,
            train_ratio=train_ratio,
            degrees=degrees,
            alphas=alphas
        )
        rows_metrics.append(metrics)

        # 2) Genera el forecast a n_meses usando todos los datos
        serie = df[col].dropna().astype(float).values
        future = forecast_n_steps_full(
            serie=serie,
            dimension=dimension,
            delay=delay,
            degree=metrics['degree'],
            alpha=metrics['alpha'],
            n_steps=n_meses
        )

        # Construye diccionario de forecast
        forecast_dict = {'series': series_name}
        for i, val in enumerate(future, start=1):
            forecast_dict[f't+{i}'] = val
        rows_forecasts.append(forecast_dict)

        # 3) Muestra el forecast en consola
        print(f"\nForecast a {n_meses} meses para '{series_name}':")
        print(future)

    # DataFrame final de métricas y forecasts
    df_metrics   = pd.DataFrame(rows_metrics)
    df_forecasts = pd.DataFrame(rows_forecasts)

    # Rutas de salida
    out_metrics_csv    = os.path.join(embedding_dir, "..", 'polynomial_metrics.csv')
    out_metrics_xlsx   = os.path.join(embedding_dir, "..", 'polynomial_metrics.xlsx')
    out_forecast_csv   = os.path.join(embedding_dir, "..", 'polynomial_forecasts.csv')
    out_forecast_xlsx  = os.path.join(embedding_dir, "..", 'polynomial_forecasts.xlsx')

    # Guardar archivos
    df_metrics.to_csv(out_metrics_csv,   index=False)
    df_metrics.to_excel(out_metrics_xlsx, index=False)
    df_forecasts.to_csv(out_forecast_csv, index=False)
    df_forecasts.to_excel(out_forecast_xlsx, index=False)

    print(f"\n[INFO] Métricas guardadas en:\n  • {out_metrics_csv}\n  • {out_metrics_xlsx}")
    print(f"[INFO] Forecasts guardados en:\n  • {out_forecast_csv}\n  • {out_forecast_xlsx}")

if __name__ == "__main__":
    main()