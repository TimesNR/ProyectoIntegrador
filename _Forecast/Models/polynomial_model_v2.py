# RB
"""
Pipeline multiserie de forecasting polinomial:
  - Takens embedding
  - Grid search (grado, alpha)
  - Entrenamiento + evaluación (MAE, MSE, MASE, sMAPE)
  - Gráfica de desempeño por serie
  - Acumulación de métricas en DataFrame para exportar
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.style.use("cyberpunk")

## DIRECTORIOS
base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "..", "..", "BaseDeDatos")
)
# Directorio de embeddings
embedding_dir = os.path.join(base_dir, "embeddings")

# Directorio de plots
plot_dir = os.path.join(os.path.dirname(__file__), 
                            "..",
                            "Results"
)
os.makedirs(plot_dir, exist_ok=True)

def mase_horizon(y_true_h: np.ndarray, y_pred_h: np.ndarray,
                y_train: np.ndarray, h: int):
    # Denominador: average absolute h-step naive on train
    n = len(y_train)
    denom = np.abs(y_train[h:] - y_train[:-h]).mean()
    num   = np.abs(y_true_h - y_pred_h).mean()
    return num / denom

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (en %)."""
    eps = 1e-9
    num = np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred) + eps
    return (.5) * np.mean(2. * num / den)

def mase(y_true, y_pred, y_train, m=1):
    """Mean Absolute Scaled Error (MASE)."""
    n = len(y_train)
    d = np.abs(np.diff(y_train, n=m)).sum() / (n - m)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

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
    n = len(X_full)
    train_size = int(train_ratio * n)
    X_tr, X_te = X_full[:train_size], X_full[train_size:]
    X_tr_feats, y_tr = X_tr[:, :-1], X_tr[:, -1]
    y_te = X_te[:, -1]

    scaler = StandardScaler().fit(X_tr_feats)
    X_tr_s = scaler.transform(X_tr_feats)
    poly = PolynomialFeatures(degree).fit(X_tr_s)
    X_tr_p = poly.transform(X_tr_s)

    model = Ridge(alpha=alpha).fit(X_tr_p, y_tr)
    y_tr_pred = model.predict(X_tr_p)

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

    mets = {
        "mse": mean_squared_error(y_te, y_te_pred),
        "mae": mean_absolute_error(y_te, y_te_pred),
        "mase": mase_horizon(y_te, y_te_pred, y_tr, h=1),
        "smape": smape(y_te, y_te_pred)
    }

    mets['mase_horizon'] = mase_horizon(y_te, y_te_pred, y_tr, h=len(y_te))

    return y_tr, y_tr_pred, y_te, y_te_pred, mets, train_size

def grid_search(X_full: np.ndarray, train_ratio: float, degrees: list, alphas: list):
    results = []
    for d in degrees:
        for a in alphas:
            _, _, _, _, mets, _ = train_evaluate(X_full, train_ratio, d, a)
            results.append({"degree": d, "alpha": a, **mets})
    df = pd.DataFrame(results)
    best = df.loc[df["mase"].idxmin()]
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

    # Guardar plot
    filepath = os.path.join(plot_dir, f"{series_name}.png")
    plt.savefig(filepath)
    plt.close()
    print(f"[INFO] Plot saved to {filepath}")

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
    scaler = StandardScaler().fit(X_full[:, :-1])
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
        p       = max(p, 0.0)
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
    # 1) Leer y normalizar la serie a [0,1]
    raw = df[col].dropna().astype(float).values.reshape(-1,1)
    scaler_serie = StandardScaler().fit(raw)
    serie_norm   = scaler_serie.transform(raw).ravel()

    # 2) Embedding sobre la serie escalada
    emb_path = os.path.join(embedding_dir, f"emb_{series_name}.csv")
    X_full   = takens_embedding(serie_norm, dimension, delay)

    os.makedirs(embedding_dir, exist_ok=True)
    pd.DataFrame(X_full, columns=[f"x_{i}" for i in range(X_full.shape[1])])\
      .to_csv(emb_path, index=False)
    print(f"[INFO] {series_name}: embedding guardado en {emb_path}")

    # 2) Grid search
    df_grid, best_d, best_a = grid_search(X_full, train_ratio, degrees, alphas)
    print(f"[INFO] {series_name}: mejor grado={best_d}, alpha={best_a}")

        # 3) Entrena y evalúa en la escala [0,1]
    y_tr_s, y_tr_pred_s, y_te_s, y_te_pred_s, _, train_size = train_evaluate(
        X_full, train_ratio, best_d, best_a
    )

    # 4) Des‐normalizar resultados a la escala original
    y_tr       = scaler_serie.inverse_transform(y_tr_s.reshape(-1,1)).ravel()
    y_tr_pred  = scaler_serie.inverse_transform(y_tr_pred_s.reshape(-1,1)).ravel()
    y_te       = scaler_serie.inverse_transform(y_te_s.reshape(-1,1)).ravel()
    y_te_pred  = scaler_serie.inverse_transform(y_te_pred_s.reshape(-1,1)).ravel()

    # 5) Recalcular métricas en la escala original
    mets = {
        "mse":  mean_squared_error(y_te,       y_te_pred),
        "mae":  mean_absolute_error(y_te,      y_te_pred),
        "mase": mase_horizon(y_te,       y_te_pred, y_tr,h=1),
        "smape": smape(y_te,      y_te_pred)
    }

    mets['mase_horizon'] = mase_horizon(y_te, y_te_pred, y_tr, h=len(y_te)) # mase con horizonte

    # 6) Calcular confiabilidad al 80%
    mask = y_te != 0
    errors_rel = np.abs(y_te - y_te_pred) / np.abs(y_te)
    reliability_80 = np.mean(errors_rel[mask] <= 0.2)
    mets["reliability_80"] = reliability_80

    # (opcional) imprimir detalle
    count_ok = int(np.sum(errors_rel[mask] <= 0.2))
    total    = int(mask.sum())
    print(f"[INFO] {series_name}: {count_ok}/{total} meses con error ≤20% (confiabilidad 80%)")

    # 7) Gráfica
    plot_performance(series_name, y_tr, y_tr_pred, y_te, y_te_pred, train_size)

    # 8) Retornar métricas y el escalador
    metrics = {"series": series_name, "degree": best_d, "alpha": best_a, **mets}
    return metrics, scaler_serie

def main():
    # ---------------- Configuración general ----------------

    # Dataset Rappi (sin índice de fecha)
    rappi_csv  = os.path.join(base_dir, "DATOSLIMPIOSRAPPI.csv")
    rappi_cols = [
        "Entregas Black",
        "Entregas Ocean Plastic",
        "Entregas Cardjolote Black",
        "Entregas Cincita",
        "Entregas Onix",
        "Entregas Pride 2022",
        "Entregas Mundial",
        "Entregas Pride Hologlam",
        "Entregas Pride Colors",
        "Entregas Cardlaveritas",
        "Entregas Cardtrinas",
        "Entregas Cardjolote White",
        "Entregas PlayCard",
        "Entregas Gotica",
        "Entregas Pride 2024",
        "Entregas Minion Rayo",
        "Entregas Minion Mega",
        "Entregas Cardlaveritas 2024"
    ]
    # Dataset BancaCred (con columna Fecha)
    bancacred_csv  = os.path.join(base_dir, "BD_HIST_BANCACRED_IF.csv")
    bancacred_date = "Fecha"

    #Dataset Alan (sin índice de fecha)
    alan_csv = os.path.join(base_dir, "nuevas_series.csv")
    alan_cols = [
        "mastercard", 
        "not_mastercard_or_visa",
        "tarjetas_debito",
        "visa"
    ]

    # Parámetros comunes
    dimension   = 3
    delay       = 1
    train_ratio = 0.71
    degrees_rappi     = list(range(1, 11))
    degrees_bancacred = list(range(1, 3))
    alphas      = [0.01, 0.1, 1.0, 10.0, 100.0]
    n_meses     = 6

    all_metrics   = []
    all_forecasts = []

# -----------------------------------------------------------------------------

    # — Procesar Rappi (con gráfico estándar) —
    df_r = pd.read_csv(rappi_csv, encoding="latin1")
    for col in rappi_cols:
        key = f"rappi_{col.replace(' ', '_').lower()}"
        metrics, scaler_serie = process_series(
            series_name=key,
            df=df_r,
            col=col,
            embedding_dir=embedding_dir,
            dimension=dimension,
            delay=delay,
            train_ratio=train_ratio,
            degrees=degrees_rappi,
            alphas=alphas
        )
        all_metrics.append(metrics)

        ## Forecast final
        # 1) Preparamos la serie normalizada para forecast
        raw = df_r[col].dropna().astype(float).values.reshape(-1,1)
        serie_norm = scaler_serie.transform(raw).ravel()

        # 2) Forecast en escala [0,1]
        future_norm = forecast_n_steps_full(
            serie=serie_norm,
            dimension=dimension,
            delay=delay,
            degree=metrics["degree"],
            alpha=metrics["alpha"],
            n_steps=n_meses
        )
        # 3) Volvemos a escala original
        future = scaler_serie.inverse_transform(future_norm.reshape(-1,1)).ravel()

        # Construimos el diccionario de pronóstico
        fc = {"series": key}
        for i, v in enumerate(future, 1):
            fc[f"t+{i}"] = v
        all_forecasts.append(fc)
        print(f"\nForecast a {n_meses} meses para '{key}':\n", future)

# -----------------------------------------------------------------------------

    # -- Procesar Datos Alan --
    df_a = pd.read_csv(alan_csv)
    for col in alan_cols:
        key = f"alan_{col.replace(' ', '_').lower()}"
        metrics, scaler_serie = process_series(
            series_name=key,
            df=df_a,
            col=col,
            embedding_dir=embedding_dir,
            dimension=dimension,
            delay=delay,
            train_ratio=train_ratio,
            degrees=degrees_bancacred,
            alphas=alphas
        )
        all_metrics.append(metrics)

        ## Forecast final
        # 1) Preparamos la serie normalizada para forecast
        raw = df_a[col].dropna().astype(float).values.reshape(-1,1)
        serie_norm = scaler_serie.transform(raw).ravel()

        # 2) Forecast en escala [0,1]
        future_norm = forecast_n_steps_full(
            serie=serie_norm,
            dimension=dimension,
            delay=delay,
            degree=metrics["degree"],
            alpha=metrics["alpha"],
            n_steps=n_meses
        )
        # 3) Volvemos a escala original
        future = scaler_serie.inverse_transform(future_norm.reshape(-1,1)).ravel()

        # Construimos el diccionario de pronóstico
        fc = {"series": key}
        for i, v in enumerate(future, 1):
            fc[f"t+{i}"] = v
        all_forecasts.append(fc)
        print(f"\nForecast a {n_meses} meses para '{key}':\n", future)

# -----------------------------------------------------------------------------

    # — Procesar BancaCred —  
    original_plot = globals()['plot_performance']
    globals()['plot_performance'] = lambda *args, **kwargs: None

    df_b = pd.read_csv(bancacred_csv, encoding="latin1",
                       parse_dates=[bancacred_date])
    df_b.set_index(bancacred_date, inplace=True)
    bancacred_cols = df_b.columns.tolist()
    date_idx       = df_b.index

    for col in bancacred_cols:
        key, raw = f"bancacred_{col.replace(' ', '_').lower()}", None
        # 1) Métricas + obtención de scaler
        metrics, scaler_serie = process_series(
            series_name=key,
            df=df_b,
            col=col,
            embedding_dir=embedding_dir,
            dimension=dimension,
            delay=delay,
            train_ratio=train_ratio,
            degrees=degrees_bancacred,
            alphas=alphas
        )
        all_metrics.append(metrics)

        # 2) Forecast escala [0,1] → escala original
        raw = df_b[col].dropna().astype(float).values.reshape(-1,1)
        serie_norm = scaler_serie.transform(raw).ravel()
        future_norm = forecast_n_steps_full(
            serie=serie_norm,
            dimension=dimension,
            delay=delay,
            degree=metrics["degree"],
            alpha=metrics["alpha"],
            n_steps=n_meses
        )
        future = scaler_serie.inverse_transform(future_norm.reshape(-1,1)).ravel()
        fc = {"series": key}
        for i, v in enumerate(future, 1):
            fc[f"t+{i}"] = v
        all_forecasts.append(fc)
        print(f"\nForecast a {n_meses} meses para '{key}':\n", future)

        # 3) Gráfico único con fechas (usar serie_norm + scaler_serie)
        #   - Reconstruimos embedding normalizado
        X_full_n = takens_embedding(serie_norm, dimension, delay)
        #   - Entrenamos/evaluamos sobre valores normalizados
        y_tr_s, y_tr_pred_s, y_te_s, y_te_pred_s, _, train_size = train_evaluate(
            X_full_n, train_ratio, metrics["degree"], metrics["alpha"]
        )
        #   - Des-normalizamos para plot
        y_tr      = scaler_serie.inverse_transform(y_tr_s.reshape(-1,1)).ravel()
        y_tr_pred = scaler_serie.inverse_transform(y_tr_pred_s.reshape(-1,1)).ravel()
        y_te      = scaler_serie.inverse_transform(y_te_s.reshape(-1,1)).ravel()
        y_te_pred = scaler_serie.inverse_transform(y_te_pred_s.reshape(-1,1)).ravel()

        #   - Fechas alineadas al embedding
        full_dates  = date_idx[(dimension - 1) * delay:]
        train_dates = full_dates[:train_size]
        test_dates  = full_dates[train_size:]

        plt.figure()
        plt.plot(train_dates,    y_tr,      label="Real - Entrenamiento")
        plt.plot(train_dates,    y_tr_pred, linestyle="--", label="Predicción - Entrenamiento")
        plt.plot(test_dates,     y_te,      label="Real - Prueba")
        plt.plot(test_dates,     y_te_pred, linestyle="--", label="Predicción - Prueba")
        plt.title(f"Desempeño: {key}")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Restaurar función de plot original
    globals()['plot_performance'] = original_plot

# -----------------------------------------------------------------------------

    # — Exportar métricas y forecasts —
    df_metrics   = pd.DataFrame(all_metrics)
    df_forecasts = pd.DataFrame(all_forecasts)
    df_metrics.to_csv(os.path.join(base_dir, "polynomial_metrics.csv"), index=False)
    df_forecasts.to_csv(os.path.join(base_dir, "polynomial_forecasts.csv"), index=False)

    print(f"\n[INFO] Métricas guardadas en:  {os.path.join(base_dir, 'polynomial_metrics.csv')}")
    print(f"[INFO] Forecasts guardados en: {os.path.join(base_dir, 'polynomial_forecasts.csv')}")

if __name__ == "__main__":
    main()