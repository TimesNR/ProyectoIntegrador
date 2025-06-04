import os
import numpy as np
import pandas as pd

from routes.registros import obtener_columnas_patron
from db import engine
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mase_horizon(y_true_h, y_pred_h, y_train, h):
    n = len(y_train)
    diff = np.abs(y_train[h:] - y_train[:-h])
    denom = diff.mean() if len(diff) > 0 else np.nan
    if np.isnan(denom) or denom == 0:
        return np.nan
    return np.abs(y_true_h - y_pred_h).mean() / denom


def smape(y_true, y_pred):
    eps = 1e-9
    num = np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred) + eps
    return 100. * np.mean(2. * num / den)


def mase(y_true, y_pred, y_train, m=1):
    n = len(y_train)
    d = np.abs(np.diff(y_train, n=m)).sum() / (n - m)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d


def takens_embedding(serie, dimension, delay):
    n = len(serie)
    if n < (dimension - 1) * delay + 1:
        raise ValueError("Serie demasiado corta para embedding solicitado.")
    emb = []
    for i in range((dimension - 1) * delay, n):
        window = [serie[i - j * delay] for j in range(dimension)]
        emb.append(window[::-1])
    return np.array(emb, dtype=float)


def train_evaluate(X_full, train_ratio, degree, alpha):
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

    mets["mase_horizon"] = mase_horizon(y_te, y_te_pred, y_tr, h=len(y_te))
    return y_tr, y_tr_pred, y_te, y_te_pred, mets, train_size


def grid_search(X_full, train_ratio, degrees, alphas):
    results = []
    for d in degrees:
        for a in alphas:
            _, _, _, _, mets, _ = train_evaluate(X_full, train_ratio, d, a)
            results.append({"degree": d, "alpha": a, **mets})
    df = pd.DataFrame(results)
    if df["mase"].dropna().empty:
        return None, None, None
    best = df.loc[df["mase"].idxmin()]
    return df, int(best["degree"]), float(best["alpha"])


def forecast_n_steps_full(serie, dimension, delay, degree, alpha, n_steps):
    X_full = takens_embedding(serie, dimension, delay)
    X_feats = X_full[:, :-1]
    y = X_full[:, -1]

    scaler = StandardScaler().fit(X_feats)
    X_s = scaler.transform(X_feats)
    poly = PolynomialFeatures(degree).fit(X_s)
    X_p = poly.transform(X_s)

    model = Ridge(alpha=alpha).fit(X_p, y)

    emb_ext = X_full.copy()
    preds = []
    for _ in range(n_steps):
        last = emb_ext[-1]
        f_s = scaler.transform(last[:-1].reshape(1, -1))
        f_p = poly.transform(f_s)
        p = model.predict(f_p)[0]
        p = max(p, 0.0)
        preds.append(p)
        new_row = np.roll(last, -1)
        new_row[-1] = p
        emb_ext = np.vstack([emb_ext, new_row])
    return np.array(preds)


def process_series(series_name, df, col, embedding_dir,
                   dimension, delay, train_ratio, degrees, alphas):
    raw = df[col].dropna().astype(float).values.reshape(-1, 1)
    if len(raw) < (dimension - 1) * delay + 2:
        print(f"[SKIP] Serie '{series_name}' demasiado corta para Takens embedding")
        return None, None

    scaler_serie = StandardScaler().fit(raw)
    serie_norm = scaler_serie.transform(raw).ravel()

    try:
        X_full = takens_embedding(serie_norm, dimension, delay)
    except ValueError as e:
        print(f"[ERROR] {series_name}: {e}")
        return None, None

    os.makedirs(embedding_dir, exist_ok=True)

    df_grid, best_d, best_a = grid_search(X_full, train_ratio, degrees, alphas)
    if best_d is None or best_a is None:
        print(f"[SKIP] Serie '{series_name}' no tiene hiperparámetros válidos.")
        return None, None

    y_tr_s, y_tr_pred_s, y_te_s, y_te_pred_s, _, train_size = train_evaluate(
        X_full, train_ratio, best_d, best_a
    )

    y_tr = scaler_serie.inverse_transform(y_tr_s.reshape(-1, 1)).ravel()
    y_tr_pred = scaler_serie.inverse_transform(y_tr_pred_s.reshape(-1, 1)).ravel()
    y_te = scaler_serie.inverse_transform(y_te_s.reshape(-1, 1)).ravel()
    y_te_pred = scaler_serie.inverse_transform(y_te_pred_s.reshape(-1, 1)).ravel()

    mets = {
        "mse": mean_squared_error(y_te, y_te_pred),
        "mae": mean_absolute_error(y_te, y_te_pred),
        "mase": mase_horizon(y_te, y_te_pred, y_tr, h=1),
        "smape": smape(y_te, y_te_pred)
    }

    mets["mase_horizon"] = mase_horizon(y_te, y_te_pred, y_tr, h=len(y_te))

    eps = 1e-9
    errors_rel = np.abs(y_te - y_te_pred) / (np.abs(y_te) + eps)
    reliability_80 = np.mean(errors_rel <= 0.2)
    mets["reliability_80"] = reliability_80

    metrics = {"series": series_name, "degree": best_d, "alpha": best_a, **mets}
    return metrics, scaler_serie


def main():
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "BaseDeDatos")
    )
    query = "SELECT * FROM tarjetas_data;"
    rappi_csv = pd.read_sql(query, engine)
    rappi_cols = obtener_columnas_patron(patron='entregas')

    embedding_dir = os.path.join(base_dir, "embeddings")
    dimension = 3
    delay = 1
    train_ratio = 0.71
    degrees_rappi = list(range(1, 11))
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    n_meses = 6

    all_metrics = []
    all_forecasts = []

    df_r = rappi_csv
    for col in rappi_cols:
        key = f"rappi_{col.replace(' ', '_').lower()}"
        result = process_series(
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
        if result is None:
            continue
        metrics, scaler_serie = result
        if metrics is None or scaler_serie is None:
            continue
        all_metrics.append(metrics)

        raw = df_r[col].dropna().astype(float).values.reshape(-1, 1)
        serie_norm = scaler_serie.transform(raw).ravel()
        future_norm = forecast_n_steps_full(
            serie=serie_norm,
            dimension=dimension,
            delay=delay,
            degree=metrics["degree"],
            alpha=metrics["alpha"],
            n_steps=n_meses
        )
        future = scaler_serie.inverse_transform(future_norm.reshape(-1, 1)).ravel()
        fc = {"series": key}
        for i, v in enumerate(future, 1):
            fc[f"t{i}"] = v  # ← CAMBIO AQUÍ
        all_forecasts.append(fc)

    df_metrics = pd.DataFrame(all_metrics)
    df_forecasts = pd.DataFrame(all_forecasts)
    df_metrics.to_csv("polynomial_metrics.csv", index=False)
    df_forecasts.to_csv("polynomial_forecasts.csv", index=False)


if __name__ == "__main__":
    main()
