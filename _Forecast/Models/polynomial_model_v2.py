# RB
"""
Módulo para:
1. Generar embedding de Takens para una serie de tiempo.
2. Realizar regresión polinomial con regularización Ridge.
3. Buscar la mejor combinación de grado y coeficiente de regularización.
4. Almacenar y mostrar métricas de desempeño (MAE, MSE, sMAPE).
5. Generar un gráfico comparativo de desempeño en entrenamiento vs prueba.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el sMAPE (Symmetric Mean Absolute Percentage Error).
    Devuelve un valor en porcentaje.
    """
    eps = 1e-9
    num = np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred) + eps
    return 100. * np.mean(2. * num / den)

def takens_embedding(serie: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """
    Aplica el teorema de Takens para generar un embedding:
      - serie: array unidimensional de valores x_t
      - dimension: número de componentes d
      - delay: retardo τ
    Retorna un array de forma (N - (d-1)*τ, d), donde cada fila es
    [x_{t-(d-1)τ}, ..., x_{t-τ}, x_t].
    """
    n = len(serie)
    if n < (dimension - 1) * delay + 1:
        raise ValueError("Serie demasiado corta para el embedding solicitado.")
    emb = []
    for i in range((dimension - 1) * delay, n):
        vect = [serie[i - j * delay] for j in range(dimension)]
        emb.append(vect[::-1])
    return np.array(emb, dtype=float)

def create_embedding_file(
    path_serie_csv: str,
    col_objetivo: str,
    dimension: int,
    delay: int,
    output_path: str
) -> None:
    """
    Lee la serie desde un CSV, genera el embedding de Takens y lo guarda
    como CSV en output_path.
    """
    # Carga de serie
    df = pd.read_csv(path_serie_csv, encoding='latin1')
    serie = df[col_objetivo].dropna().astype(float).values
    # Generar embedding
    X = takens_embedding(serie, dimension, delay)
    # Formar DataFrame y guardar
    cols = [f"x_{i}" for i in range(X.shape[1])]
    df_emb = pd.DataFrame(X, columns=cols)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_emb.to_csv(output_path, index=False)
    print(f"[INFO] Embedding guardado en {output_path} (shape={X.shape})")

def train_evaluate(
    X_full: np.ndarray,
    train_ratio: float,
    degree: int,
    alpha: float
):
    """
    Separa X_full en entrenamiento/prueba, entrena regresión polinomial con
    regularización Ridge, predice de forma recursiva y calcula métricas.
    Retorna:
      y_train, y_train_pred, y_test, y_test_pred, dict métricas
    """
    n = len(X_full)
    train_size = int(train_ratio * n)
    if train_size < 1 or train_size >= n:
        raise ValueError("train_ratio inválido para los datos disponibles.")
    # Split
    X_train = X_full[:train_size]
    X_test  = X_full[train_size:]
    # Features y targets
    X_train_feats = X_train[:, :-1]
    y_train       = X_train[:,  -1]
    y_test        = X_test[:,   -1]
    # Escalado y polinomio
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_feats)
    poly   = PolynomialFeatures(degree)
    X_train_p = poly.fit_transform(X_train_s)
    # Entrenamiento
    model = Ridge(alpha=alpha)
    model.fit(X_train_p, y_train)
    # Predicción entrenamiento
    y_train_pred = model.predict(X_train_p)
    # Predicción recursiva prueba
    emb_ext = X_train.copy()
    preds = []
    for _ in range(len(X_test)):
        last = emb_ext[-1]
        feats = last[:-1].reshape(1, -1)
        feats_s = scaler.transform(feats)
        feats_p = poly.transform(feats_s)
        p = model.predict(feats_p)[0]
        preds.append(p)
        # desplazamiento y añadir predicción
        new_row = np.empty_like(last)
        new_row[:-1] = np.roll(last[:-1], -1)
        new_row[-1]  = p
        emb_ext = np.vstack([emb_ext, new_row])
    y_test_pred = np.array(preds)
    # Métricas
    mse   = mean_squared_error(y_test, y_test_pred)
    mae   = mean_absolute_error(y_test, y_test_pred)
    smp   = smape(y_test, y_test_pred)
    mets  = {"mse": mse, "mae": mae, "smape": smp}
    return y_train, y_train_pred, y_test, y_test_pred, mets

def grid_search(
    X_full: np.ndarray,
    train_ratio: float,
    degrees: list,
    alphas: list
) -> (pd.DataFrame, int, float):
    """
    Itera sobre combinaciones de (degree, alpha), ejecuta train_evaluate,
    recopila métricas en DataFrame y retorna el mejor grado y alpha
    según menor sMAPE.
    """
    resultados = []
    for d in degrees:
        for a in alphas:
            _, _, _, _, mets = train_evaluate(X_full, train_ratio, d, a)
            resultados.append({
                "degree": d,
                "alpha": a,
                "mse":   mets["mse"],
                "mae":   mets["mae"],
                "smape": mets["smape"]
            })
            print(f"[TEST] grado={d}, alpha={a} -> sMAPE={mets['smape']:.2f}%")
    df_res = pd.DataFrame(resultados)
    mejor = df_res.loc[df_res["smape"].idxmin()]
    return df_res, int(mejor["degree"]), float(mejor["alpha"])

def plot_performance(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    train_size: int
) -> None:
    """
    Genera un gráfico comparando valores reales vs predichos en entrenamiento
    y en prueba en una sola figura.
    """
    # Índices para eje x
    idx_train = np.arange(train_size)
    idx_test  = np.arange(train_size, train_size + len(y_test))
    plt.figure()
    plt.plot(idx_train, y_train,      label="Real - Entrenamiento")
    plt.plot(idx_train, y_train_pred, linestyle="--", label="Predicción - Entrenamiento")
    plt.plot(idx_test,  y_test,       label="Real - Prueba")
    plt.plot(idx_test,  y_test_pred,  linestyle="--", label="Predicción - Prueba")
    plt.title("Desempeño: Entrenamiento vs Prueba")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Valor de la serie")
    plt.legend()
    plt.tight_layout()
    plt.show()

def forecast_n_steps(
    model,
    scaler,
    poly,
    X_full: np.ndarray,
    n_steps: int
) -> np.ndarray:
    """
    Genera un forecast de n_steps usando el último estado del embedding completo.
    
    Parámetros:
    - model: regresor Ridge ya entrenado.
    - scaler: StandardScaler ajustado con los datos de entrenamiento.
    - poly: PolynomialFeatures ajustado con los datos de entrenamiento.
    - X_full: array del embedding completo (todas las filas reales).
    - n_steps: número de pasos a predecir hacia adelante.
    
    Retorna:
    - preds: array unidimensional con las n_steps predicciones sucesivas.
    """
    # Partimos del último vector del embedding real
    emb = X_full.copy()
    preds = []
    for _ in range(n_steps):
        last = emb[-1]                        # último estado [x_{t-d+1},…,x_t]
        feats = last[:-1].reshape(1, -1)      # features: todos menos el target
        feats_s = scaler.transform(feats)     # estandarizar
        feats_p = poly.transform(feats_s)     # polinomiales
        p = model.predict(feats_p)[0]         # nueva predicción
        preds.append(p)
        # Construir nuevo estado desplazando y añadiendo p
        new_row = np.roll(last, -1)
        new_row[-1] = p
        emb = np.vstack([emb, new_row])
    return np.array(preds)


def main():
    # ------------------- Configuración -------------------
    # Ruta al CSV original con la serie de tiempo
    path_serie = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "..", "..",  "BaseDeDatos", "DATOSLIMPIOSRAPPI.csv")
    )
    # Nombre de la columna objetivo en dicho CSV
    col_obj = "Entregas Black"
    # Parámetros de embedding
    dimension = 3   # d
    delay     = 1   # τ
    # Ruta donde se guardará el embedding
    path_emb  = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "..", "..", "BaseDeDatos", "embedding.csv")
    )
    # Parámetros de modelo y búsqueda
    train_ratio = 0.75
    degrees     = [1, 2, 3, 4]
    alphas      = [0.01, 0.1, 1.0, 10.0, 100.0]
    # ------------------------------------------------------

    # 1) Generar y guardar embedding
    create_embedding_file(path_serie, col_obj, dimension, delay, path_emb)

    # 2) Cargar embedding
    df_emb = pd.read_csv(path_emb)
    X_full = df_emb.values

    # 3) Búsqueda de hiperparámetros
    df_results, best_d, best_a = grid_search(
        X_full, train_ratio, degrees, alphas
    )
    print("\n=== Resultados completos ===")
    print(df_results.sort_values("smape").to_string(index=False))
    print(f"\n>> Mejor configuración: grado={best_d}, alpha={best_a:.4f}")

    # 4) Evaluar con la mejor configuración para graficar
    y_tr, y_tr_pred, y_te, y_te_pred, _ = train_evaluate(
        X_full, train_ratio, best_d, best_a
    )
    train_size = int(train_ratio * len(X_full))

    # 5) Mostrar gráfico de desempeño
    plot_performance(y_tr, y_tr_pred, y_te, y_te_pred, train_size)
    
    # # ---- FORECAST ----
    # # 1) Después de hacer grid_search() y obtener best_d, best_a:
    # train_size = int(train_ratio * len(X_full))
    # X_tr_feats = X_full[:train_size, :-1]
    # y_tr        = X_full[:train_size,  -1]

    # # 2) Re-entrenas el modelo final sobre el train set con los mejores hiperparámetros:
    # scaler = StandardScaler().fit(X_tr_feats)
    # X_tr_s = scaler.transform(X_tr_feats)
    # poly   = PolynomialFeatures(best_d).fit(X_tr_s)
    # X_tr_p = poly.transform(X_tr_s)

    # model = Ridge(alpha=best_a).fit(X_tr_p, y_tr)

    # # 3) Ahora sí llamas a forecast_n_steps pasándole ese model:
    # n_meses = 12
    # future_preds = forecast_n_steps(
    #     model=model,
    #     scaler=scaler,
    #     poly=poly,
    #     X_full=X_full,
    #     n_steps=n_meses
    # )

    # print(f"\nPredicciones a {n_meses} meses:", future_preds)

    # # (Opcional) Graficar forecast
    # idx_future = np.arange(len(X_full), len(X_full) + n_meses)
    # plt.figure()
    # plt.plot(idx_future, future_preds, marker='o', linestyle='-',
    #          label=f'Forecast {n_meses} meses')
    # plt.title('Pronóstico futuro')
    # plt.xlabel('Índice de muestra (continuación)')
    # plt.ylabel('Valor pronosticado')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
