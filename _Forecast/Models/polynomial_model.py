import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =======================
#    FUNCION sMAPE
# =======================
def smape(y_true, y_pred):
    """
    Calcula el sMAPE (Symmetric Mean Absolute Percentage Error).
    Devuelve un valor en porcentaje.
    """
    epsilon = 1e-9  # para evitar división por cero
    numerador = np.abs(y_true - y_pred)
    denominador = np.abs(y_true) + np.abs(y_pred) + epsilon
    return 100.0 * np.mean(2.0 * numerador / denominador)

# ==============================
#       CARGA DE DATOS
# ==============================
def cargar_embedding(path_csv):
    # Lee un CSV, verifica existencia y retorna matriz X con columnas x_0, x_1, x_2
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {path_csv}")
    df = pd.read_csv(path_csv)
    X = df[['x_0', 'x_1', 'x_2']].values
    return X

# ==============================
#  AJUSTE DE SUPERFICIE
# ==============================
def ajustar_superficie(X_2d, Y, degree=3, scale=True, reg_type='ridge', alpha=1.0):
    """
    Ajusta una superficie polinomial (x0, x1) -> x2.
    Retorna un diccionario con:
     - scaler: escalador (o None)
     - poly: transformador polinomial
     - reg: regresor entrenado
    """
    if scale:
        scaler = StandardScaler()
        X_2d_scaled = scaler.fit_transform(X_2d)
    else:
        scaler = None
        X_2d_scaled = X_2d
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_2d_scaled)

    if reg_type == 'ridge':
        reg = Ridge(alpha=alpha)
    elif reg_type == 'lasso':
        reg = Lasso(alpha=alpha)
    else:
        reg = LinearRegression()

    reg.fit(X_poly, Y)
    return {
        "scaler": scaler,
        "poly": poly,
        "reg": reg
    }

# ====================================
#  PREDICCION SINGLE-STEP (no rolling)
# ====================================
def predict_one_step_ahead(model_dict, X_full, start_idx):
    """
    Predice x2 en un índice específico sin usar predicciones previas,
    sino el valor real de X_full en ese instante (single-step).
    """
    if start_idx < 2:
        raise ValueError("No hay suficientes puntos para 2-lags.")
    
    # Suponiendo X_full[t,:] = [x_{t-2}, x_{t-1}, x_t]
    X_2d_to_predict = X_full[start_idx, 0:2].reshape(1, -1)

    scaler = model_dict['scaler']
    poly = model_dict['poly']
    reg = model_dict['reg']

    if scaler is not None:
        X_2d_to_predict = scaler.transform(X_2d_to_predict)
    
    X_poly = poly.transform(X_2d_to_predict)
    pred = reg.predict(X_poly)[0]
    return pred

# =================================
#  PREDICCION MULTI-STEP (rolling)
# =================================
def predict_recursive_multi_step(model_dict, init_data, steps=1):
    """
    Predicción recursiva usando SHIFT en el embedding.
    Agrega nuevos puntos [x_{t-1}, x_t, x_{t+1}] sucesivamente.
    """
    scaler = model_dict['scaler']
    poly = model_dict['poly']
    reg = model_dict['reg']

    embedding_extended = init_data.copy()
    preds_list = []

    for _ in range(steps):
        last_point = embedding_extended[-1]
        to_predict = last_point[:2].reshape(1, -1)

        if scaler is not None:
            to_predict = scaler.transform(to_predict)
        
        to_predict_poly = poly.transform(to_predict)
        next_val = reg.predict(to_predict_poly)[0]

        # SHIFT: [x_{t-1}, x_t, x_{t+1}]
        new_point = np.append(last_point[1:], next_val)
        embedding_extended = np.vstack([embedding_extended, new_point])
        preds_list.append(next_val)

    return embedding_extended, np.array(preds_list)

# ==================================
#   ENTRENAR Y EVALUAR EL MODELO
# ==================================
def train_and_evaluate(
    X, 
    train_ratio=0.8, 
    degree=3, 
    scale=True,
    reg_type='ridge', 
    alpha=1.0, 
    plot=True,
    forecast_mode='recursive'):
    """
    Divide X en train/test. Ajusta el modelo polinomial con la parte train.
    Luego predice en la parte test según el modo:
      - 'recursive': rolling multi-step
      - 'single_step': cada paso se alimenta con datos reales (no acumula error).
    Calcula MSE, MAE y sMAPE.
    """
    n = len(X)
    train_size = int(train_ratio * n)
    if train_size < 1:
        raise ValueError("train_ratio muy bajo, sin datos de entrenamiento.")
    if train_size >= n:
        raise ValueError("train_ratio muy alto, sin datos de test.")

    # Separar en train/test
    X_train = X[:train_size]
    X_test = X[train_size:]

    # Entrenar modelo
    X_train_2d = X_train[:, :2]
    Y_train = X_train[:, 2]
    model_dict = ajustar_superficie(
        X_train_2d, Y_train,
        degree=degree, scale=scale, reg_type=reg_type, alpha=alpha
    )

    # Real test
    real_test = X_test[:, 2]

    # Predicciones
    if forecast_mode == 'recursive':
        embedding_with_preds, preds_array = predict_recursive_multi_step(
            model_dict,
            init_data=X_train,
            steps=len(X_test)
        )
    elif forecast_mode == 'single_step':
        preds_list = []
        embedding_with_preds = X_train.copy()

        for i in range(len(X_test)):
            test_row = X_test[i]
            X_2d_to_predict = test_row[:2].reshape(1, -1)

            scaler = model_dict['scaler']
            poly   = model_dict['poly']
            reg    = model_dict['reg']

            if scaler is not None:
                X_2d_to_predict = scaler.transform(X_2d_to_predict)
            X_poly = poly.transform(X_2d_to_predict)
            pred_val = reg.predict(X_poly)[0]
            preds_list.append(pred_val)

        preds_array = np.array(preds_list)
        test_pred_rows = X_test.copy()
        test_pred_rows[:, 2] = preds_array
        embedding_with_preds = np.vstack([embedding_with_preds, test_pred_rows])
    else:
        raise ValueError("forecast_mode debe ser 'recursive' o 'single_step'")

    # Métricas
    mse_val = mean_squared_error(real_test, preds_array)
    mae_val = mean_absolute_error(real_test, preds_array)
    smape_val = smape(real_test, preds_array)

    metrics = {
        "test_mse": mse_val,
        "test_mae": mae_val,
        "test_smape": smape_val
    }

    # Print
    print(f"[INFO] Modo de predicción: {forecast_mode}")
    print(f"[INFO] Grado={degree}, alpha={alpha}, MSE={mse_val:.4f}, MAE={mae_val:.4f}, sMAPE={smape_val:.2f}%")

    # Graficar
    if plot:
        _plot_train_test_results(X_train, X_test, embedding_with_preds, preds_array, model_dict, forecast_mode)

    return metrics

# ===============================
#   BÚSQUEDA DE GRADO y ALPHA
# ===============================
def search_best_degree_alpha(
    X,
    train_ratio=0.8,
    degrees=[1,2,3],
    alphas=[0.01, 0.1, 1.0],
    scale=True,
    reg_type='ridge',
    forecast_mode='recursive',
    plot=False
):
    """
    Prueba distintas combinaciones (grado, alpha) entrenando y evaluando 
    con train_and_evaluate en el mismo train/test. 
    Se queda con la que tenga menor sMAPE en test.

    Retorna:
      - best_config: (mejor_grado, mejor_alpha)
      - best_metrics: dict con métricas (MSE, MAE, sMAPE)
      - all_results: lista con ((deg, alpha), metrics)
    """
    n = len(X)
    train_size = int(train_ratio * n)
    if train_size < 1 or train_size >= n:
        raise ValueError("train_ratio inválido para el tamaño de X.")

    best_smape = float('inf')
    best_config = None
    best_metrics = None
    all_results = []

    # Desactivamos el plot en la búsqueda, para no graficar mil veces
    # Pero daremos la opción final de graficar la config ganadora
    for deg in degrees:
        for alpha_val in alphas:
            # Llamamos a train_and_evaluate con plot=False
            mets = train_and_evaluate(
                X,
                train_ratio=train_ratio,
                degree=deg,
                scale=scale,
                reg_type=reg_type,
                alpha=alpha_val,
                plot=False,
                forecast_mode=forecast_mode
            )
            all_results.append(((deg, alpha_val), mets))
            smp = mets['test_smape']
            if smp < best_smape:
                best_smape = smp
                best_config = (deg, alpha_val)
                best_metrics = mets

    print("\n=== Resultados de la búsqueda (ordenados por sMAPE) ===")
    sorted_list = sorted(all_results, key=lambda x: x[1]['test_smape'])
    for combo, mets in sorted_list:
        d, a = combo
        print(f"Grado={d}, alpha={a}, MSE={mets['test_mse']:.2f}, "
              f"MAE={mets['test_mae']:.2f}, sMAPE={mets['test_smape']:.2f}%")

    print(f"\n>> Mejor configuración: Grado={best_config[0]}, alpha={best_config[1]}, sMAPE={best_smape:.2f}%")

    # (Opcional) generar un plot final con la config ganadora
    if plot:
        print("\n--- Graficando la mejor config ---")
        train_and_evaluate(
            X,
            train_ratio=train_ratio,
            degree=best_config[0],
            scale=scale,
            reg_type=reg_type,
            alpha=best_config[1],
            plot=True,
            forecast_mode=forecast_mode
        )

    return best_config, best_metrics, all_results


def _plot_train_test_results(X_train, X_test, embedding_with_preds, preds_array, model_dict, forecast_mode):
    # Grafica 3D con train, test y predicciones
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(X_train[:, 0], X_train[:, 1], X_train[:, 2],
            label="Train set", color='blue', marker='o')
    ax.plot(X_test[:, 0],  X_test[:, 1],  X_test[:, 2],
            label="Test set",  color='green', marker='o')

    if forecast_mode == 'recursive':
        pred_start = len(X_train)
        predicted_points = embedding_with_preds[pred_start:]
        ax.plot(predicted_points[:, 0],
                predicted_points[:, 1],
                predicted_points[:, 2],
                label="Predicción (recursive)", color='red', marker='^')
    else:
        pred_start = len(X_train)
        predicted_points = embedding_with_preds[pred_start:]
        ax.plot(predicted_points[:, 0],
                predicted_points[:, 1],
                predicted_points[:, 2],
                label="Predicción (single-step)", color='red', marker='^')

    _plot_surface(ax, model_dict, X_train, X_test)
    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_zlabel("x₂")
    ax.set_title(f"Train/Test + Predicciones ({forecast_mode})")
    ax.legend()
    plt.tight_layout()
    plt.show()

def _plot_surface(ax, model_dict, X_train, X_test, padding=0.1):
    # Genera una malla en [x0_min, x0_max] x [x1_min, x1_max] y evalúa la superficie
    X_all = np.vstack([X_train, X_test])
    x0_min, x0_max = X_all[:, 0].min(), X_all[:, 0].max()
    x1_min, x1_max = X_all[:, 1].min(), X_all[:, 1].max()

    range_x0 = x0_max - x0_min
    range_x1 = x1_max - x1_min

    x0_vals = np.linspace(x0_min - padding*range_x0, x0_max + padding*range_x0, 80)
    x1_vals = np.linspace(x1_min - padding*range_x1, x1_max + padding*range_x1, 80)
    x0_grid, x1_grid = np.meshgrid(x0_vals, x1_vals)

    grid_2d = np.column_stack([x0_grid.ravel(), x1_grid.ravel()])

    scaler = model_dict['scaler']
    poly = model_dict['poly']
    reg = model_dict['reg']

    if scaler is not None:
        grid_2d_scaled = scaler.transform(grid_2d)
    else:
        grid_2d_scaled = grid_2d

    grid_poly = poly.transform(grid_2d_scaled)
    z_pred = reg.predict(grid_poly).reshape(x0_grid.shape)

    ax.plot_surface(x0_grid, x1_grid, z_pred, alpha=0.35, cmap='viridis')

# ============================
#            MAIN
# ============================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_csv = os.path.join(base_dir, "..", "..", "BaseDeDatos", "embedding.csv")

    X = cargar_embedding(path_csv)

    # Ejemplo de uso: buscar la mejor combinación (grado, alpha)
    # Y luego graficar la mejor.
    best_config, best_metrics, all_results = search_best_degree_alpha(
        X,
        train_ratio=0.7,
        degrees=[1,2,3,4],
        alphas=[0.18, 0.712, 0.1, 0.12, 0.15],
        scale=True,
        reg_type='ridge',
        forecast_mode='single_step', # elejir modo de prediccion: 'single_step' ó 'recursive'
        plot=True   # esto grafica solo la config ganadora al final
    )