import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================================================
#                 DATA LOADING
# =========================================================
def cargar_embedding(path_csv):
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {path_csv}")
    df = pd.read_csv(path_csv)
    X = df[['x_0', 'x_1', 'x_2']].values
    return X


# =========================================================
#                 MODEL TRAINING
# =========================================================
def ajustar_superficie(X_2d, Y, degree=3, scale=True, reg_type='ridge', alpha=1.0):
    """
    Ajusta una superficie polinomial (x0, x1) -> x2.
    
    Parámetros:
    -----------
    X_2d: array de shape (n_samples, 2)
    Y:    array de shape (n_samples,)
    degree: grado del polinomio
    scale: bool, si se aplica StandardScaler
    reg_type: 'none', 'ridge', 'lasso'
    alpha: float, regularización

    Retorna:
    --------
    model_dict = {
        "scaler": scaler or None,
        "poly":   poly_features,
        "reg":    reg
    }
    """
    # 1) Escalado (opcional)
    if scale:
        scaler = StandardScaler()
        X_2d_scaled = scaler.fit_transform(X_2d)
    else:
        scaler = None
        X_2d_scaled = X_2d
    
    # 2) Polinomial
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_2d_scaled)

    # 3) Escoger tipo de regresión
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


def predict_n_steps_forward(model_dict, init_data, steps=1):
    """
    Dado un embedding actual (init_data), predice 'steps' puntos futuros.
    'init_data' es un array 2D con shape (m, 3); tomamos el último para arrancar.
    
    Retorna:
    --------
    - predicted_3d: el embedding extendido, de forma (m + steps, 3)
    - preds_only: vector 1D con las nuevas x2 predichas (una por cada paso).
    """
    scaler = model_dict['scaler']
    poly = model_dict['poly']
    reg = model_dict['reg']

    embedding_extended = init_data.copy()
    preds_list = []

    for _ in range(steps):
        last_point = embedding_extended[-1]  # shape (3,) -> [x0, x1, x2]
        
        # Tomar las dos primeras coords para predecir la tercera
        to_predict = last_point[:2].reshape(1, -1)

        # Escalar si corresponde
        if scaler is not None:
            to_predict = scaler.transform(to_predict)
        
        # Expandir polinomialmente
        to_predict_poly = poly.transform(to_predict)
        next_val = reg.predict(to_predict_poly)[0]

        # SHIFT -> [ x_{t-1}, x_{t}, prediccion ]
        new_point = np.append(last_point[1:], next_val)  
        embedding_extended = np.vstack([embedding_extended, new_point])
        preds_list.append(next_val)

    return embedding_extended, np.array(preds_list)


# =========================================================
#                 TRAIN & EVALUATE
# =========================================================
def train_and_evaluate(X, train_ratio=0.8, degree=3, scale=True,
                       reg_type='ridge', alpha=1.0, plot=True):
    """
    Realiza un split train/test según el train_ratio.
    Ajusta el modelo polinomial en la parte de training,
    luego predice recursivamente sobre la parte de test y
    calcula el MSE y MAE (forward).

    Parámetros:
    -----------
    X: np.array (n_samples, 3)
    train_ratio: float [0..1], porcentaje de datos para entrenar
    degree: grado del polinomio
    scale: bool
    reg_type: 'none', 'ridge', 'lasso'
    alpha: float
    plot: bool -> si graficamos resultados

    Retorna:
    --------
    metrics: dict con 'mse' y 'mae' en el set de test
    """
    n = len(X)
    train_size = int(train_ratio * n)
    if train_size < 1:
        raise ValueError("train_ratio too low, no training data.")
    if train_size >= n:
        raise ValueError("train_ratio too high, no test data.")

    # === Split train/test
    X_train = X[:train_size]  # shape (train_size, 3)
    X_test = X[train_size:]   # shape (n - train_size, 3)

    # Entrenamos en la porción de train
    X_train_2d = X_train[:, :2]  # features
    Y_train = X_train[:, 2]      # target
    model_dict = ajustar_superficie(
        X_train_2d, Y_train,
        degree=degree, scale=scale, reg_type=reg_type, alpha=alpha
    )

    # === Multi-step forward forecast
    #  Tomamos el último punto de X_train como estado inicial, 
    #  y predecimos len(X_test) pasos.
    #  Sin overlapping or re-initialization, it's a purely “rolled” forecast.

    # Option A: start from the *ENTIRE TRAIN SET* as "history"
    # So if train set is [0..train_size-1], the last row is index train_size-1.
    # We'll do forward predictions from that final known point onward:

    embedding_with_preds, preds_array = predict_n_steps_forward(
        model_dict,
        X_train,
        steps=len(X_test)  # forecast the same length as the test set
    )

    # `preds_array` should correspond 1-to-1 with X_test[:, 2]
    real_test = X_test[:, 2]

    mse_val = mean_squared_error(real_test, preds_array)
    mae_val = mean_absolute_error(real_test, preds_array)

    metrics = {
        "test_mse": mse_val,
        "test_mae": mae_val
    }

    print(f"[INFO] Multi-step Test MSE: {mse_val:.4f}, MAE: {mae_val:.4f}")

    # === Plot (training + test + predictions)
    if plot:
        _plot_train_test_results(X_train, X_test, embedding_with_preds, preds_array, model_dict)

    return metrics


def _plot_train_test_results(X_train, X_test, embedding_with_preds, preds_array, model_dict):
    """
    Simple 3D plot to visualize the train set, the test set, 
    and the predicted extension in the embedding space.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Entire data for reference
    ax.plot(X_train[:, 0], X_train[:, 1], X_train[:, 2],
            label="Train set", color='blue', marker='o')
    ax.plot(X_test[:, 0],  X_test[:, 1],  X_test[:, 2],
            label="Test set",  color='green', marker='o')

    # Predicted points
    # embedding_with_preds has shape (train_size + len(X_test), 3)
    # the final len(X_test) rows are newly predicted
    pred_start = len(X_train)  # index in embedding_with_preds where test preds begin
    predicted_points = embedding_with_preds[pred_start:]

    ax.plot(predicted_points[:, 0],
            predicted_points[:, 1],
            predicted_points[:, 2],
            label="Predictions", color='red', marker='^')

    # Optionally: plot the polynomial surface, but we need to build a mesh
    _plot_surface(ax, model_dict, X_train, X_test)

    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_zlabel("x₂")
    ax.set_title("Train/Test Split - Embedding + Predictions")
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_surface(ax, model_dict, X_train, X_test, padding=0.1):
    """
    Plots the fitted polynomial surface (for x0, x1 -> x2).
    We'll consider the union of train+test to define the bounding box.
    """
    X_all = np.vstack([X_train, X_test])
    x0_min, x0_max = X_all[:, 0].min(), X_all[:, 0].max()
    x1_min, x1_max = X_all[:, 1].min(), X_all[:, 1].max()

    range_x0 = x0_max - x0_min
    range_x1 = x1_max - x1_min

    x0_vals = np.linspace(x0_min - padding*range_x0, x0_max + padding*range_x0, 80)
    x1_vals = np.linspace(x1_min - padding*range_x1, x1_max + padding*range_x1, 80)
    x0_grid, x1_grid = np.meshgrid(x0_vals, x1_vals)

    grid_2d = np.column_stack([x0_grid.ravel(), x1_grid.ravel()])

    # Apply the same transformations: scale, poly, predict
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


# =========================================================
#                   MAIN
# =========================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_csv = os.path.join(base_dir, "..", "..", "BaseDeDatos", "embedding.csv")

    # Cargar embedding
    X = cargar_embedding(path_csv)
    # shape: (n_samples, 3) -> [x0, x1, x2]

    # Hiperparámetros
    train_ratio = 0.8
    degree = 3
    scale = True
    reg_type = 'ridge'   # 'none', 'ridge', 'lasso'
    alpha = 1.0

    # Entrenar y evaluar
    results = train_and_evaluate(X,
                                 train_ratio=train_ratio,
                                 degree=degree,
                                 scale=scale,
                                 reg_type=reg_type,
                                 alpha=alpha,
                                 plot=True)
    print("Final test metrics:", results)