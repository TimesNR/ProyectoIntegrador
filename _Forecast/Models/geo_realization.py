import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D  
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
    
    Retorna un diccionario con el scaler (opcional),
    la expansión polinomial y el regresor entrenado.
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


# =========================================================
#         FORECAST MODES
# =========================================================
def predict_one_step_ahead(model_dict, X_full, start_idx):
    """
    For a single test index 'start_idx', we take the real embedding 
    [x_{t-2}, x_{t-1}] from X_full at (start_idx-1, start_idx) 
    and predict x_{t}.  The function returns the predicted x2 value.

    Used repeatedly for each test point. 
    This does NOT do rolling recursion. 
    (We always rely on the actual previous points from X_full.)
    """
    if start_idx < 2:
        raise ValueError("Need at least 2 preceding points to do 2-lag embedding.")
    
    # We'll use the real data for the last two time steps:
    x0 = X_full[start_idx - 2, 2]  # x_{t-2}
    x1 = X_full[start_idx - 1, 2]  # x_{t-1}

    # But note: your data is stored as columns x_0, x_1, x_2. 
    # So if you're storing the entire time series in X_full, 
    # ensure you interpret them consistently. 
    # Alternatively, if your X_full is truly [x0, x1, x2], 
    # then x0, x1 are columns 0,1, not from the time-lag dimension. 
    # For typical "time-lag" embedding, 
    # you'd do something like X_full[start_idx, :2].

    # -------------
    # However, in your code, X_full[t, 0] = x_{t-2}, 
    #                   X_full[t, 1] = x_{t-1}, 
    #                   X_full[t, 2] = x_t
    # So "start_idx" row already has the embedding [ x_{t-2}, x_{t-1}, x_t ]
    # We only need X_full[start_idx, 0:2].
    # -------------
    # For clarity, let's assume your entire dataset is an array of shape (n, 3),
    # each row: [ x_{t-2}, x_{t-1}, x_t].
    # Then "one step ahead" for row t is basically:
    
    X_2d_to_predict = X_full[start_idx, 0:2].reshape(1, -1)

    # Scale if needed
    scaler = model_dict['scaler']
    poly = model_dict['poly']
    reg = model_dict['reg']

    if scaler is not None:
        X_2d_to_predict = scaler.transform(X_2d_to_predict)
    
    X_poly = poly.transform(X_2d_to_predict)
    pred = reg.predict(X_poly)[0]
    return pred


def predict_recursive_multi_step(model_dict, init_data, steps=1):
    """
    EXACTLY as your existing function 'predict_n_steps_forward',
    uses SHIFT in the embedding space to recursively predict steps into the future.
    """
    scaler = model_dict['scaler']
    poly = model_dict['poly']
    reg = model_dict['reg']

    embedding_extended = init_data.copy()
    preds_list = []

    for _ in range(steps):
        last_point = embedding_extended[-1]  # shape (3,) -> [x0, x1, x2]
        to_predict = last_point[:2].reshape(1, -1)

        if scaler is not None:
            to_predict = scaler.transform(to_predict)
        
        to_predict_poly = poly.transform(to_predict)
        next_val = reg.predict(to_predict_poly)[0]

        # SHIFT -> [ x_{t-1}, x_{t}, prediccion ]
        new_point = np.append(last_point[1:], next_val)
        embedding_extended = np.vstack([embedding_extended, new_point])
        preds_list.append(next_val)

    return embedding_extended, np.array(preds_list)


# =========================================================
#         TRAIN & EVALUATE
# =========================================================
def train_and_evaluate(X, train_ratio=0.8, degree=3, scale=True,
                       reg_type='ridge', alpha=1.0, plot=True,
                       forecast_mode='recursive'):
    """
    Splits data into train & test sets. 
    Train the polynomial model on the train set. 
    Then forecast on the test set in one of two modes:
      - 'recursive' (multi-step rolling)
      - 'single_step' (predict each test point from real data)
    """
    n = len(X)
    train_size = int(train_ratio * n)
    if train_size < 1:
        raise ValueError("train_ratio too low, no training data.")
    if train_size >= n:
        raise ValueError("train_ratio too high, no test data.")

    # Split train/test
    X_train = X[:train_size]  # each row: [x_{t-2}, x_{t-1}, x_t]
    X_test = X[train_size:]   # shape (n - train_size, 3)

    # Train model
    X_train_2d = X_train[:, :2]
    Y_train = X_train[:, 2]
    model_dict = ajustar_superficie(
        X_train_2d, Y_train,
        degree=degree, scale=scale, reg_type=reg_type, alpha=alpha
    )

    real_test = X_test[:, 2]
    preds_array = None
    embedding_with_preds = None

    if forecast_mode == 'recursive':
        # -- Rolling multi-step approach --
        embedding_with_preds, preds_array = predict_recursive_multi_step(
            model_dict,
            init_data=X_train,
            steps=len(X_test)
        )

    elif forecast_mode == 'single_step':
        # -- Single-step approach: for each row in X_test, 
        #    we directly predict from X_test's first 2 coords.
        preds_list = []
        embedding_with_preds = X_train.copy()  # not strictly needed, just for plotting continuity

        for i in range(len(X_test)):
            # The row i in X_test corresponds to global index (train_size + i) in the full data
            # So we can do:
            test_row = X_test[i]  # shape (3,) -> [x_{t-2}, x_{t-1}, x_t]
            # We want to predict x_t from [x_{t-2}, x_{t-1}].
            # This is basically test_row[:2], but we can be explicit:
            X_2d_to_predict = test_row[:2].reshape(1, -1)

            scaler = model_dict['scaler']
            poly = model_dict['poly']
            reg = model_dict['reg']

            if scaler is not None:
                X_2d_to_predict = scaler.transform(X_2d_to_predict)
            X_poly = poly.transform(X_2d_to_predict)
            pred_val = reg.predict(X_poly)[0]
            preds_list.append(pred_val)

        preds_array = np.array(preds_list)

        # For plotting predictions in 3D, let's extend embedding_with_preds 
        # by appending the "predicted" rows. 
        # But in single-step mode, each test row is still 
        # [x_{t-2}, x_{t-1}, x_t], we only replaced x_t with pred. 
        # So let's do that:
        test_pred_rows = X_test.copy()
        test_pred_rows[:, 2] = preds_array  # place predicted x2 in the last col
        embedding_with_preds = np.vstack([embedding_with_preds, test_pred_rows])
    else:
        raise ValueError("forecast_mode must be either 'recursive' or 'single_step'")

    # Evaluate
    mse_val = mean_squared_error(real_test, preds_array)
    mae_val = mean_absolute_error(real_test, preds_array)

    metrics = {
        "test_mse": mse_val,
        "test_mae": mae_val
    }
    print(f"[INFO] Forecast Mode: {forecast_mode}")
    print(f"[INFO] Multi-step Test MSE: {mse_val:.4f}, MAE: {mae_val:.4f}")

    # Optional: Plot
    if plot:
        _plot_train_test_results(X_train, X_test, embedding_with_preds, preds_array, model_dict, forecast_mode)

    return metrics


def _plot_train_test_results(X_train, X_test, embedding_with_preds, preds_array, model_dict, forecast_mode):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(X_train[:, 0], X_train[:, 1], X_train[:, 2],
            label="Train set", color='blue', marker='o')
    ax.plot(X_test[:, 0],  X_test[:, 1],  X_test[:, 2],
            label="Test set",  color='green', marker='o')

    # If forecast_mode == 'recursive', the last 'len(X_test)' rows 
    # in embedding_with_preds are the predicted points 
    if forecast_mode == 'recursive':
        pred_start = len(X_train)  
        predicted_points = embedding_with_preds[pred_start:]
        ax.plot(predicted_points[:, 0],
                predicted_points[:, 1],
                predicted_points[:, 2],
                label="Predictions (recursive)", color='red', marker='^')
    else:
        # single_step: the entire embedding_with_preds = [train_rows + test_rows_with_pred_x2]
        pred_start = len(X_train)
        predicted_points = embedding_with_preds[pred_start:]
        ax.plot(predicted_points[:, 0],
                predicted_points[:, 1],
                predicted_points[:, 2],
                label="Predictions (single-step)", color='red', marker='^')

    _plot_surface(ax, model_dict, X_train, X_test)
    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_zlabel("x₂")
    ax.set_title(f"Train/Test - Embedding + Predictions ({forecast_mode})")
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_surface(ax, model_dict, X_train, X_test, padding=0.1):
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


# =========================================================
#                   MAIN
# =========================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_csv = os.path.join(base_dir, "..", "..", "BaseDeDatos", "embedding.csv")

    X = cargar_embedding(path_csv)
    # shape: (n_samples, 3) -> [ x_{t-2}, x_{t-1}, x_t ]

    train_ratio = 0.8
    degree = 3
    scale = True
    reg_type = 'ridge'
    alpha = 1.0

    print("\n--- Single-step approach ---")
    results_single = train_and_evaluate(
        X,
        train_ratio=train_ratio,
        degree=degree,
        scale=scale,
        reg_type=reg_type,
        alpha=alpha,
        plot=True,
        forecast_mode='single_step'
    )
    print("Test metrics (single-step):", results_single)

    print("\n--- Recursive approach ---")
    results_recursive = train_and_evaluate(
        X,
        train_ratio=train_ratio,
        degree=degree,
        scale=scale,
        reg_type=reg_type,
        alpha=alpha,
        plot=True,
        forecast_mode='recursive'
    )
    print("Test metrics (recursive):", results_recursive)