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
    epsilon = 1e-9
    numerador = np.abs(y_true - y_pred)
    denominador = np.abs(y_true) + np.abs(y_pred) + epsilon
    return 100.0 * np.mean(2.0 * numerador / denominador)

# ===================================================================
#  CARGAR CSV => Ultima columna = target, resto = features
# ===================================================================
def cargar_embedding(path_csv):
    """
    Lee un CSV con shape (N, k).
    - Asume que las primeras k-1 columnas son features
    - La última columna es el target x_t
    Retorna:
      (X_feats, y_target, X_full)
      donde X_full = (N, k), X_feats = (N, k-1), y_target=(N,)
    """
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {path_csv}")
    df = pd.read_csv(path_csv)
    X_all = df.values  # (N, k)

    if X_all.shape[1] < 2:
        raise ValueError("Se requieren al menos 2 columnas (features + 1 target).")

    X_feats = X_all[:, :-1]
    y_target= X_all[:, -1]
    return X_feats, y_target, X_all

# ===================================================================
#  AJUSTE SUPERFICIE POLINOMIAL (X_feats->Y)
# ===================================================================
def ajustar_superficie(X_feats, Y, degree=3, scale=True, reg_type='ridge', alpha=1.0):
    """
    Ajusta polinomio: (k-1) features => 1 target
    """
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_feats)
    else:
        scaler = None
        X_scaled = X_feats

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_scaled)

    if reg_type=='ridge':
        reg = Ridge(alpha=alpha)
    elif reg_type=='lasso':
        reg = Lasso(alpha=alpha)
    else:
        reg = LinearRegression()

    reg.fit(X_poly, Y)
    return {
        "scaler": scaler,
        "poly": poly,
        "reg": reg
    }

# ===================================================================
#  SHIFT MULTI-STEP (RECURSIVE)
# ===================================================================
def predict_recursive_multi_step(model_dict, init_data, steps=1):
    """
    init_data: (n_train, k).  k>1 => k-1 features, 1 target (última col).
    
    - Si k==3, replicamos EXACTAMENTE la lógica antigua:
        last_point[:2] => predict => new_point=append(last_point[1:], pred)
    - Si k>3, hacemos un SHIFT generalizado: 
        * Tomamos last_point[:-1] (features),
        * predecimos,
        * 'desplazamos' dichas features 1 paso y metemos la pred al final.
    """
    scaler = model_dict['scaler']
    poly   = model_dict['poly']
    reg    = model_dict['reg']

    embedding_extended = init_data.copy()
    preds_list = []

    for _ in range(steps):
        last_row = embedding_extended[-1]  # (k,)

        # ============= CASO K=3 => EXACTAMENTE igual que el script viejo =============
        if embedding_extended.shape[1] == 3:
            # old code logic
            # features = last_point[:2]
            feats = last_row[:2].reshape(1, -1)
            if scaler is not None:
                feats = scaler.transform(feats)
            feats_poly = poly.transform(feats)
            pred_val = reg.predict(feats_poly)[0]
            preds_list.append(pred_val)

            # SHIFT => new_point = np.append(last_point[1:], pred_val)
            # last_point es [x_{t-2}, x_{t-1}, x_t],
            # new_point => [x_{t-1}, x_t, pred_val]
            new_point = np.append(last_row[1:], pred_val)
            embedding_extended = np.vstack([embedding_extended, new_point])

        else:
            # ============= CASO K>3 => SHIFT generalizado =============
            feats = last_row[:-1].reshape(1, -1)  # (1, k-1)
            if scaler is not None:
                feats = scaler.transform(feats)
            feats_poly = poly.transform(feats)
            pred_val = reg.predict(feats_poly)[0]
            preds_list.append(pred_val)

            # SHIFT "extendido":
            # Desplazamos las (k-1) features a la izquierda y metemos pred_val en la última col
            new_row = np.empty_like(last_row)
            # Ejemplo: new_row[:-1] = roll(last_row[:-1], -1)
            new_row[:-1] = np.roll(last_row[:-1], -1)
            new_row[-1] = pred_val

            embedding_extended = np.vstack([embedding_extended, new_row])

    return embedding_extended, np.array(preds_list)

# ===================================================================
#  SINGLE-STEP
# ===================================================================
def predict_one_step_ahead(model_dict, X_full, idx):
    """
    idx: fila de X_full para predecir su target sin SHIFT (en single-step).
    - Si k=3 => 
        feats = row[:2], se predice => return pred
      Igualmente a la lógica vieja
    - Si k>3 => 
        feats = row[:-1], 
        se aplica transform y se predice
    """
    if idx<0 or idx>= len(X_full):
        raise ValueError("idx fuera de rango.")
    row = X_full[idx]  # (k,)

    if X_full.shape[1] == 3:
        # old code approach
        feats = row[:2].reshape(1, -1)
    else:
        feats = row[:-1].reshape(1, -1)

    scaler = model_dict['scaler']
    poly   = model_dict['poly']
    reg    = model_dict['reg']

    if scaler is not None:
        feats = scaler.transform(feats)
    feats_poly = poly.transform(feats)
    pred_val   = reg.predict(feats_poly)[0]
    return pred_val

# ===================================================================
# ENTRENAR Y EVALUAR
# ===================================================================
def train_and_evaluate(
    X_full,     # (N, k)
    train_ratio=0.8,
    degree=3,
    scale=True,
    reg_type='ridge',
    alpha=1.0,
    plot=True,
    forecast_mode='recursive',
    invertir_transf=False,  # Nuevo parámetro para invertir la transformación
    offset=1e-9             # Offset usado en la transformación log
):
    """
    Separa X_full en [train, test], entrena polinomio, 
    predice en test (shift multi-step o single-step), mide MSE/MAE/sMAPE.
    
    - Si k=3 => se replica EXACTAMENTE la lógica antigua con SHIFT:
        * features = row[:2], target=row[2]
    - Si k>3 => se extiende la lógica a k-1 features, 1 target
    """
    n = len(X_full)
    train_size = int(train_ratio*n)
    if train_size < 1:
        raise ValueError("train_ratio muy bajo.")
    if train_size >= n:
        raise ValueError("train_ratio muy alto, sin test.")

    X_train_full = X_full[:train_size]  # (train_size, k)
    X_test_full  = X_full[train_size:]   # (test_size,  k)

    # =========== Extraer features/target de train para entrenar ==============
    if X_full.shape[1] == 3:
        # Lógica vieja: 
        # X_train_feats = X_train_full[:, :2]
        # y_train = X_train_full[:, 2]
        X_train_feats = X_train_full[:, :2]
        y_train       = X_train_full[:, 2]
    else:
        # k>3 => (k-1) features
        X_train_feats = X_train_full[:, :-1]
        y_train       = X_train_full[:, -1]

    # Ajustar polinomio
    model_dict = ajustar_superficie(
        X_train_feats, y_train,
        degree=degree, scale=scale,
        reg_type=reg_type, alpha=alpha
    )

    # =========== PREDICCION en test =============
    if forecast_mode == 'recursive':
        embedding_with_preds, preds_array = predict_recursive_multi_step(
            model_dict, X_train_full, steps=len(X_test_full)
        )
        # Real test
        if X_full.shape[1] == 3:
            real_test = X_test_full[:, 2]
        else:
            real_test = X_test_full[:, -1]
    elif forecast_mode == 'single_step':
        preds_list = []
        for i in range(len(X_test_full)):
            pred_val = predict_one_step_ahead(model_dict, X_test_full, i)
            preds_list.append(pred_val)
        preds_array = np.array(preds_list)
        if X_full.shape[1] == 3:
            real_test = X_test_full[:, 2]
        else:
            real_test = X_test_full[:, -1]
        # Para consistencia
        embedding_with_preds = np.vstack([X_train_full, X_test_full.copy()])
    else:
        raise ValueError("forecast_mode debe ser 'recursive' o 'single_step'")

    # Si los datos fueron transformados (por ejemplo, se aplicó log en embedding.csv),
    # se aplica la transformación inversa (exponencial) antes de calcular las métricas.
    if invertir_transf:
        preds_array = np.exp(preds_array) - offset
        real_test   = np.exp(real_test) - offset

    # =========== Calcular MSE/MAE/sMAPE ============
    mse_val   = mean_squared_error(real_test, preds_array)
    mae_val   = mean_absolute_error(real_test, preds_array)
    smape_val = smape(real_test, preds_array)
    metrics = {
        "test_mse": mse_val,
        "test_mae": mae_val,
        "test_smape": smape_val
    }
    print(f"[INFO] Mode={forecast_mode}, k={X_full.shape[1]}, deg={degree}, alpha={alpha}")
    print(f"[INFO] MSE={mse_val:.4f}, MAE={mae_val:.4f}, sMAPE={smape_val:.2f}%")

    # Solo graficar si k=3 => 2 features + 1 target
    if plot and X_full.shape[1] == 3:
        _plot_train_test_3D(
            X_train_full, X_test_full,
            embedding_with_preds, preds_array,
            model_dict, forecast_mode
        )
    return metrics


def _plot_train_test_3D(
    X_train_full, X_test_full,
    embedding_with_preds, preds_array,
    model_dict, forecast_mode
):
    # Replicar la grafica 3D original
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # train
    ax.plot(X_train_full[:, 0], X_train_full[:,1], X_train_full[:,2],
            'o', label="Train", color='blue')
    ax.plot(X_train_full[:, 0], X_train_full[:,1], X_train_full[:,2],
            color='blue')
    # test
    ax.plot(X_test_full[:, 0], X_test_full[:,1], X_test_full[:,2],
            'o', label="Test", color='green')
    ax.plot(X_test_full[:, 0], X_test_full[:,1], X_test_full[:,2],
            color='green')

    if forecast_mode=='recursive':
        pred_start = len(X_train_full)
        predicted_points = embedding_with_preds[pred_start:]
        # Los predichos => color rojo
        ax.plot(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2],
                '^', label="Pred (rec)", color='red')
        ax.plot(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2],
                color='red')

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("x2")
    ax.set_title("Train/Test + Predictions")
    ax.legend()
    plt.tight_layout()
    plt.show()

# ===================================================================
# BÚSQUEDA DE GRADO Y ALPHA
# ===================================================================
def search_best_degree_alpha(
    X_full,
    train_ratio=0.8,
    degrees=[1,2,3],
    alphas=[0.1, 1.0],
    scale=True,
    reg_type='ridge',
    forecast_mode='recursive',
    plot=False,
    invertir=False
):
    """
    Itera sobre (degree, alpha), llama a train_and_evaluate.
    Se queda con la combinación de sMAPE más baja.
    """
    n = len(X_full)
    train_size = int(train_ratio*n)
    if train_size<1 or train_size>=n:
        raise ValueError("train_ratio inválido para el tamaño de X.")

    best_smape   = float('inf')
    best_config  = None
    best_metrics = None
    all_results  = []

    for d in degrees:
        for a in alphas:
            mets = train_and_evaluate(
                X_full,
                train_ratio=train_ratio,
                degree=d,
                scale=scale,
                reg_type=reg_type,
                alpha=a,
                plot=False,
                forecast_mode=forecast_mode,
                invertir_transf=invertir
            )
            smp = mets['test_smape']
            all_results.append(((d,a), mets))
            if smp< best_smape:
                best_smape   = smp
                best_config  = (d,a)
                best_metrics = mets

    # Ordenar
    sorted_res = sorted(all_results, key=lambda x: x[1]['test_smape'])
    print("\n=== Resultados (ordenados por sMAPE) ===")
    for (d,a), mm in sorted_res:
        print(f"Grado={d}, alpha={a}, MSE={mm['test_mse']:.2f}, "
              f"MAE={mm['test_mae']:.2f}, sMAPE={mm['test_smape']:.2f}%")

    print(f"\n>> Mejor config => Grado={best_config[0]}, alpha={best_config[1]}, sMAPE={best_smape:.2f}%")

    if plot:
        print("\n--- Graficando la mejor config ---")
        train_and_evaluate(
            X_full,
            train_ratio=train_ratio,
            degree=best_config[0],
            scale=scale,
            reg_type=reg_type,
            alpha=best_config[1],
            plot=True,
            forecast_mode=forecast_mode,
            invertir_transf=invertir
        )

    return best_config, best_metrics, all_results

# ===================================================================
#                 MAIN
# ===================================================================
if __name__=="__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_csv = os.path.join(base_dir,"..","..","BaseDeDatos","embedding.csv")

    # 1) Leer el embedding
    from pprint import pprint
    X_feats, y_target, X_full = cargar_embedding(path_csv)
    print(f"Shape embedding => {X_full.shape} (k={X_full.shape[1]})")

    # 2) Ejemplo: busqueda
    best_cfg, best_mets, all_res = search_best_degree_alpha(
        X_full,
        train_ratio=0.7,
        degrees=np.arange(1,5,1),
        alphas=np.arange(0,600,1),
        scale=True,
        reg_type='lasso', # (ridge) // (lasso) // (linear_regresion)
        forecast_mode='recursive', # recursive // single_step
        plot=True,
        invertir=False
    )