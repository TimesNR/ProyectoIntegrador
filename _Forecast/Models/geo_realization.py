import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =======================
#    FUNCION sMAPE
# =======================
def smape(y_true, y_pred):
    """
    Calcula sMAPE en porcentaje.
    """
    epsilon = 1e-9
    numerador = np.abs(y_true - y_pred)
    denominador = np.abs(y_true) + np.abs(y_pred) + epsilon
    return 100.0 * np.mean(2.0 * numerador / denominador)

# ============================
#   Bloque Geometrical Realization
# ============================
class GeometricRealizationBlock(nn.Module):
    """
    Versión simplificada de GR-block, con 2 "pasos" Ws*X*Wd + LeakyReLU
    y salida final 1D.
    """
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        # Primer capa Ws1 + Wd1
        self.Ws1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wd1 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.act1= nn.LeakyReLU(negative_slope=0.1)
        # Segunda capa Ws2 + Wd2
        self.Ws2 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wd2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.act2= nn.LeakyReLU(negative_slope=0.1)
        # Capa final => 1 neurona de salida
        self.fc_out = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        """
        x: tensor de shape (batch_size, input_dim)
        """
        # Paso 1
        h1 = self.Ws1(x)           # shape (batch, hidden_dim)
        o1 = self.Wd1(self.act1(h1))  # back to (batch, input_dim)
        # Paso 2
        h2 = self.Ws2(o1)
        o2 = self.Wd2(self.act2(h2))
        # Salida final (1D)
        pred = self.fc_out(o2)     # (batch, 1)
        return pred

# ============================
#   LEER CSV de Embedding
# ============================
def cargar_embedding(path_csv):
    """
    Lee CSV => asumimos (N, k). 
    Primeras k-1 columnas => features, ultima => target
    Retorna: X_full shape (N, k)
    """
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"No se encontró: {path_csv}")
    df = pd.read_csv(path_csv)
    return df.values  # (N, k)

# ============================
#   Entrenar Bloque GR
# ============================
def train_gr_block(X_feats, y_target, epochs=1000, lr=1e-3, hidden_dim=16):
    """
    X_feats: (n_train, k-1) => features
    y_target: (n_train,) => target
    Retorna modelo entrenado (GeometricRealizationBlock).
    """
    X_t = torch.tensor(X_feats, dtype=torch.float32)
    y_t = torch.tensor(y_target, dtype=torch.float32).view(-1,1)

    input_dim = X_feats.shape[1]
    model = GeometricRealizationBlock(input_dim, hidden_dim=hidden_dim)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion= nn.MSELoss()

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_t)  # (n_train, 1)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

        if (ep+1)%200==0:
            print(f"Epoch {ep+1}, Loss={loss.item():.4f}")

    return model

# ============================
#   Predicción SHIFT Recursiva
# ============================
def predict_recursive_multi_step(model, init_data, steps=1):
    """
    model: la red GR entrenada
    init_data: np.array de shape (train_size, k). 
               (k-1) features, ultima col => target
    steps: num de pasos a predecir

    Devuelve: embedding_extendido, preds_array
    - SHIFT "exacto" si k=3 (lógica antigua)
    - SHIFT extendido si k>3
    """
    embedding_extended = init_data.copy()
    preds_list = []

    # Pasamos a eval
    model.eval()

    for _ in range(steps):
        last_row = embedding_extended[-1]
        # CASO k=3 => Lógica antigua
        if embedding_extended.shape[1] == 3:
            feats = last_row[:2].reshape(1, -1)  # (1,2)
            X_t = torch.tensor(feats, dtype=torch.float32)
            with torch.no_grad():
                pred_val = model(X_t).item()
            preds_list.append(pred_val)
            # SHIFT => [x_{t-1}, x_t, pred_val]
            new_row = np.append(last_row[1:], pred_val)
            embedding_extended = np.vstack([embedding_extended, new_row])
        else:
            # CASO k>3 => SHIFT extendido
            feats = last_row[:-1].reshape(1, -1) # (1, k-1)
            X_t = torch.tensor(feats, dtype=torch.float32)
            with torch.no_grad():
                pred_val = model(X_t).item()
            preds_list.append(pred_val)
            # SHIFT => "roll" en las features
            new_row = np.empty_like(last_row)
            new_row[:-1] = np.roll(last_row[:-1], -1)
            new_row[-1] = pred_val
            embedding_extended = np.vstack([embedding_extended, new_row])

    return embedding_extended, np.array(preds_list)


# ============================
#   Predicción SINGLE-STEP
# ============================
def predict_one_step_ahead(model, X_full, idx):
    """
    Toma la fila X_full[idx], usa sus (k-1) features => predice => return valor float
    - Si k=3 => feats=row[:2]
    - Si k>3 => feats=row[:-1]
    """
    if idx<0 or idx>= len(X_full):
        raise ValueError("Indice fuera de rango.")
    row = X_full[idx]
    if X_full.shape[1] == 3:
        feats = row[:2].reshape(1, -1)
    else:
        feats = row[:-1].reshape(1, -1)

    model.eval()
    X_t = torch.tensor(feats, dtype=torch.float32)
    with torch.no_grad():
        pred_val = model(X_t).item()
    return pred_val

# ============================
#   TRAIN & EVALUATE
# ============================
def train_and_evaluate(
    X_full,
    train_ratio=0.8,
    epochs=1000,
    lr=1e-3,
    hidden_dim=16,
    forecast_mode='recursive'
):
    """
    X_full: (N, k) => (k-1) features + 1 target en la ultima col
    1) split train/test
    2) entrenar GR-block
    3) predecir en test (SHIFT recursivo o single-step)
    4) medir MSE, MAE, sMAPE
    """
    n = len(X_full)
    train_size = int(train_ratio*n)
    if train_size<1:
        raise ValueError("train_ratio muy bajo")
    if train_size>=n:
        raise ValueError("train_ratio muy alto")

    X_train_full = X_full[:train_size]
    X_test_full  = X_full[train_size:]

    # separar features y target en train
    if X_full.shape[1] == 3:
        # caso 2-lags
        X_train_feats = X_train_full[:, :2]
        y_train       = X_train_full[:, 2]
    else:
        # caso k>3
        X_train_feats = X_train_full[:, :-1]
        y_train       = X_train_full[:, -1]

    # entrenar
    model = train_gr_block(X_train_feats, y_train, epochs=epochs, lr=lr, hidden_dim=hidden_dim)

    # predecir
    if forecast_mode=='recursive':
        embedding_extended, preds_array = predict_recursive_multi_step(
            model, X_train_full, steps=len(X_test_full)
        )
        if X_full.shape[1] == 3:
            real_test = X_test_full[:, 2]
        else:
            real_test = X_test_full[:, -1]
    elif forecast_mode=='single_step':
        preds_list=[]
        for i in range(len(X_test_full)):
            pval = predict_one_step_ahead(model, X_test_full, i)
            preds_list.append(pval)
        preds_array = np.array(preds_list)
        if X_full.shape[1] == 3:
            real_test = X_test_full[:, 2]
        else:
            real_test = X_test_full[:, -1]
        embedding_extended = np.vstack([X_train_full, X_test_full.copy()])
    else:
        raise ValueError("forecast_mode debe ser 'recursive' o 'single_step'")

    # mse, mae, smape
    mse_val = mean_squared_error(real_test, preds_array)
    mae_val = mean_absolute_error(real_test, preds_array)
    smape_val=smape(real_test, preds_array)

    metrics = {
        "test_mse": mse_val,
        "test_mae": mae_val,
        "test_smape": smape_val
    }
    print(f"[INFO] GR-block => forecast_mode={forecast_mode}, epochs={epochs}, lr={lr}, hidden_dim={hidden_dim}")
    print(f"[INFO] MSE={mse_val:.4f}, MAE={mae_val:.4f}, sMAPE={smape_val:.2f}%")

    return model, metrics

# ============================
#    MAIN DEMO
# ============================
if __name__=="__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_csv = os.path.join(base_dir,"..","..","BaseDeDatos","embedding.csv")

    X_full = cargar_embedding(path_csv)
    print("Shape embedding:", X_full.shape)

    # Ejemplo => entrenar y predecir en SHIFT recursivo
    model, mets = train_and_evaluate(
        X_full,
        train_ratio=0.7,
        epochs=800,
        lr=1e-3,
        hidden_dim=16,
        forecast_mode='recursive'
    )
    print("Final metrics:", mets)