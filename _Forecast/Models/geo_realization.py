import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
from mpl_toolkits.mplot3d import Axes3D

# === CONFIGURACIÓN ===
def cargar_embedding(path_csv):
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {path_csv}")
    df = pd.read_csv(path_csv)
    X = df[['x_0', 'x_1']].values
    Y = df['x_2'].values
    return X, Y

# === VISUALIZACIÓN ===
def visualizar_embedding(X, Y, title="Embedding 3D"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], Y, marker='o', color='orangered', label="Embedding")
    ax.plot(X[:, 0], X[:, 1], Y, color='black', linewidth=0.8)
    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_zlabel("x₂")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# === AJUSTE POLINOMIAL ===
def ajustar_superficie(X, Y, degree=3, padding_ratio=0.15):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, Y)

    # Padding proporcional
    x0_min, x0_max = X[:,0].min(), X[:,0].max()
    x1_min, x1_max = X[:,1].min(), X[:,1].max()

    x0_range = x0_max - x0_min
    x1_range = x1_max - x1_min

    x0_vals = np.linspace(x0_min - padding_ratio * x0_range, x0_max + padding_ratio * x0_range, 100)
    x1_vals = np.linspace(x1_min - padding_ratio * x1_range, x1_max + padding_ratio * x1_range, 100)
    
    x0_grid, x1_grid = np.meshgrid(x0_vals, x1_vals)
    X_grid = np.column_stack((x0_grid.ravel(), x1_grid.ravel()))
    z_pred = model.predict(poly.transform(X_grid)).reshape(x0_grid.shape)

    return model, poly, x0_grid, x1_grid, z_pred

# === VISUALIZACIÓN DEL AJUSTE ===
def visualizar_superficie(X, Y, x0_grid, x1_grid, z_pred):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], Y, 'o', color='mediumblue', label="Embedding original")
    ax.plot(X[:, 0], X[:, 1], Y, color='black', linewidth=0.8)
    ax.plot_surface(x0_grid, x1_grid, z_pred, alpha=0.5, cmap='spring', label="Superficie ajustada")
    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_zlabel("x₂")
    ax.set_title("Embedding + Superficie Ajustada")
    plt.tight_layout()
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    # Ruta a CSV
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_csv = os.path.join(base_dir, "..", "..", "BaseDeDatos", "embedding.csv")

    # Cargar datos
    X, Y = cargar_embedding(path_csv)

    # Visualizar embedding
    visualizar_embedding(X, Y)

    # Ajuste y visualización de la superficie
    model, poly, x0_grid, x1_grid, z_pred = ajustar_superficie(X, Y, degree=3, padding_ratio=0.2)
    visualizar_superficie(X, Y, x0_grid, x1_grid, z_pred)