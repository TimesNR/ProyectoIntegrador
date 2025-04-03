import pandas as pd
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# === CONFIGURACIÓN ===

# Output path (Embedding)
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../BaseDeDatos/embedding.csv"))

# Estilo de seaborn
sns.set(style="whitegrid")

# Estilo matplotlib para gui
matplotlib.use('TkAgg')

# Estilo pyplot
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.facecolor"] = "#1e1e1e"
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["text.color"] = "white"
plt.rcParams["figure.facecolor"] = "#1e1e1e"
plt.rcParams["savefig.facecolor"] = "#1e1e1e"

# Ruta a base de datos (Datos Historicos CNBV)
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../BaseDeDatos/BD_DatosHist_Filtrado.csv"))

# Parametros de embedding
delay = 1 # tau (τ)
dimension = 3 # d
col_objetivo = "24" # <- Cambiar este vato pa otra serie si se ocupa i.g. creditos emitidos

# === CARGA DE DATOS ===

df = pd.read_csv(path)
serie = df[col_objetivo].dropna().reset_index(drop=True) # Asegurarse que la columna esta limpia (lista pa danzar)
serie = serie.drop(labels=[0, 1], axis=0,).reset_index(drop=True)
serie = serie.dropna().astype(float)

# === FUNCION DE EMBEDDING ===

def embedding(serie, d, tau):
	n = len(serie)
	emb = []

	for i in range((d - 1) * tau, n):
		punto = [serie[i - (j * tau)] for j in range(d)]
		emb.append(punto[::-1])

	return np.array(emb)

# === GENERAR EMBEDDING ===

X = embedding(serie, dimension, delay)
X = np.array(X, dtype=float)

df_embedding = pd.DataFrame(X, columns=[f"x_{i}" for i in range(dimension)]) # Visualizar array mas chidamente
df_embedding.to_csv(output_path, index=False)

# === VISUALIZACION ===

def plot_serie(serie):
    """Visualiza la serie original de tiempo."""
    plt.plot(serie, marker='o', color='peru', label="Serie original")
    plt.title("Serie de Tiempo: Tarjetas de Crédito Emitidas")
    plt.xlabel("Trimestre")
    plt.ylabel("Cantidad")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_embedding(X, dimension):
    """Visualiza el embedding en 2D o 3D."""
    if dimension == 3 and X.shape[1] < 3:
        raise ValueError("Embedding no tiene suficientes dimensiones para graficar en 3D.")

    if dimension == 2:
        plt.plot(X[:, 0], X[:, 1], marker='o', color='magenta')
        plt.title("Embedding 2D de la Serie")
        plt.xlabel("x₀")
        plt.ylabel("x₁")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif dimension == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[:, 0], X[:, 1], X[:, 2], 'o', color='peru')
        ax.plot(X[:, 0], X[:, 1], X[:, 2], color='black', linewidth=0.8)
        ax.set_title("Embedding 3D de la Serie")
        ax.set_xlabel("x₀")
        ax.set_ylabel("x₁")
        ax.set_zlabel("x₂")
        plt.tight_layout()
        plt.show()

# === CALLS ===

plot_serie(serie)
plot_embedding(X, 3)