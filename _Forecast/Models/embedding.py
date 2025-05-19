import pandas as pd
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# === CONFIGURACIÓN ===

# Output path
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../BaseDeDatos/embedding.csv"))

sns.set(style="whitegrid")
matplotlib.use('TkAgg')

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.facecolor"] = "#1e1e1e"
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["text.color"] = "white"
plt.rcParams["figure.facecolor"] = "#1e1e1e"
plt.rcParams["savefig.facecolor"] = "#1e1e1e"

# Path a base de datos
path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../BaseDeDatos/rappiCard_data.csv")) 

# Columna objetivo
<<<<<<< HEAD
col_objetivo = "# de Usuarios"
=======
col_objetivo = "Demanda Total Tarjetas" # (# de Usuarios) // (Demanda Total Tarjetas) // (Stocks Cardbody Black)
>>>>>>> 95f3a28fc800cadb06ff9dfd2b4720a588f751a7

# Carga de la serie
df = pd.read_csv(path2, encoding="latin1")
serie = df[col_objetivo].dropna().reset_index(drop=True)
serie = serie.drop(labels=[0, 1], axis=0).reset_index(drop=True)
serie = serie.dropna().astype(float)

# === FUNCION DE EMBEDDING BÁSICA (igual que antes) ===
def embedding(serie, d, tau):
    n = len(serie)
    emb = []
    # Para cada índice i >= (d-1)*tau
    for i in range((d - 1) * tau, n):
        # Construir un vector con d puntos a saltos de tau
        punto = [serie[i - (j * tau)] for j in range(d)]
        # Lo invertimos para que sea [x_{t-2}, x_{t-1}, x_t] en lugar de [x_t, x_{t-1}, x_{t-2}]
        emb.append(punto[::-1])
    return np.array(emb, dtype=float)

# === NUEVO: HD-EMBEDDING CON VARIOS "MINI-EMBEDDINGS" ===
def hd_embedding(serie, configs):
    """
    Genera un 'HD-embedding' combinando varios mini-embeddings.
    configs: lista de (dimension, delay), por ejemplo [(3,1), (3,2), (3,4)]
             donde dimension=d, delay=tau.

    Retorna: np.array de shape (N, sum(d_i)).
    Nota: Para alinear la longitud, tomamos el mínimo 'n_filas' 
          de todos los mini-embeddings y luego hacemos hstack.
    """
    all_embs = []
    min_rows = None

    for (d, tau) in configs:
        emb = embedding(serie, d, tau)
        all_embs.append(emb)
        # Cuántas filas generó este embedding
        if min_rows is None:
            min_rows = emb.shape[0]
        else:
            min_rows = min(min_rows, emb.shape[0])
    
    # Ahora recortamos cada embedding a 'min_rows'
    # y hacemos np.hstack
    trimmed = []
    for emb in all_embs:
        trimmed.append(emb[-min_rows:])  # tomamos las últimas min_rows filas
    final_emb = np.hstack(trimmed)  # (min_rows, sum_of_dims)
    return final_emb

# === Visualizaciones (opcional) ===
def plot_serie(serie):
    plt.figure(figsize=(10, 5))
    plt.plot(serie, marker='o', color='peru', label="Serie original")
    plt.title(f"Serie de Tiempo: {col_objetivo}")
    plt.xlabel("Mes")
    plt.ylabel("Cantidad")
    plt.xticks(ticks=np.arange(len(serie)), rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_embedding_2D(X):
    plt.plot(X[:,0], X[:,1], marker='o', color='magenta')
    plt.title("Embedding 2D de la Serie")
    plt.xlabel("x₀")
    plt.ylabel("x₁")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_embedding_3D(X):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], X[:, 2], 'o', color='peru')
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color='red', linewidth=0.8)
    ax.set_title("Embedding 3D de la Serie")
    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_zlabel("x₂")
    plt.tight_layout()
    plt.show()

# === TRANSFORMACION DE DATOS ===
def transformar_serie(serie, metodo="log", offset=1e-9):
    """
    Transforma la serie univariada según el método especificado.

    Parámetros:
      serie: np.array o lista con los valores de la serie original.
      metodo: cadena que especifica el tipo de transformación. Por ejemplo,
              "log" para aplicar logaritmo.
      offset: valor pequeño para evitar el logaritmo de cero (en caso de que la
              serie tenga ceros).

    Retorna:
      serie_transformada: np.array con la serie transformada.
    """
    serie = np.array(serie, dtype=float)
    if metodo == "log":
        # Usamos log(x + offset) para evitar problemas con ceros
        return np.log(serie + offset)
    # Aquí se pueden agregar otras transformaciones, por ejemplo:
    # if metodo == "log1p":
    #     return np.log1p(serie)
    # if metodo == "sqrt":
    #     return np.sqrt(serie)
    # ...
    # Por defecto, no transforma la serie:
    return serie

# === MAIN DEMO ===
if __name__ == "__main__":
    # Ejemplo: definimos k "mini-embeddings"
    # digamos 3 mini-embeddings:
    #  1) dimension=3, delay=1
    #  2) dimension=3, delay=2
    #  3) dimension=3, delay=4
    # Ajustalo a tus necesidades
    configs = [
        (3, 1)
        # (3, 2),
        # (3, 4),
        # (3, 5),
        # (3, 6),
        # (5, 2),
        # (5, 4),
        # (7, 1)
    ]

    # Generar HD-embedding
    X_hd = hd_embedding(serie, configs)
    print("Shape del HD-Embedding:", X_hd.shape)  # (N, 9) si cada mini-embedding es dimension=3 => 3*3=9

    # Guardar en CSV (columnas x_0, x_1, x_2, ..., x_8)
    col_names = [f"x_{i}" for i in range(X_hd.shape[1])]
    df_embedding = pd.DataFrame(X_hd, columns=col_names)
    df_embedding.to_csv(output_path, index=False)
    print(f"HD-Embedding guardado en {output_path}")

    # Si quieres plotear alguna sección, por ejemplo las primeras 3 columnas:
    # Ojo: ya no es un "embedding de 3D" literalmente, sino 9D
    # Podrías plotear las primeras 3 como un 3D si gustas:
    if X_hd.shape[1] >= 3:
        plot_embedding_3D(X_hd[:, :3])
    else:
        plot_embedding_2D(X_hd)
    
    # Por último, si quieres ver la serie original
    plot_serie(serie)
    # plot_serie(transformar_serie(serie))