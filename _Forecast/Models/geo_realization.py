import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from embedding import hd_embedding, transformar_serie
from polynomial_model import cargar_embedding, train_and_evaluate

# Use the specified matplotlib style
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

# Specify dataset path
path_csv = "../../BaseDeDatos/rappiCard_data.csv"

# Columns to forecast separately
columns_to_forecast = [
    "Entregas Black",
    "Entregas Cardjolote Black",
    "Entregas Ocean Plastic",
    "Demanda Total Tarjetas"
]

# Embedding configurations
configs = [(3, 1)]  # adjust based on your preference

# Forecast each column separately
for col in columns_to_forecast:
    # Load and preprocess series
    df = pd.read_csv(path_csv, encoding="latin1")
    serie = df[col].dropna().reset_index(drop=True).drop(labels=[0, 1], axis=0).reset_index(drop=True).astype(float)

    # Create embedding
    X_hd = hd_embedding(serie, configs)

    # Save embedding temporarily for model
    temp_embedding_path = f"embedding_{col.replace(' ', '_')}.csv"
    pd.DataFrame(X_hd, columns=[f"x_{i}" for i in range(X_hd.shape[1])]).to_csv(temp_embedding_path, index=False)

    # Load embedding
    X_feats, y_target, X_full = cargar_embedding(temp_embedding_path)

    # Forecast and plot
    metrics = train_and_evaluate(
        X_full,
        train_ratio=0.7,
        degree=1,
        scale=True,
        reg_type='ridge',
        alpha=1.0,
        plot=True,
        forecast_mode='recursive',
        invertir_transf=False
    )

    # Display metrics
    print(f"Metrics for {col}:")
    print(metrics)

plt.show()