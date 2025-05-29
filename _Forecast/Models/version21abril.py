# %%
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


import pandas as pd

# Ruta completa utilizando barras normales
file_path = r"C:/Users/IKER/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/Desktop/Proyecto Ingeniería Matemática/Datos/DATOSLIMPIOSRAPPI.csv"

# Si prefieres usar barras invertidas, asegúrate de escapar las barras invertidas con doble barra invertida '\\'
# file_path = r"C:\\Users\\IKER\\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\\Desktop\\Proyecto Ingeniería Matemática\\Datos\\DATOSLIMPIOSRAPPI.csv"

# Leer el archivo CSV
df = pd.read_csv(file_path)

# %%

# Función para calcular SMAPE
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

# Búsqueda exhaustiva de parámetros SARIMAX
def sarimax_grid_search(series, 
                        p_range=range(0, 3),
                        d_range=range(0, 2),
                        q_range=range(0, 3),
                        P_range=range(0, 2),
                        D_range=range(0, 2),
                        Q_range=range(0, 2),
                        s=12,
                        train_size=0.8,
                        verbose=True):

    best_score = np.inf
    best_model = None
    best_order = None
    best_seasonal_order = None

    series = series.dropna()
    train_size = int(len(series) * train_size)
    train, test = series.iloc[:train_size], series.iloc[train_size:]

    param_combinations = list(product(p_range, d_range, q_range, P_range, D_range, Q_range))
    total_combinations = len(param_combinations)

    if verbose:
        print(f"Probando {total_combinations} combinaciones...")

    for p, d, q, P, D, Q in tqdm(param_combinations):
        order = (p, d, q)
        seasonal_order = (P, D, Q, s)

        try:
            model = SARIMAX(train,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit(disp=False)
            forecast = results.forecast(steps=len(test))
            score = smape(test.values, forecast.values)

            if score < best_score:
                best_score = score
                best_model = results
                best_order = order
                best_seasonal_order = seasonal_order

        except Exception as e:
            if verbose:
                print(f"Error con orden {order}, seasonal {seasonal_order}: {e}")

    if verbose:
        print(f"\nMejor SMAPE: {best_score:.2f}%")
        print(f"Orden: {best_order}, Orden estacional: {best_seasonal_order}")

    return best_model, best_order, best_seasonal_order, best_score



def plot_sarimax_forecast(series, order, seasonal_order, forecast_steps=12):
    series = series.dropna()

    model = SARIMAX(series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Construir índice para el forecast
    if isinstance(series.index, pd.DatetimeIndex):
        freq = series.index.inferred_freq or 'M'
        forecast_index = pd.date_range(
            start=series.index[-1] + pd.tseries.frequencies.to_offset(freq),
            periods=forecast_steps,
            freq=freq
        )
    else:
        # Calcular el paso del índice (si hay al menos dos valores)
        if len(series.index) >= 2:
            step = series.index[-1] - series.index[-2]
        else:
            step = 1
        forecast_index = np.arange(series.index[-1] + step,
                                   series.index[-1] + step * (forecast_steps + 1),
                                   step)

    # Extraer el nombre de la serie si tiene uno
    series_name = series.name if series.name is not None else 'Serie de tiempo'

    # Graficar
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Serie original')
    plt.plot(forecast_index, forecast_mean, label='Pronóstico', color='orange')

    
    plt.title(f'Forecasting para {series_name}')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def usar_sarimax_con_forecast(series,
                               p_range=range(0, 3),
                               d_range=range(0, 2),
                               q_range=range(0, 3),
                               P_range=range(0, 2),
                               D_range=range(0, 2),
                               Q_range=range(0, 2),
                               s=12,
                               forecast_steps=12,
                               train_size=0.8,
                               verbose=True):
    
    # Llamar a la búsqueda de modelo
    best_model, order, seasonal_order, smape_score = sarimax_grid_search(
        series,
        p_range=p_range,
        d_range=d_range,
        q_range=q_range,
        P_range=P_range,
        D_range=D_range,
        Q_range=Q_range,
        s=s,
        train_size=train_size,
        verbose=verbose
    )

    # Llamar a la función de graficado con los mejores parámetros
    plot_sarimax_forecast(series, order, seasonal_order, forecast_steps=forecast_steps)

    return best_model, order, seasonal_order, smape_score




# %%
x=df['Demanda Total Tarjetas']
# Solo llamas a esta función, y se hace todo
usar_sarimax_con_forecast(x)



