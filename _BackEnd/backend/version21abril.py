# %%
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import kpss
import statsmodels as sm
import numpy as np


from sklearn.metrics import mean_squared_error
import warnings

# Ignorar todos los warnings
warnings.filterwarnings("ignore")

# Crear el rango de fechas mensual
index = pd.date_range(start='2023-01-01', end='2025-02-01', freq='MS')
df = pd.read_csv('rappiCard_data.csv', encoding="windows-1252")
df.index=index
df=df.drop('Fecha',axis=1)
df=df.fillna(0)

# %%
# Definir separación de los datos
separacion = int(0.8* len(x))
xtrain = x[:separacion]
xtest = x[separacion:]

# %%

# Ajustar el modelo con los datos de entrenamiento
model = ARIMA(xtrain, order=(2, 0, 2))
model_fit = model.fit()  # Aquí es donde ajustas el modelo con xtrain

# %%
df_ = pd.DataFrame(model_fit.fittedvalues)
df_.to_csv("ARIMA_demandatotal.csv")