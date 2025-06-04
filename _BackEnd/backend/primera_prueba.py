# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# aquí va la ruta del archivo que les mandé

ruta = r"ArchivosRappi.xlsx"
df = pd.read_excel(ruta)
s=df['Data Tec'].values
df.index=s
df=df.drop('Data Tec',axis=1)
df=df.T
df=df.replace('-',0)
df=df.fillna(df.mean())

x=df['Entregas Demanda Total']

# %%
modelo = ARIMA(x, order=(1,2,0))  # Reemplaza p, d, q con los valores obtenidos
resultado = modelo.fit()

# %%
# Predecir con el conjunto de test (fuera de muestra)
predicciones = resultado.forecast(steps=12)

# %%
# Generar las fechas para los próximos 12 meses
forecast_index = pd.date_range(start=x.index[-1], periods=13, freq='M')[1:]

# Graficar la serie de tiempo original
plt.figure(figsize=(10, 6))
plt.plot(x, label='Serie Temporal Histórica', color='blue')

# Graficar el pronóstico de los próximos 12 meses
plt.plot(forecast_index, predicciones, label='Pronóstico (12 meses)', color='red', linestyle='--')

# Etiquetas y título del gráfico
plt.title('Pronóstico de Demanda total de tarjetas')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.axhline(0)
plt.legend()

# Guardar gráfico
plt.savefig("grafica.png")
plt.close()

# Imprimir salida para Flask
print("Script ejecutado correctamente. Predicción generada.")

