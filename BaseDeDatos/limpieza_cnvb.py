import pandas as pd 
import os

# === CONFIGURACIÓN ===

# Ruta a archivo (Base de Datos Inclu. Financ. 2024-06 -segundo trimestere-)
user_home = os.path.expanduser("~")
bd_cnbv_24_2trim = os.path.join (user_home, "Downloads", "Base_de_Datos_de_Inclusion_Financiera_202406.xlsx")

# Verificar que el archivo exista
if not os.path.exists(bd_cnbv_24_2trim):
	raise FileNotFoundError(f"No se encuentra el archivo en: {bd_cnbv_24_2trim}")

# === CARGAR Y EXPLORAR ===

df_raw = pd.read_excel(bd_cnbv_24_2trim, sheet_name="BD Banca", header=None)

# Mostrar unas cuantas filillas de este pedo
print("Primeras 20 filas crudas del archivo:")
print(df_raw.head(20))