import pandas as pd 
import os
from tabulate import tabulate

# === CONFIGURACIÓN ===

## Configurar formato de output en consola
# Mostrar más columnas y más ancho para que no te las trunque
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 50)  # Máximo ancho por celda

## Leer 1er Archivo: Base_de_Datos_de_Inclusion_Financiera_202406.xlsx
# Ruta a archivo (Base de Datos Inclu. Financ. 2024-06 -segundo trimestere-)
user_home = os.path.expanduser("~")
bd_cnbv_24_2trim = os.path.join (user_home, "Downloads", "Base_de_Datos_de_Inclusion_Financiera_202406.xlsx")

# Verificar que el archivo exista
if not os.path.exists(bd_cnbv_24_2trim):
	raise FileNotFoundError(f"No se encuentra el archivo en: {bd_cnbv_24_2trim}")

# === CARGAR Y EXPLORAR ===

## Hoja de Datos Históricos
df_HistData = pd.read_excel(bd_cnbv_24_2trim, sheet_name="BD Datos históricos", header=None, skiprows=9)
df_HistData.drop(df_HistData.columns[0], axis=1, inplace=True)

df_HistData.dropna(how='all', inplace=True)

# Obtener nombres de columnas (temas y subtemas)
temas = pd.Series(df_HistData.iloc[0]).ffill()
subtemas = pd.Series(df_HistData.iloc[1]).ffill()

# print("Temas principales detectados:")
# for t in temas.unique():
# 	print("-", t)

# print("\n Subtemas detectados:")
# for s in subtemas.unique():
# 	print("->", s)

# Estructura de DF (Tema->Subtema->Num.Col.)
columnas = df_HistData.columns
df_structure = pd.DataFrame({
	"columna": columnas,
	"tema": temas.reset_index(drop=True),
	"subtema": subtemas.reset_index(drop=True)
	})

# print('\n Estructura completa tema-subtema-columna:')
# print(df_structure.head())

# Funcion para filtrar datos de DF original usando estructura
def filtra_cols(df_data, df_estructura, palabra_clave):
	"""
	Devuelve columnas de df_data cuyo "tema" contenga la palabra_clave.
	Incluye siempre las priemras 3 columnas de indice temporal.
	"""
	# Filtrar columnas cuyo tema contenga palabra_clave
	columnas_filtradas = df_estructura[
		df_estructura["tema"].str.contains(palabra_clave, case=False, na=False)
		]["columna"].tolist()

	# Asegutar que las 3 primeras cols. (clave periodo, año, trimestre) esten incluidas
	columnas_indice = df_estructura['columna'].iloc[:3].tolist()
	columnas_resultado = columnas_indice + columnas_filtradas

	return df_data[columnas_resultado]

# Creamos nuevo DF (cols. de df_HistData con tema que incuya 'crédito')
df_credito = filtra_cols(df_HistData, df_structure, "crédito")

# print(df_credito.head())

# Filtrar columnas interesantes
columnas_chilas = list(df_HistData.columns[23:31]) + \
				  list (df_HistData.columns[40:43]) + \
				  list(df_HistData.columns[49:52])
columnas_index = list(df_HistData.columns[:3])
df_credito_filtrado = df_HistData[columnas_index + columnas_chilas]

print(df_credito_filtrado.head())

# === EXPORTAR BASES DE DATOS ===
## Base de datos 1: Datos Historicos (Filtrados)]
df_credito_filtrado.to_csv("BD_DatosHist_Filtrado.csv", index=False)
#print("Archivo exportado bien vgas.")