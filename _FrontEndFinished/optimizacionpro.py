import pandas as pd
import pulp

# Leer archivo transpuesto
df = pd.read_csv("polynomial_forecasts.csv", index_col=0)

# Asegurarte de que los índices son "mes 1", "mes 2", etc.
df.index.name = "Mes"

# Nombres de tarjetas (columnas)
tarjetas = df.columns.tolist()

# Nombres de meses ya formateados
meses_formateados = ["mar_2025", "abr_2025", "may_2025", "jun_2025", "jul_2025", "ago_2025"]
meses_originales = df.index.tolist()[:6]  # por si hay más filas en el CSV

# Crear diccionario de demanda
demanda = {
    meses_formateados[i]: {
        tarjeta: df.loc[meses_originales[i], tarjeta] for tarjeta in tarjetas
    }
    for i in range(len(meses_formateados))
}

print(demanda)

# Costos ficticios (ajusta con los reales si los tienes)
costo = {tarjeta: 10 + i for i, tarjeta in enumerate(tarjetas)}

# Modelo
modelo = pulp.LpProblem("Produccion_tarjetas_mes_a_mes", pulp.LpMinimize)

# Variables de decisión
x = {
    (mes, tarjeta): pulp.LpVariable(f"x_{mes}_{tarjeta}", lowBound=0, cat="Integer")
    for mes in meses_formateados
    for tarjeta in tarjetas
}

# Objetivo: minimizar el costo total
modelo += pulp.lpSum(x[(mes, tarjeta)] * costo[tarjeta] for mes in meses_formateados for tarjeta in tarjetas)

# Restricciones: satisfacer demanda mensual
for mes in meses_formateados:
    for tarjeta in tarjetas:
        modelo += x[(mes, tarjeta)] >= demanda[mes][tarjeta]

# Resolver
modelo.solve()

# Resultados

for mes in meses_formateados:
    print()
    print(f"\n--- Lote de producción: {mes} ---")
    for tarjeta in tarjetas:
        print(f"{tarjeta}: {int(x[(mes, tarjeta)].varValue)}")

