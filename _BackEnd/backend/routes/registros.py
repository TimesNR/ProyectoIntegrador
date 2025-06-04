from flask import Blueprint, request, jsonify
from sqlalchemy import text
from db import engine
import pandas as pd
import os
from sqlalchemy import text
from sqlalchemy.sql import bindparam


registros_bp = Blueprint('registros', __name__)
columnas_bp  = Blueprint('columnas', __name__)

def obtener_columnas_patron(patron: str  =  ""):
    if patron:
        query = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'tarjetas_data'
              AND table_schema = 'public'
              AND column_name ILIKE :patron
            ORDER BY ordinal_position
        """)
        params = {"patron": f'%{patron}%'}
    else:
        query = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'tarjetas_data'
              AND table_schema = 'public'
            ORDER BY ordinal_position
        """)
        params = {}

    with engine.connect() as conn:
        result = conn.execute(query, params)
        columnas = [row[0] for row in result.fetchall()]
    return columnas


@registros_bp.route('/agregar_registro', methods=['POST'])
def agregar_registro():
    try:
        data = request.json
        print("Datos recibidos:", data)

        columnas = list(data.keys())
        columnas_sql = ', '.join(columnas)
        placeholders = ', '.join([f":{col}" for col in columnas])
        query = text(f"INSERT INTO tarjetas_data ({columnas_sql}) VALUES ({placeholders})")

        print("Consulta SQL generada:", query)

        with engine.begin() as conn:
            conn.execute(query, data)

        return jsonify({"mensaje": "Registro agregado correctamente"}), 200
    except Exception as e:
        print("ERROR AL INSERTAR:", e)
        return jsonify({"error":str(e)}),500

@registros_bp.route('/agregar_tarjeta', methods=['POST'])
def agregar_tarjeta():
    try:
        data = request.json
        nombre_columna = data.get('nombre_columna')

        if not nombre_columna:
            return jsonify({"error": "Falta el nombre de la columna"}), 400

        columnas_existentes = obtener_columnas_patron()
        if nombre_columna in columnas_existentes:
            return jsonify({"error": f"La columna '{nombre_columna}' ya existe."}), 400

        # Por defecto, agregamos la columna como tipo INTEGER
        query = text(f"ALTER TABLE tarjetas_data ADD COLUMN {nombre_columna} INTEGER")

        with engine.begin() as conn:
            conn.execute(query)

        return jsonify({"mensaje": f"Columna '{nombre_columna}' agregada exitosamente."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@registros_bp.route('/quitar_tarjeta', methods=['POST'])
def quitar_tarjeta():
    try:
        data = request.json
        nombre_columna = data.get('nombre_columna')

        if not nombre_columna:
            return jsonify({"error": "Falta el nombre de la columna"}), 400

        columnas_existentes = obtener_columnas_patron()
        if nombre_columna not in columnas_existentes:
            return jsonify({"error": f"La columna '{nombre_columna}' no existe."}), 400

        query = text(f"ALTER TABLE tarjetas_data DROP COLUMN {nombre_columna}")

        with engine.begin() as conn:
            conn.execute(query)

        return jsonify({"mensaje": f"Columna '{nombre_columna}' eliminada exitosamente."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@columnas_bp.route('/columnas', methods=['GET'])

def obtener_columnas():
    try:
        columnas = obtener_columnas_patron()
        return jsonify(columnas)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from sqlalchemy import insert, Table, MetaData

@registros_bp.route('/actualizar_con_tabla', methods=['POST'])
def actualizar_con_tabla():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No se recibió ningún archivo"}), 400

        # Leer el archivo Excel como DataFrame
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip()  # Quita espacios al inicio y final

        # Convertir fechas a string (opcional, si no lo soporta el engine)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d')

        # Preparar la tabla
        metadata = MetaData()
        metadata.reflect(bind=engine)
        tarjetas_data = Table('tarjetas_data', metadata, autoload_with=engine)

        # Insertar registros
        with engine.begin() as conn:
            conn.execute(insert(tarjetas_data), df.to_dict(orient='records'))

        return jsonify({"mensaje": f"Archivo procesado exitosamente. {len(df)} registros insertados."}), 200

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


@registros_bp.route('/eliminar_por_fecha', methods=['POST'])
def eliminar_por_fecha():
    try:
        data = request.get_json()
        fecha = data.get('fecha')
        if not fecha:
            return jsonify({"error": "Fecha no proporcionada"}), 400

        with engine.begin() as conn:
            conn.execute(text("DELETE FROM tarjetas_data WHERE fecha = :fecha"), {"fecha": fecha})

        return jsonify({"mensaje": f"Fila con fecha {fecha} eliminada."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@registros_bp.route('/registros', methods=['GET'])
def obtener_registros():
    try:
        query = text("SELECT * FROM tarjetas_data")
        with engine.connect() as conn:
            result = conn.execute(query)
            rows = [dict(row._mapping) for row in result]
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

