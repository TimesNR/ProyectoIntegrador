from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import subprocess
import psycopg2
import pandas as pd
import os

from routes.registros import registros_bp, columnas_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(registros_bp)
app.register_blueprint(columnas_bp)

# Configuración de PostgreSQL
DB_CONFIG = {
    'dbname': 'rappicarddb',
    'user': 'postgres',
    'password': 'Tc25pirc$',
    'host': 'localhost',
    'port': '5432'
}

@app.route('/forecast', methods=['GET'])
def forecast():
    try:
        subprocess.run(['python', 'version21abril.py'], check=True)
        return jsonify({"mensaje": "Forecast generado correctamente"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Error al generar el forecast: {str(e)}"}), 500

@app.route('/forecast_csv', methods=['GET'])
def forecast_csv():
    return send_file('output_forecast.csv', mimetype='text/csv')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        columnas = list(df.columns)
        placeholders = ','.join(['%s'] * len(columnas))
        columnas_sql = ','.join(columnas)

        for row in df.itertuples(index=False, name=None):
            cur.execute(
                f"INSERT INTO {'tarjetas_data'} ({columnas_sql}) VALUES ({placeholders})",
                row
            )

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"mensaje": "Tabla actualizada correctamente"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔁 NUEVA RUTA: Ejecutar polynomial_model_v2.py
@app.route('/run-polynomial-model', methods=['GET'])
def run_polynomial_model():
    try:
        result = subprocess.run(
            ['python', 'polynomial_model_v2.py'],
            check=True,
            capture_output=True,
            text=True
        )
        print("[STDOUT]", result.stdout)
        print("[STDERR]", result.stderr)
        return jsonify({"status": "ok", "output": result.stdout})
    except subprocess.CalledProcessError as e:
        print("[ERROR - STDOUT]", e.stdout)
        print("[ERROR - STDERR]", e.stderr)
        return jsonify({
            "status": "error",
            "message": "Falló el script",
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500


# 📄 NUEVA RUTA: Servir polynomial_forecasts.csv
@app.route('/get-forecast-data', methods=['GET'])
def get_forecast_data():
    file_path = 'polynomial_forecasts.csv'
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='text/csv')
    else:
        return jsonify({"error": "CSV file not found"}), 404

if __name__ == '__main__':
    app.run(port=5000)
