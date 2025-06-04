#Registro de todos los blueprints
from .registros import registros_bp
from .registros import columnas_bp

def register_routes(app):
    app.register_blueprint(registros_bp)
    app.register_blueprint(columnas_bp)

from flask import Flask
from routes import register_routes

app = Flask(__name__)
register_routes(app)

if __name__ == '__main__':
    app.run(debug=True)