# Archivo para centralizar la conexión a la base de datos con SQLAlchemy
from sqlalchemy import create_engine

db_user = 'postgres'
db_password = 'Tc25pirc$'
db_host = 'rappicarddb.cb60ue2usube.us-east-2.rds.amazonaws.com'
db_port = '5432'  # Por defecto en PostgreSQL
db_name = 'rappicarddb'

db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(db_url)


