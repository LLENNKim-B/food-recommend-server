from sqlalchemy import create_engine
# database.py
import pyodbc

def get_db_connection():
    try:
        conn = pyodbc.connect(
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=LAPTOP-HJRRM85F;"
            "Database=FoodRecommendDB;"
            "Trusted_Connection=yes;"
        )
        return conn
    except Exception as e:
        print("Database connection failed:", e)
        return None



def get_db_engine():
    try:
        engine = create_engine(
            "mssql+pyodbc://@FoodRecommendDB?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        )
        return engine
    except Exception as e:
        print("Database engine creation failed:", e)
        return None
