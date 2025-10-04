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
