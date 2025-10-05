# auth.py
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from database import get_db_connection

# ใช้ HTTPBearer สำหรับ Swagger + Authorization header
bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    ตรวจสอบ token จาก header Authorization
    """
    token = credentials.credentials  # token จริงจาก header

    # ตรวจสอบ token ในฐานข้อมูล
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM tokens WHERE token = ?", (token,))
    record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not record:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # คืนค่า user_id ไปยัง endpoint
    return record[0]
