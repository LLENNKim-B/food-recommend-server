from fastapi import APIRouter, Depends, HTTPException
from database import get_db_connection
from bcrypt import hashpw,checkpw,gensalt
from model.users import User
import secrets
from datetime import datetime
from auth import verify_token
router = APIRouter()

@router.get("/{user_id}")
async def read_user(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_2 WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    print(user)
    return 'Hello world'

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password, user_id FROM user_2 WHERE email = ?", (email,))
    record = cursor.fetchone()
    if record is None or not checkpw(password.encode('utf-8'), record[0].encode('utf-8')):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    user_id = record[1]
    cursor.execute("SELECT username, email FROM user_2 WHERE user_id = ?", (user_id,))
    data = cursor.fetchone()
    cursor.close()
    conn.close()

    # ✅ สร้าง token ใหม่ให้ผู้ใช้
    token = secrets.token_hex(32)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tokens WHERE user_id = ?", (user_id,))
    cursor.execute("INSERT INTO tokens (user_id, token, created_at) VALUES (?, ?, ?)", 
                   (user_id, token, datetime.now()))
    conn.commit()
    cursor.close()
    conn.close()

    return {
        "user_id": user_id,
        "username": data[0],
        "email": data[1],
        "token": token
    }

@router.post("/login")
async def login_user(email: str, password: str):
    conn = get_db_connection()
    cursor = conn.cursor()

    # ตรวจสอบอีเมลในระบบ
    cursor.execute("SELECT user_id, username, email, password FROM user_2 WHERE email = ?", (email,))
    record = cursor.fetchone()

    if record is None or not checkpw(password.encode('utf-8'), record[3].encode('utf-8')):
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid email or password")

    user_id = record[0]

    # ลบ token เดิม (ถ้ามี)
    cursor.execute("DELETE FROM tokens WHERE user_id = ?", (user_id,))

    # สร้าง token ใหม่
    token = secrets.token_hex(32)

    # บันทึก token ลงฐานข้อมูล
    cursor.execute(
        "INSERT INTO tokens (user_id, token, created_at) VALUES (?, ?, ?)",
        (user_id, token, datetime.now())
    )

    conn.commit()
    cursor.close()
    conn.close()

    return {
        "user_id": record[0],
        "username": record[1],
        "email": record[2],
        "token": token
    }

@router.post("/register")
async def register_user(user: User):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "select 1 from user_2 where email = ?", (user.email,)
    )
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = hashpw(user.password.encode('utf-8'), gensalt())
    cursor.execute(
        "INSERT INTO user_2 (username, email, password) VALUES (?, ?, ?)",
        (user.username, user.email, hashed_password.decode('utf-8'))
    )
    conn.commit()
    cursor.close()
    conn.close()

    return {"message": "User registered successfully"}


@router.post("/all")
async def get_all_users(user_id: int = Depends(verify_token)):
    print(f"Authenticated user_id: {user_id}")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id,username,email FROM user_2")
    users = cursor.fetchall()
    cursor.close()
    conn.close()

    return [{"user_id": row[0], "username": row[1], "email": row[2]} for row in users]

@router.post("/check-auth")
async def check_auth(user_id: int = Depends(verify_token)):
    return {"message": f"Authenticated as user_id {user_id}"}