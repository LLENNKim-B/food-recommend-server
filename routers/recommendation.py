# routers/recommendation.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pyodbc
from datetime import datetime
from database import get_db_connection
router = APIRouter()

# ==============================
# 1. เชื่อม SQL Server
# ==============================
conn = pyodbc.connect(
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=LAPTOP-HJRRM85F;"
    "Database=FoodRecommendDB;"
    "Trusted_Connection=yes;"
)
cursor = conn.cursor()

# ==============================
# 2. โหลดตารางจาก DB
# ==============================
users = pd.read_sql("SELECT * FROM User_2", conn)
menu_lookup = pd.read_sql("SELECT * FROM menus_2", conn)
df_final = pd.read_csv("df_final2.csv")  
user_ratings = pd.read_sql("SELECT * FROM Ratings", conn)
disease_prep = pd.read_csv(r"preposition/disease_prep.csv")  # โหลด health constraints

# normalize columns
df_final.columns = df_final.columns.str.lower()
menu_lookup.columns = menu_lookup.columns.str.lower()
user_ratings.columns = user_ratings.columns.str.lower()
users.columns = users.columns.str.lower()

# ==============================
# 3. ฟังก์ชันช่วยคำนวณ
# ==============================
def calculate_age(birthdate):
    if pd.isna(birthdate):
        return None
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

def calculate_bmi(weight, height_cm):
    if weight is None or height_cm is None:
        return None
    height_m = height_cm / 100
    return round(weight / (height_m**2), 2)

def calculate_bmr(weight, height_cm, age, gender):
    if None in [weight, height_cm, age]:
        return None
    if gender.lower() == 'male':
        return 88.36 + 13.4*weight + 4.8*height_cm - 5.7*age
    else:
        return 447.6 + 9.25*weight + 3.1*height_cm - 4.33*age

def calculate_tdee(bmr, activity_factor=1.55):
    if bmr is None:
        return None
    return bmr * activity_factor

def get_latest_mood(user_id):
    query = f"""
        SELECT cu.user_id, m.moods
        FROM CurrentUserMood cu
        JOIN Mood m ON cu.mood_id = m.mood_id
        WHERE cu.user_id = {user_id}
        ORDER BY cu.timestamp DESC
    """
    mood_df = pd.read_sql(query, conn)
    if mood_df.empty:
        return None
    return mood_df.iloc[0]['moods'].lower()

# ==============================
# 4. ฟังก์ชัน health constraints
# ==============================
target_diseases = [
    "diabetes", "hypertension", "gout", "obesity", "kidney",
    "dyslipidemia", "heart_disease", "celiac", "lactose_intolerance",
    "anemia", "hyperuricemia"
]

def filter_by_health_constraints(user_id, df_final, df_menu_lookup):
    user_df = users[users['user_id'] == user_id]
    if user_df.empty:
        print(f"⚠️ User ID {user_id} ไม่มีในฐานข้อมูล")
        return df_final
    user = user_df.iloc[0]

    user_diseases = [d for d in target_diseases if d in str(user['disease']).split(',')]

    if not user_diseases:
        return df_final[["food_id"]]

    nutrient_cols = ['food_id','calories','protein','fat','carbs','sugar','sodium','cholesterol','potassium','saturated_fat']
    df_nutrients = df_menu_lookup[nutrient_cols].copy()
    df_nutrients.columns = [c.lower() for c in df_nutrients.columns]

    df_filtered = df_final[['food_id']].drop_duplicates().merge(df_nutrients, on='food_id', how='left')

    for disease in user_diseases:
        disease_df = disease_prep[disease_prep['disease'] == disease]
        if disease_df.empty:
            print(f"⚠️ ไม่มีข้อมูลโรค {disease} ใน disease_prep")
            continue
        disease_row = disease_df.iloc[0]

        avoid_nutrients = str(disease_row['avoid_nutrients']).split(',')
        for nutrient in avoid_nutrients:
            nutrient = nutrient.strip().lower()
            if nutrient and nutrient != "none" and nutrient in df_filtered.columns:
                if nutrient == "sugar":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 10]
                elif nutrient == "sodium":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 400]
                elif nutrient == "fat":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 20]
                elif nutrient == "calories":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 600]
                elif nutrient == "cholesterol":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 100]
                elif nutrient == "potassium":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 400]
                elif nutrient == "protein":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 20]
                elif nutrient == "saturated_fat":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 10]

        avoid_ingredients = str(disease_row['avoid_ingredients']).split(',')
        for ing in avoid_ingredients:
            ing = ing.strip()
            if ing and ing != "none" and ing in df_final.columns:
                df_filtered = df_filtered[df_final[ing] <= 0.1]

    return df_filtered[["food_id"]]

# ==============================
# 5. Content-Based Recommendation
# ==============================
def content_based_recommend(user_id, top_n=5):
    user_df = users[users['user_id'] == user_id]
    if user_df.empty:
        print(f"⚠️ User ID {user_id} ไม่มีในฐานข้อมูล")
        return df_final
    user = user_df.iloc[0]

    # คำนวณ bmi/bmr/tdee
    age = calculate_age(user['birthdate'])
    bmi = calculate_bmi(user['weight'], user['height'])
    bmr = calculate_bmr(user['weight'], user['height'], age, user['gender'])
    tdee = calculate_tdee(bmr)
    current_mood = get_latest_mood(user_id)

    # กรอง diet_type
    if user['diet_type'] == 'veg':
        df_filtered = df_final[df_final['veg_non_veg'] == 1].copy()
    else:
        df_filtered = df_final.copy()

    # กรอง dislikes
    dislikes = str(user['dislikes']).split(',')
    for ing in dislikes:
        ing = ing.strip()
        if ing in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[ing] == 0]

    # กรองตาม health constraints
    allowed_foods = set(filter_by_health_constraints(user_id, df_final, menu_lookup)["food_id"].unique())
    df_filtered = df_filtered[df_filtered['food_id'].isin(allowed_foods)]

    # Features
    features = [c for c in df_filtered.columns if c not in ["food_id","food_name"]]
    df_filtered[features] = df_filtered[features].fillna(0)

    # User vector
    user_vector = np.zeros(len(features))
    if "calories" in features:
        user_vector[features.index("calories")] = tdee * 0.9
    if "protein" in features:
        user_vector[features.index("protein")] = 20 if bmi >= 25 else 15
    if "fat" in features:
        user_vector[features.index("fat")] = 20 if bmi < 18.5 else 10

    # Mood
    mood_cols = [col for col in features if col in ["happy","neutral","stressed","sad"]]
    for mc in mood_cols:
        user_vector[features.index(mc)] = 0
    if current_mood in mood_cols:
        user_vector[features.index(current_mood)] = 1

    # Likes
    likes = str(user['likes']).split(',')
    for like in likes:
        like = like.strip()
        if like in features:
            user_vector[features.index(like)] = 1

    similarity = cosine_similarity(df_filtered[features].values, [user_vector])
    top_indices = np.argsort(similarity[:,0])[::-1][:top_n]

    food_ids = df_filtered.iloc[top_indices]['food_id'].values
    result = menu_lookup[menu_lookup['food_id'].isin(food_ids)][
        ["food_id","food_name","calories","protein","carbs","fat","sugar","sodium"]
    ]
    result = result.set_index('food_id').reindex(food_ids).reset_index()

    return result

# ==============================
# 6. Collaborative Recommendation (ใช้ SVD ของ SciPy)
# ==============================
def collaborative_recommend(user_id, top_n=5, include_eaten=True):
    if user_ratings.empty:
        return pd.DataFrame()

    # จัดการกรณี duplicate
    ratings_matrix = user_ratings.groupby(['user_id','food_id'])['rating'].mean().unstack(fill_value=0)
    
    user_ids = ratings_matrix.index.tolist()
    food_ids = ratings_matrix.columns.tolist()
    R = ratings_matrix.values

    # Centering
    user_ratings_mean = np.mean(R, axis=1).reshape(-1,1)
    R_demeaned = R - user_ratings_mean

    # SVD
    U, sigma, Vt = svds(R_demeaned, k=min(20, min(R.shape)-1))
    sigma = np.diag(sigma)
    R_pred = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean

    pred_df = pd.DataFrame(R_pred, index=user_ids, columns=food_ids)

    if user_id not in pred_df.index:
        return pd.DataFrame()

    allowed_foods = set(filter_by_health_constraints(user_id, df_final, menu_lookup)["food_id"].unique())
    eaten_foods = set(user_ratings[user_ratings['user_id'] == user_id]['food_id'].unique())

    if include_eaten:
        candidate_foods = list(allowed_foods)
    else:
        candidate_foods = list(allowed_foods - eaten_foods)

    # ✅ กรองให้เฉพาะ food_ids ที่ pred_df มี
    candidate_foods_in_pred = [f for f in candidate_foods if f in pred_df.columns]
    if not candidate_foods_in_pred:
        return pd.DataFrame()

    pred_scores = pred_df.loc[user_id, candidate_foods_in_pred].sort_values(ascending=False)
    top_food_ids = pred_scores.head(top_n).index.tolist()


    result = menu_lookup[menu_lookup['food_id'].isin(top_food_ids)][
        ["food_id", "food_name", "calories", "protein", "carbs", "fat", "sugar"]
    ]
    result['pred_score'] = pred_scores[top_food_ids].values
    result = result.set_index('food_id').reindex(top_food_ids).reset_index()

    return result

# ==============================
# 7. Hybrid Recommendation
# ==============================
def hybrid_recommend(user_id, top_n=5, alpha=0.5):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT 1 FROM User_2 WHERE user_id = ?", user_id)
    row = cursor.fetchone()
    if(not row):
        # print("Hello world")
        return [{"message": "เอ๋อ"}];
    content_rec = content_based_recommend(user_id, top_n=20)
    collab_rec = collaborative_recommend(user_id, top_n=20)

    if collab_rec.empty:
        return content_rec.head(top_n)[["food_id","food_name", "calories", "protein", "carbs", "fat","sugar","sodium"]]

    content_rec = content_rec.copy().sort_values("food_id")
    content_rec["content_score"] = np.arange(len(content_rec), 0, -1)

    collab_rec = collab_rec.copy().sort_values("food_id")
    collab_rec["collab_score"] = np.arange(len(collab_rec), 0, -1)

    merged = pd.concat([content_rec, collab_rec], ignore_index=True)
    merged = merged.groupby("food_name", as_index=False).first()
    merged["content_score"] = merged.get("content_score", 0)
    merged["collab_score"] = merged.get("collab_score", 0)
    merged["hybrid_score"] = alpha * merged["content_score"] + (1 - alpha) * merged["collab_score"]

    return merged.sort_values("hybrid_score", ascending=False).head(top_n)[
        ["food_id","food_name", "calories", "protein", "carbs", "fat","sugar",'sodium']
    ]

# ==============================
# 8. API Endpoint
# ==============================
@router.get("/recommend/{user_id}")
async def get_recommendation(user_id: int, top_n: int = 5):
    hybrid = hybrid_recommend(user_id, top_n=top_n)
    
    # ถ้า hybrid เป็น DataFrame
    if isinstance(hybrid, pd.DataFrame):
        hybrid = hybrid.fillna(0)
        return hybrid.to_dict(orient='records')
    else:
        # กรณี return list ของ dict
        return hybrid
