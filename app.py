from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    url_for,
    flash
)
import os
import sqlite3
from datetime import datetime
import numpy as np

from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

app.secret_key = "waste_ai_secret_key_2026"

# ---------------- LOAD MODEL ----------------
from tensorflow.keras.applications.resnet50 import preprocess_input

model = load_model(
    "best_model.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

# ---------------- CLASS NAMES ----------------
# ⚠️ MUST MATCH TRAINING ORDER EXACTLY
classes = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
# ---------------- DATABASE INIT ----------------
def init_db():
    conn = sqlite3.connect("database.db")
    conn.close()

init_db()

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- HOME ----------------
@app.route('/')
def home():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("index.html")

# ---------------- UPLOAD ----------------
@app.route('/upload', methods=['POST'])
def upload():
    if "user_id" not in session:
        return redirect(url_for("login"))
    file = request.files['image']

    if file.filename == '':
        return redirect('/')

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    return render_template('preview.html', filename=file.filename)

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    if "user_id" not in session:
        return redirect(url_for("login"))
    filename = request.form['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # -------- IMAGE PREPROCESSING (FIXED) --------
    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224))

    img = np.array(img)
    img = preprocess_input(img)   # 🔥 CRITICAL FIX
    img = np.expand_dims(img, axis=0)

    # -------- PREDICTION --------
    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    confidence_value = np.max(prediction)

    category = classes[class_index]
    confidence = str(round(confidence_value * 100, 2)) + "%"

    # -------- SAVE TO DATABASE --------
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # Get category ID
    cursor.execute(
        "SELECT id FROM categories WHERE category_name=?",
        (category,)
    )
    category_id = cursor.fetchone()[0]

    # Save prediction
    cursor.execute("""
        INSERT INTO predictions
        (user_id, category_id, filename, confidence, time)
        VALUES (?, ?, ?, ?, ?)
    """,
                   (
                       session["user_id"],
                       category_id,
                       filename,
                       confidence,
                       datetime.now().strftime("%H:%M")
                   ))
    prediction_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        filename=filename,
        category=category,
        confidence=confidence,
        prediction_id=prediction_id
    )
#---------------FEEDBACK-------------------
@app.route("/feedback", methods=["POST"])
def feedback():

    if "user_id" not in session:
        return redirect(url_for("login"))

    prediction_id = request.form["prediction_id"]
    rating = request.form["rating"]
    comments = request.form["comments"]

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO feedback
    (prediction_id, rating, comments)
    VALUES (?, ?, ?)
    """,
    (
        prediction_id,
        rating,
        comments
    ))

    conn.commit()
    conn.close()

    flash("Feedback submitted successfully!")

    return redirect(url_for("dashboard"))

# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET','POST'])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE username=?",
            (username,)
        )

        user = cursor.fetchone()

        conn.close()

        if user and user["password"] == password:

            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]

            return redirect(url_for("home"))

        flash("Invalid Username or Password")

    return render_template("login.html")

#------------------LOGOUT-------------------
@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))

# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("""
    SELECT
        predictions.filename,
        categories.category_name,
        predictions.confidence,
        predictions.time,
        feedback.rating,
        feedback.comments
    FROM predictions
    JOIN categories
    ON predictions.category_id = categories.id

    LEFT JOIN feedback
    ON predictions.id = feedback.prediction_id
    """)
    rows = cursor.fetchall()
    cursor.execute("SELECT category_name FROM categories")
    categories = [row[0] for row in cursor.fetchall()]
    conn.close()
#-----------------------HISTORY---------------------
    history = []

    for row in rows:
        history.append({

            "filename": row[0],
            "category": row[1],
            "confidence": row[2],
            "time": row[3],
            "rating": row[4],
            "comments": row[5]

        })

    counts = {}

    for cat in categories:
        counts[cat] = sum(
            1 for r in history
            if r["category"] == cat
        )

    data = {
        "total": len(rows),
        "counts": counts
    }

    return render_template("dashboard.html", data=data, history=history)

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)