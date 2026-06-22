import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from urllib.parse import quote_plus
from datetime import datetime
from bson import ObjectId
from dotenv import load_dotenv
from run_model import generate_text

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-fallback-key")

client = MongoClient(os.environ.get("MONGODB_URI"))

db = client["tinystoriesgpt"]
stories_collection = db["stories"]
users_collection = db["users"]

# Test connection
try:
    client.admin.command("ping")
    print("✅ MongoDB connected successfully!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.name = user_data["name"]
        self.email = user_data["email"]

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({"_id": ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

def serialize(story):
    story["_id"] = str(story["_id"])
    return story

# ─── Pages ────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    return render_template("index.html", name=current_user.name)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.json
        existing = users_collection.find_one({"email": data["email"]})
        if existing:
            return jsonify({"error": "Email already registered"}), 400
        users_collection.insert_one({
            "name": data["name"],
            "email": data["email"],
            "password": generate_password_hash(data["password"]),
            "created_at": datetime.now()
        })
        return jsonify({"success": True})
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.json
        user_data = users_collection.find_one({"email": data["email"]})
        if not user_data or not check_password_hash(user_data["password"], data["password"]):
            return jsonify({"error": "Invalid email or password"}), 401
        login_user(User(user_data))
        return jsonify({"success": True})
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ─── API ──────────────────────────────────────────────

@app.route("/stories", methods=["GET"])
@login_required
def get_stories():
    stories = list(stories_collection.find({"user_id": current_user.id}, {"title": 1}))
    return jsonify([serialize(s) for s in stories])

@app.route("/stories/<story_id>", methods=["GET"])
@login_required
def get_story(story_id):
    story = stories_collection.find_one({"_id": ObjectId(story_id), "user_id": current_user.id})
    return jsonify(serialize(story))

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.json
    story_id = data.get("story_id")
    user_message = data.get("message")

    ai_response = generate_text(user_message, max_new_tokens=80)

    message_pair = [
        {"role": "user", "content": user_message},
        {"role": "ai", "content": ai_response}
    ]

    if story_id:
        stories_collection.update_one(
            {"_id": ObjectId(story_id), "user_id": current_user.id},
            {"$push": {"messages": {"$each": message_pair}}}
        )
    else:
        result = stories_collection.insert_one({
            "title": user_message[:40],
            "messages": message_pair,
            "user_id": current_user.id,
            "created_at": datetime.now()
        })
        story_id = str(result.inserted_id)

    return jsonify({"story_id": story_id, "response": ai_response})

@app.route("/edit-name", methods=["POST"])
@login_required
def edit_name():
    data = request.json
    users_collection.update_one(
        {"_id": ObjectId(current_user.id)},
        {"$set": {"name": data["name"]}}
    )
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")
