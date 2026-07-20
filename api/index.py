"""
DHUN AI — Main Flask Application
Vercel serverless entry-point: api/index.py

Routes
------
GET  /                          Serve SPA shell
GET  /api/health                Health check

POST /api/auth/register         Face registration
POST /api/auth/login            Face login (with anti-spoofing embedding)
GET  /api/auth/verify           Token validation

POST /api/admin/login           Admin PIN login
GET  /api/admin/analytics       Dashboard analytics
GET  /api/admin/users           List all users
DELETE /api/admin/users/<id>    Delete user + their songs
GET  /api/admin/songs           List all songs (paginated)
DELETE /api/admin/songs/<id>    Hard-delete song
GET|POST /api/admin/settings    Read/update app settings

POST /api/songs/generate        Full AI generation pipeline
GET  /api/songs                 User's song library (search/filter/page)
DELETE /api/songs/<id>          Soft-delete song
POST /api/songs/<id>/play       Increment play count
"""

import os, sys, uuid, json, base64, hashlib, logging, requests
from datetime import datetime, timedelta
from functools import wraps

# ── Path fix so Vercel can find templates/static from api/ sub-dir ──────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from cryptography.fernet import Fernet
from bson import ObjectId
import jwt as pyjwt
import numpy as np

from config import config

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("dhun_ai")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
)
app.secret_key = config.SECRET_KEY
CORS(app, origins="*", supports_credentials=True)

# ── Encryption ────────────────────────────────────────────────────────────────
_fernet = Fernet(config.get_fernet_key())


def encrypt_embedding(embedding: list) -> str:
    payload = json.dumps(embedding).encode()
    return _fernet.encrypt(payload).decode()


def decrypt_embedding(encrypted: str) -> list:
    payload = _fernet.decrypt(encrypted.encode())
    return json.loads(payload)


def euclidean_distance(a: list, b: list) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a_arr - b_arr))


# ── MongoDB ───────────────────────────────────────────────────────────────────
_mongo_client: MongoClient | None = None


def get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(
            config.MONGO_URI, serverSelectionTimeoutMS=8000, connectTimeoutMS=8000
        )
    return _mongo_client[config.DB_NAME]


def serialize_doc(doc: dict) -> dict:
    """Make a MongoDB document JSON-serialisable."""
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    for key in ("created_at", "last_login", "timestamp"):
        if key in doc and hasattr(doc[key], "isoformat"):
            doc[key] = doc[key].isoformat()
    return doc


# ── JWT helpers ───────────────────────────────────────────────────────────────
def create_token(face_id: str, is_admin: bool = False) -> str:
    payload = {
        "face_id": face_id,
        "is_admin": is_admin,
        "exp": datetime.utcnow() + timedelta(days=config.TOKEN_EXPIRE_DAYS),
        "iat": datetime.utcnow(),
    }
    return pyjwt.encode(payload, config.JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> dict:
    return pyjwt.decode(token, config.JWT_SECRET, algorithms=["HS256"])


def _get_token_from_request() -> str:
    auth_header = request.headers.get("Authorization", "")
    return auth_header.replace("Bearer ", "").strip()


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            token = _get_token_from_request()
            if not token:
                return jsonify({"error": "Authentication required"}), 401
            request.user = verify_token(token)
        except pyjwt.ExpiredSignatureError:
            return jsonify({"error": "Session expired — please log in again"}), 401
        except pyjwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)

    return decorated


def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            token = _get_token_from_request()
            if not token:
                return jsonify({"error": "Admin authentication required"}), 401
            payload = verify_token(token)
            if not payload.get("is_admin"):
                return jsonify({"error": "Administrator access only"}), 403
            request.user = payload
        except Exception:
            return jsonify({"error": "Invalid or expired admin token"}), 401
        return f(*args, **kwargs)

    return decorated


# ── OpenRouter helpers ────────────────────────────────────────────────────────
def _or_headers() -> dict:
    return {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": config.OPENROUTER_REFERER,
        "X-Title": config.OPENROUTER_TITLE,
    }


def _get_live_models(db) -> dict:
    """Fetch model config from DB (admin-overridable) with fallback to config."""
    settings = db.settings.find_one({"_id": "app_settings"}) or {}
    return {
        "lyrics": settings.get("lyrics_model", config.LYRICS_MODEL),
        "cover": settings.get("cover_model", config.COVER_MODEL),
        "audio": settings.get("audio_model", config.AUDIO_MODEL),
    }


def ai_generate_lyrics(prompt: str, genre: str, mood: str, language: str, duration: str, model: str) -> dict:
    """Call OpenRouter chat completions to write song lyrics."""
    system_msg = (
        "You are a Grammy-winning professional songwriter and music producer. "
        "Generate complete, emotionally powerful song lyrics with multiple verses, "
        "a memorable chorus, and a bridge. Return ONLY valid JSON."
    )
    user_msg = (
        f"Write a {duration} {genre} song with a {mood} mood in {language}.\n"
        f"Theme / inspiration: {prompt}\n\n"
        'Return JSON exactly like: {"title": "...", "lyrics": "...", "structure": "verse-chorus-verse-chorus-bridge-chorus"}'
    )
    resp = requests.post(
        f"{config.OPENROUTER_BASE}/chat/completions",
        headers=_or_headers(),
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.85,
            "max_tokens": 1500,
        },
        timeout=90,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return json.loads(raw)


def ai_generate_cover(title: str, genre: str, mood: str, prompt: str, model: str) -> str | None:
    """Generate album cover art via OpenRouter images endpoint."""
    image_prompt = (
        f"Professional album cover art for a {genre} song titled '{title}'. "
        f"Mood: {mood}. Inspired by: {prompt[:120]}. "
        "Ultra-HD, artistic, cinematic lighting, music industry quality, "
        "vibrant but elegant colors, abstract or symbolic imagery, absolutely NO text or letters."
    )
    try:
        resp = requests.post(
            f"{config.OPENROUTER_BASE}/images/generations",
            headers=_or_headers(),
            json={
                "model": model,
                "prompt": image_prompt,
                "n": 1,
                "size": "512x512",
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and data["data"]:
            item = data["data"][0]
            if "url" in item:
                return item["url"]
            if "b64_json" in item:
                return f"data:image/png;base64,{item['b64_json']}"
    except Exception as e:
        logger.warning(f"Cover generation failed ({model}): {e}")

    # Fallback: vibrant placeholder via picsum with seed
    seed = hashlib.md5(title.encode()).hexdigest()[:8]
    return f"https://picsum.photos/seed/{seed}/512/512"


def ai_generate_audio(title: str, lyrics: str, genre: str, mood: str, model: str) -> dict:
    """Generate audio via OpenRouter Lyria model."""
    result = {"audio_url": None, "audio_b64": None}
    try:
        resp = requests.post(
            f"{config.OPENROUTER_BASE}/chat/completions",
            headers=_or_headers(),
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Create a {genre} song with {mood} mood titled '{title}'.\n"
                            f"Use these lyrics:\n{lyrics[:800]}"
                        ),
                    }
                ],
                "max_tokens": 2048,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            # Lyria may embed audio URL or base64 in content
            if content.startswith("http"):
                result["audio_url"] = content.strip()
            elif content.startswith("data:audio"):
                result["audio_b64"] = content.strip()
            else:
                # Some models return structured data
                if "audio" in data:
                    audio_data = data["audio"]
                    result["audio_url"] = audio_data.get("url")
                    result["audio_b64"] = audio_data.get("data")
    except Exception as e:
        logger.warning(f"Audio generation failed ({model}): {e}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════════════════

# ── Static / SPA ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)


@app.route("/sw.js")
def service_worker():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "sw.js")


@app.route("/manifest.json")
def manifest():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "manifest.json")


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    try:
        get_db().command("ping")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"
    return jsonify({"status": "ok", "app": "DHUN AI", "version": "1.0.0", "db": db_status})


# ═════════════════════════════════════════════════════════════════════════════
#  AUTH
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/register", methods=["POST"])
def register():
    """Register a new user via face embedding extracted by face-api.js."""
    try:
        data = request.get_json(force=True)
        embedding = data.get("embedding", [])
        name = str(data.get("name", "")).strip()
        age = data.get("age")
        terms_accepted = data.get("terms_accepted") is True

        if not isinstance(embedding, list) or len(embedding) != 128:
            return jsonify({"error": "Invalid face embedding — expected 128-dimensional array"}), 400

        if len(name) < 2 or len(name) > 80:
            return jsonify({"error": "Please provide a valid full name"}), 400
        if isinstance(age, bool) or not isinstance(age, int) or age < 18 or age > 120:
            return jsonify({"error": "You must be 18 or older to register"}), 400
        if not terms_accepted:
            return jsonify({"error": "Terms acceptance is required to register"}), 400

        db = get_db()

        # Check if face already exists (prevents duplicate registrations)
        all_users = list(db.users.find({}, {"face_embedding": 1, "face_id": 1, "name": 1}))
        for user in all_users:
            try:
                stored = decrypt_embedding(user["face_embedding"])
                dist = euclidean_distance(embedding, stored)
                if dist < config.FACE_MATCH_THRESHOLD:
                    # Already registered — auto-login instead
                    db.users.update_one(
                        {"face_id": user["face_id"]},
                        {"$set": {"last_login": datetime.utcnow()}},
                    )
                    token = create_token(user["face_id"])
                    logger.info(f"Re-registration detected, auto-login: {user['face_id']}")
                    return jsonify({"already_registered": True, "token": token, "face_id": user["face_id"], "display_name": user.get("name", "there")}), 200
            except Exception:
                continue

        # New user — encrypt & store
        face_id = str(uuid.uuid4())
        user_doc = {
            "face_id": face_id,
            "face_embedding": encrypt_embedding(embedding),
            "name": name,
            "age": age,
            "terms_accepted_at": datetime.utcnow(),
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow(),
            "song_count": 0,
            "is_active": True,
        }
        db.users.insert_one(user_doc)
        db.analytics.insert_one({"event": "user_registered", "face_id": face_id, "timestamp": datetime.utcnow()})

        token = create_token(face_id)
        logger.info(f"New user registered: {face_id}")
        return jsonify({"success": True, "token": token, "face_id": face_id, "display_name": name}), 201

    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return jsonify({"error": "Registration failed", "detail": str(e)}), 500


@app.route("/api/auth/login", methods=["POST"])
def login():
    """Authenticate a returning user by comparing face embeddings."""
    try:
        data = request.get_json(force=True)
        embedding = data.get("embedding", [])

        if not isinstance(embedding, list) or len(embedding) != 128:
            return jsonify({"error": "Invalid face embedding"}), 400

        db = get_db()
        all_users = list(db.users.find({"is_active": {"$ne": False}}, {"face_embedding": 1, "face_id": 1, "name": 1}))

        best_user = None
        best_dist = float("inf")

        for user in all_users:
            try:
                stored = decrypt_embedding(user["face_embedding"])
                dist = euclidean_distance(embedding, stored)
                if dist < best_dist:
                    best_dist = dist
                    best_user = user
            except Exception:
                continue

        if best_user and best_dist < config.FACE_MATCH_THRESHOLD:
            db.users.update_one(
                {"face_id": best_user["face_id"]},
                {"$set": {"last_login": datetime.utcnow()}},
            )
            db.analytics.insert_one({
                "event": "user_login",
                "face_id": best_user["face_id"],
                "confidence": round(1.0 - best_dist, 4),
                "timestamp": datetime.utcnow(),
            })
            token = create_token(best_user["face_id"])
            logger.info(f"User logged in: {best_user['face_id']} (dist={best_dist:.3f})")
            return jsonify({"success": True, "token": token, "face_id": best_user["face_id"], "display_name": best_user.get("name", "there")}), 200

        # Failed attempt log
        db.analytics.insert_one({
            "event": "login_failed",
            "best_distance": best_dist if best_dist != float("inf") else -1,
            "timestamp": datetime.utcnow(),
        })
        return jsonify({"error": "Face not recognised. Please register first."}), 401

    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({"error": "Login failed", "detail": str(e)}), 500


@app.route("/api/auth/verify", methods=["GET"])
@require_auth
def verify_auth():
    user = get_db().users.find_one({"face_id": request.user["face_id"]}, {"name": 1}) or {}
    return jsonify({"valid": True, "face_id": request.user["face_id"], "display_name": user.get("name", "")})


# ═════════════════════════════════════════════════════════════════════════════
#  ADMIN
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json(force=True)
    if str(data.get("pin", "")).strip() == config.ADMIN_PIN:
        token = create_token("admin", is_admin=True)
        logger.info("Admin panel access granted")
        return jsonify({"success": True, "token": token})
    return jsonify({"error": "Invalid PIN"}), 401


@app.route("/api/admin/analytics", methods=["GET"])
@require_admin
def admin_analytics():
    try:
        db = get_db()
        total_users = db.users.count_documents({})
        total_songs = db.songs.count_documents({"is_deleted": {"$ne": True}})

        play_agg = list(db.songs.aggregate([{"$group": {"_id": None, "total": {"$sum": "$play_count"}}}]))
        total_plays = play_agg[0]["total"] if play_agg else 0

        genre_stats = list(db.songs.aggregate([
            {"$match": {"is_deleted": {"$ne": True}}},
            {"$group": {"_id": "$genre", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]))

        mood_stats = list(db.songs.aggregate([
            {"$match": {"is_deleted": {"$ne": True}}},
            {"$group": {"_id": "$mood", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]))

        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        daily_songs = list(db.analytics.aggregate([
            {"$match": {"event": "song_generated", "timestamp": {"$gte": seven_days_ago}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "count": {"$sum": 1},
            }},
            {"$sort": {"_id": 1}},
        ]))

        recent_events = list(db.analytics.find({}).sort("timestamp", DESCENDING).limit(30))

        return jsonify({
            "total_users": total_users,
            "total_songs": total_songs,
            "total_plays": total_plays,
            "genre_stats": genre_stats,
            "mood_stats": mood_stats,
            "daily_songs": daily_songs,
            "recent_events": [serialize_doc(e) for e in recent_events],
        })
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/users", methods=["GET"])
@require_admin
def admin_list_users():
    try:
        db = get_db()
        users = list(db.users.find({}, {"face_embedding": 0}).sort("created_at", DESCENDING))
        return jsonify({"users": [serialize_doc(u) for u in users]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/users/<face_id>", methods=["DELETE"])
@require_admin
def admin_delete_user(face_id):
    try:
        db = get_db()
        db.users.delete_one({"face_id": face_id})
        db.songs.update_many({"user_face_id": face_id}, {"$set": {"is_deleted": True}})
        db.analytics.insert_one({"event": "admin_deleted_user", "face_id": face_id, "timestamp": datetime.utcnow()})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/songs", methods=["GET"])
@require_admin
def admin_list_songs():
    try:
        db = get_db()
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 20))
        query = {"is_deleted": {"$ne": True}}
        total = db.songs.count_documents(query)
        songs = list(
            db.songs.find(query, {"audio_b64": 0})
            .sort("created_at", DESCENDING)
            .skip((page - 1) * limit)
            .limit(limit)
        )
        return jsonify({"songs": [serialize_doc(s) for s in songs], "total": total, "page": page})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/songs/<song_id>", methods=["DELETE"])
@require_admin
def admin_delete_song(song_id):
    try:
        db = get_db()
        db.songs.delete_one({"song_id": song_id})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/settings", methods=["GET", "POST"])
@require_admin
def admin_settings():
    try:
        db = get_db()
        default_settings = {
            "_id": "app_settings",
            "lyrics_model": config.LYRICS_MODEL,
            "cover_model": config.COVER_MODEL,
            "audio_model": config.AUDIO_MODEL,
            "face_threshold": config.FACE_MATCH_THRESHOLD,
            "max_songs_per_user": 100,
            "app_name": "DHUN AI",
            "maintenance_mode": False,
        }
        if request.method == "GET":
            settings = db.settings.find_one({"_id": "app_settings"}) or default_settings
            settings["_id"] = str(settings["_id"])
            return jsonify(settings)
        else:
            data = request.get_json(force=True)
            data.pop("_id", None)  # Don't overwrite the ID field
            db.settings.update_one({"_id": "app_settings"}, {"$set": data}, upsert=True)
            return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
#  SONGS
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/songs/generate", methods=["POST"])
@require_auth
def generate_song():
    """Full AI generation pipeline: lyrics → cover → audio → save."""
    try:
        data = request.get_json(force=True)
        prompt = (data.get("prompt") or "").strip()
        genre = data.get("genre", "Pop")
        mood = data.get("mood", "Happy")
        language = data.get("language", "English")
        duration = data.get("duration", "3 minutes")
        face_id = request.user["face_id"]

        if not prompt:
            return jsonify({"error": "A prompt is required"}), 400

        db = get_db()
        models = _get_live_models(db)
        song_id = str(uuid.uuid4())

        logger.info(f"Generating song for {face_id[:8]}…: '{prompt[:60]}'")

        # ── Step 1: Lyrics ────────────────────────────────────────────
        try:
            lr = ai_generate_lyrics(prompt, genre, mood, language, duration, models["lyrics"])
            title = lr.get("title") or f"My {genre} Song"
            lyrics = lr.get("lyrics") or ""
            structure = lr.get("structure", "verse-chorus-verse-chorus-bridge-chorus")
        except Exception as e:
            logger.warning(f"Lyrics AI failed: {e}. Using fallback.")
            title = f"My {genre} Song"
            lyrics = (
                f"[Verse 1]\n{prompt}\n\n"
                "[Chorus]\nThis is my song\nA melody for you\n\n"
                "[Verse 2]\nThe music plays on\nThrough the night and day\n\n"
                "[Bridge]\nEvery note I play\nIs a word I cannot say\n\n"
                "[Chorus]\nThis is my song\nA melody for you"
            )
            structure = "verse-chorus-verse-chorus-bridge-chorus"

        # ── Step 2: Cover art ─────────────────────────────────────────
        cover_url = ai_generate_cover(title, genre, mood, prompt, models["cover"])

        # ── Step 3: Audio ─────────────────────────────────────────────
        audio_result = ai_generate_audio(title, lyrics, genre, mood, models["audio"])

        # ── Save to MongoDB ───────────────────────────────────────────
        song_doc = {
            "song_id": song_id,
            "title": title,
            "prompt": prompt,
            "lyrics": lyrics,
            "structure": structure,
            "genre": genre,
            "mood": mood,
            "language": language,
            "duration": duration,
            "cover_url": cover_url,
            "audio_url": audio_result.get("audio_url"),
            "audio_b64": audio_result.get("audio_b64"),  # stored but not returned in list
            "user_face_id": face_id,
            "created_at": datetime.utcnow(),
            "play_count": 0,
            "is_deleted": False,
        }
        db.songs.insert_one(song_doc)

        # Update user song count
        db.users.update_one({"face_id": face_id}, {"$inc": {"song_count": 1}})

        db.analytics.insert_one({
            "event": "song_generated",
            "song_id": song_id,
            "face_id": face_id,
            "genre": genre,
            "mood": mood,
            "model_lyrics": models["lyrics"],
            "model_cover": models["cover"],
            "model_audio": models["audio"],
            "timestamp": datetime.utcnow(),
        })

        # Return sanitised doc (omit raw base64 audio from response to save bandwidth)
        song_doc.pop("audio_b64", None)
        song_doc["_id"] = str(song_doc["_id"]) if "_id" in song_doc else None
        song_doc["created_at"] = song_doc["created_at"].isoformat()

        logger.info(f"Song generated: '{title}' ({song_id[:8]})")
        return jsonify({"success": True, "song": song_doc}), 201

    except Exception as e:
        logger.error(f"Song generation error: {e}", exc_info=True)
        return jsonify({"error": "Generation failed", "detail": str(e)}), 500


@app.route("/api/songs", methods=["GET"])
@require_auth
def get_user_songs():
    """Return paginated, filtered song library for the authenticated user."""
    try:
        db = get_db()
        face_id = request.user["face_id"]

        search = request.args.get("search", "").strip()
        genre = request.args.get("genre", "").strip()
        mood = request.args.get("mood", "").strip()
        page = max(1, int(request.args.get("page", 1)))
        limit = min(50, int(request.args.get("limit", 20)))

        query: dict = {"user_face_id": face_id, "is_deleted": {"$ne": True}}
        if search:
            query["$or"] = [
                {"title": {"$regex": search, "$options": "i"}},
                {"prompt": {"$regex": search, "$options": "i"}},
                {"genre": {"$regex": search, "$options": "i"}},
                {"mood": {"$regex": search, "$options": "i"}},
            ]
        if genre:
            query["genre"] = genre
        if mood:
            query["mood"] = mood

        total = db.songs.count_documents(query)
        songs = list(
            db.songs.find(query, {"audio_b64": 0})
            .sort("created_at", DESCENDING)
            .skip((page - 1) * limit)
            .limit(limit)
        )

        return jsonify({
            "songs": [serialize_doc(s) for s in songs],
            "total": total,
            "page": page,
            "pages": max(1, (total + limit - 1) // limit),
        })
    except Exception as e:
        logger.error(f"Get songs error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/songs/<song_id>", methods=["DELETE"])
@require_auth
def delete_song(song_id):
    try:
        db = get_db()
        face_id = request.user["face_id"]
        result = db.songs.update_one(
            {"song_id": song_id, "user_face_id": face_id},
            {"$set": {"is_deleted": True}},
        )
        if result.modified_count == 0:
            return jsonify({"error": "Song not found or unauthorised"}), 404
        db.users.update_one({"face_id": face_id}, {"$inc": {"song_count": -1}})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/songs/<song_id>/play", methods=["POST"])
@require_auth
def increment_play(song_id):
    try:
        db = get_db()
        db.songs.update_one({"song_id": song_id}, {"$inc": {"play_count": 1}})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Dev runner (not used on Vercel) ──────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
