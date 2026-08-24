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

import os, sys, uuid, json, base64, hashlib, logging, requests, math, struct, random, urllib.parse
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
        "suno_cookie": settings.get("suno_cookie", config.SUNO_COOKIE),
    }


def ai_generate_lyrics(prompt: str, genre: str, mood: str, language: str, duration: str, model: str) -> dict:
    """Call OpenRouter chat completions to write song lyrics."""
    system_msg = (
        "You are a Grammy-winning professional songwriter and music producer. "
        "Generate complete, emotionally powerful song lyrics with multiple verses, "
        "a memorable chorus, and a bridge. Return ONLY valid JSON, with no other text."
    )
    user_msg = (
        f"Write a {duration} {genre} song with a {mood} mood in {language}.\n"
        f"Theme / inspiration: {prompt}\n\n"
        'Return JSON exactly like: {"title": "...", "lyrics": "...", "structure": "verse-chorus-verse-chorus-bridge-chorus"}'
    )
    
    json_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.85,
        "max_tokens": 1500,
    }
    
    # Only use response_format if not a free model to avoid API error
    if "free" not in model.lower():
        json_payload["response_format"] = {"type": "json_object"}
        
    try:
        resp = requests.post(
            f"{config.OPENROUTER_BASE}/chat/completions",
            headers=_or_headers(),
            json=json_payload,
            timeout=90,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        
        # Robust parsing to find JSON boundaries
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start:end+1])
            raise
    except Exception as e:
        logger.warning(f"Lyrics AI failed: {e}. Retrying without response_format...")
        if "response_format" in json_payload:
            del json_payload["response_format"]
            try:
                resp = requests.post(
                    f"{config.OPENROUTER_BASE}/chat/completions",
                    headers=_or_headers(),
                    json=json_payload,
                    timeout=90,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(raw[start:end+1])
                return json.loads(raw)
            except Exception as inner_e:
                logger.error(f"Fallback lyrics generation failed: {inner_e}")
                raise inner_e
        raise e


def ai_generate_cover(title: str, genre: str, mood: str, prompt: str, model: str) -> str | None:
    """Generate album cover art via Pollinations AI or OpenRouter images endpoint."""
    if model == "pollinations":
        encoded_prompt = urllib.parse.quote(f"album cover art, {genre}, {mood}, {title}, {prompt[:100]}")
        return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true"

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


# ── Procedural Audio Synthesizer ──────────────────────────────────────────────
BPM_MAP = {
    "Pop": 120,
    "Rock": 115,
    "Hip-Hop": 92,
    "Jazz": 85,
    "Electronic": 128,
    "Classical": 75,
    "R&B": 88,
    "Indie": 105,
    "Folk": 95,
    "Bollywood": 115
}

PROGRESSIONS = {
    "Happy": ["C", "G", "Am", "F"],
    "Sad": ["Am", "F", "C", "G"],
    "Romantic": ["C", "Am", "F", "G"],
    "Energetic": ["Am", "G", "F", "Em"],
    "Melancholic": ["Am", "Dm", "C", "G"],
    "Peaceful": ["C", "F", "C", "G"],
    "Dark": ["Am", "F", "Dm", "E"],
    "Uplifting": ["C", "G", "Am", "F"],
    "Nostalgic": ["Am", "G", "C", "F"]
}

def get_chord_notes(chord_name):
    chords = {
        "C": [261.63, 329.63, 392.00],       # C4, E4, G4
        "G": [196.00, 246.94, 293.66],       # G3, B3, D4
        "Am": [220.00, 261.63, 329.63],      # A3, C4, E4
        "F": [174.61, 220.00, 261.63],       # F3, A3, C4
        "Dm": [293.66, 349.23, 440.00],      # D4, F4, A4
        "E": [329.63, 415.30, 493.88],       # E4, G#4, B4
        "Em": [329.63, 392.00, 493.88],      # E4, G4, B4
    }
    return chords.get(chord_name, [261.63, 329.63, 392.00])

def get_bass_note(chord_name):
    bass = {
        "C": 65.41,
        "G": 98.00,
        "Am": 110.00,
        "F": 87.31,
        "Dm": 73.42,
        "E": 82.41,
        "Em": 82.41,
    }
    return bass.get(chord_name, 65.41)

def add_synth_note(mixed, start_sample, duration_samples, frequency, amplitude, wave_type, sample_rate):
    for i in range(duration_samples):
        idx = start_sample + i
        if idx >= len(mixed):
            break
        t = i / sample_rate
        if wave_type == 'sine':
            val = math.sin(2 * math.pi * frequency * t)
        elif wave_type == 'square':
            val = 1.0 if math.sin(2 * math.pi * frequency * t) >= 0 else -1.0
        elif wave_type == 'triangle':
            val = 2.0 * abs(2.0 * (t * frequency - math.floor(t * frequency + 0.5))) - 1.0
        elif wave_type == 'sawtooth':
            val = 2.0 * (t * frequency - math.floor(t * frequency + 0.5))
        else:
            val = 0.0
            
        attack = min(int(0.015 * sample_rate), duration_samples // 4)
        release = min(int(0.06 * sample_rate), duration_samples // 3)
        if i < attack:
            env = i / attack
        elif i > duration_samples - release:
            env = max(0.0, (duration_samples - i) / release)
        else:
            env = 1.0
            
        mixed[idx] += val * amplitude * env

def add_kick(mixed, start_sample, sample_rate):
    dur_sec = 0.12
    dur_samples = int(sample_rate * dur_sec)
    for i in range(dur_samples):
        idx = start_sample + i
        if idx >= len(mixed):
            break
        t = i / sample_rate
        freq = 150 - 110 * (t / dur_sec)
        val = math.sin(2 * math.pi * freq * t)
        decay = math.exp(-12 * t)
        mixed[idx] += val * 0.45 * decay

def add_snare(mixed, start_sample, sample_rate):
    dur_sec = 0.18
    dur_samples = int(sample_rate * dur_sec)
    for i in range(dur_samples):
        idx = start_sample + i
        if idx >= len(mixed):
            break
        t = i / sample_rate
        noise = random.uniform(-1, 1) * 0.7
        tone = math.sin(2 * math.pi * 180 * t) * 0.3
        decay = math.exp(-14 * t)
        mixed[idx] += (noise + tone) * 0.28 * decay

def add_hihat(mixed, start_sample, sample_rate):
    dur_sec = 0.04
    dur_samples = int(sample_rate * dur_sec)
    for i in range(dur_samples):
        idx = start_sample + i
        if idx >= len(mixed):
            break
        t = i / sample_rate
        noise = random.uniform(-1, 1)
        decay = math.exp(-60 * t)
        mixed[idx] += noise * 0.12 * decay

def normalize_and_pack(samples):
    max_val = max(abs(x) for x in samples) if samples else 0
    scale = 0.95 / max_val if max_val > 1e-5 else 1.0
    packed = bytearray()
    for s in samples:
        val = int(s * scale * 32767)
        val = max(-32768, min(32767, val))
        packed.extend(struct.pack('<h', val))
    return packed

def make_wav_header(num_samples, sample_rate, num_channels=1, bits_per_sample=16):
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = num_samples * block_align
    file_size = 36 + data_size
    header = struct.pack('<4sI4s', b'RIFF', file_size, b'WAVE')
    fmt_chunk = struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample)
    data_chunk = struct.pack('<4sI', b'data', data_size)
    return header + fmt_chunk + data_chunk

def generate_procedural_wav(genre: str, mood: str, title: str) -> bytes:
    sample_rate = 22050
    bpm = BPM_MAP.get(genre, 110)
    prog = PROGRESSIONS.get(mood, ["C", "G", "Am", "F"])
    measures = prog * 2
    steps_per_measure = 16
    total_steps = len(measures) * steps_per_measure
    step_duration = 60.0 / (bpm * 4)
    total_duration = step_duration * total_steps
    total_samples = int(sample_rate * total_duration)
    mixed = [0.0] * total_samples
    seed = int(hashlib.md5(title.encode()).hexdigest(), 16) % 1000000
    rng = random.Random(seed)
    for m_idx, chord in enumerate(measures):
        measure_start_step = m_idx * steps_per_measure
        measure_start_sample = int(measure_start_step * step_duration * sample_rate)
        chord_notes = get_chord_notes(chord)
        bass_note = get_bass_note(chord)
        for step in range(steps_per_measure):
            step_idx = measure_start_step + step
            step_start_sample = int(step_idx * step_duration * sample_rate)
            step_samples = int(step_duration * sample_rate)
            if genre in ["Electronic", "Pop", "Bollywood"]:
                if step in [0, 4, 8, 12]:
                    add_kick(mixed, step_start_sample, sample_rate)
                if step in [4, 12]:
                    add_snare(mixed, step_start_sample, sample_rate)
                if step in [2, 6, 10, 14]:
                    add_hihat(mixed, step_start_sample, sample_rate)
            elif genre in ["Hip-Hop", "R&B"]:
                if step in [0, 8, 10]:
                    add_kick(mixed, step_start_sample, sample_rate)
                if step in [4, 12]:
                    add_snare(mixed, step_start_sample, sample_rate)
                if step in [2, 6, 10, 14]:
                    add_hihat(mixed, step_start_sample, sample_rate)
            else:
                if step in [0, 8]:
                    add_kick(mixed, step_start_sample, sample_rate)
                if step in [4, 12]:
                    add_snare(mixed, step_start_sample, sample_rate)
                if step in [0, 2, 4, 6, 8, 10, 12, 14]:
                    add_hihat(mixed, step_start_sample, sample_rate)
            if step in [0, 4, 8, 12]:
                bass_dur = int(step_samples * 2.5)
                add_synth_note(mixed, step_start_sample, bass_dur, bass_note, 0.22, "triangle", sample_rate)
            if step in [0, 8]:
                pad_dur = int(step_samples * 6)
                for note in chord_notes:
                    add_synth_note(mixed, step_start_sample, pad_dur, note, 0.08, "sine", sample_rate)
            if step in [0, 3, 6, 8, 11, 14]:
                if rng.random() < 0.75:
                    melody_dur = int(step_samples * 1.5)
                    chord_note = rng.choice(chord_notes)
                    melody_note = chord_note * (2 if rng.random() < 0.6 else 1)
                    wave_type = "sine"
                    if genre == "Electronic":
                        wave_type = "sawtooth" if rng.random() < 0.3 else "sine"
                    elif genre in ["Rock", "Indie"]:
                        wave_type = "triangle"
                    add_synth_note(mixed, step_start_sample, melody_dur, melody_note, 0.15, wave_type, sample_rate)
    packed_data = normalize_and_pack(mixed)
    header = make_wav_header(len(mixed), sample_rate)
    return bytes(header + packed_data)


# ── Suno.com Music Generator ──────────────────────────────────────────────────
def generate_suno_song(prompt, genre, mood, lyrics, title, cookie_str, vocal_type="female"):
    import requests
    import time
    
    headers = {
        "Cookie": cookie_str,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Origin": "https://suno.com",
        "Referer": "https://suno.com/",
        "Content-Type": "application/json"
    }
    
    # ── Step 1: Clerk Token Exchange ──────────────────────────────────────────
    logger.info("Attempting Clerk token exchange for Suno...")
    try:
        resp = requests.get("https://clerk.suno.com/v1/client?_clerk_js_version=4.73.2", headers=headers, timeout=15)
        resp.raise_for_status()
        client_data = resp.json()
        
        response_obj = client_data.get("response", {})
        session_id = response_obj.get("last_active_session_id")
        if not session_id:
            sessions = response_obj.get("client", {}).get("sessions", [])
            if sessions:
                session_id = sessions[0].get("id")
                
        if not session_id:
            raise Exception("No active Suno session found. Make sure you are logged into suno.com and copied the full cookie.")
            
        logger.info(f"Active session found: {session_id}")
        
        token_url = f"https://clerk.suno.com/v1/client/sessions/{session_id}/tokens?_clerk_js_version=4.73.2"
        resp2 = requests.post(token_url, headers=headers, timeout=15)
        resp2.raise_for_status()
        token_data = resp2.json()
        jwt_token = token_data.get("jwt")
        if not jwt_token:
            jwt_token = token_data.get("response", {}).get("jwt")
            
        if not jwt_token:
            raise Exception("Failed to retrieve JWT token from Clerk.")
            
        logger.info("Successfully obtained Suno JWT token.")
    except Exception as e:
        logger.error(f"Clerk token exchange failed: {e}", exc_info=True)
        raise Exception(f"Suno Authentication failed: {e}")

    # ── Step 2: Custom Song Generation ───────────────────────────────────────
    logger.info("Sending generation request to Suno...")
    studio_headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Origin": "https://suno.com",
        "Referer": "https://suno.com/"
    }
    
    make_instrumental = (vocal_type == "instrumental")
    tags_list = [genre.lower(), mood.lower()]
    if vocal_type == "female":
        tags_list.append("female vocals")
    elif vocal_type == "male":
        tags_list.append("male vocals")
    elif vocal_type == "duet":
        tags_list.append("duet vocals")
        
    tags = ", ".join(tags_list)
    
    payload = {
        "prompt": lyrics[:3000] if lyrics else prompt,
        "tags": tags,
        "title": title or f"My {genre} Song",
        "make_instrumental": make_instrumental,
        "mv": "chirp-v3-5"
    }
    
    gen_resp = requests.post("https://studio-api.suno.ai/api/generate/v2/", headers=studio_headers, json=payload, timeout=20)
    gen_resp.raise_for_status()
    gen_data = gen_resp.json()
    
    clips = gen_data.get("clips", [])
    if not clips:
        raise Exception("Suno API did not return any audio clips. Check your credit balance.")
        
    clip_id = clips[0].get("id")
    logger.info(f"Suno generation started. Clip ID: {clip_id}")
    
    # ── Step 3: Poll until ready ─────────────────────────────────────────────
    poll_start = time.time()
    audio_url = None
    while time.time() - poll_start < 90:
        logger.info(f"Polling Suno clip {clip_id[:8]} status...")
        try:
            feed_resp = requests.get(f"https://studio-api.suno.ai/api/feed/?ids={clip_id}", headers=studio_headers, timeout=15)
            feed_resp.raise_for_status()
            feed_clips = feed_resp.json()
            if isinstance(feed_clips, list) and len(feed_clips) > 0:
                clip = feed_clips[0]
                status = clip.get("status")
                logger.info(f"Clip {clip_id[:8]} status: {status}")
                if status == "complete":
                    audio_url = clip.get("audio_url")
                    break
                elif status == "error":
                    raise Exception("Suno generation failed with error status.")
            time.sleep(4)
        except Exception as poll_e:
            logger.warning(f"Error polling clip: {poll_e}")
            time.sleep(4)
            
    if not audio_url:
        raise Exception("Suno song generation timed out after 90 seconds.")
        
    logger.info(f"Suno clip is ready! Downloading from: {audio_url}")
    
    # ── Step 4: Download and return bytes ────────────────────────────────────
    audio_get = requests.get(audio_url, timeout=30)
    audio_get.raise_for_status()
    return audio_get.content


# ── Hugging Face MusicGen Generator ───────────────────────────────────────────
def generate_musicgen_audio(prompt: str, genre: str, mood: str, duration: str = "15 seconds") -> bytes:
    from gradio_client import Client
    logger.info("Initializing Gradio Client for Hugging Face Space: facebook/MusicGen...")
    try:
        # Convert duration to seconds, default to 15s, cap at 30s
        duration_sec = 15
        if duration:
            try:
                num = int(''.join(filter(str.isdigit, duration)))
                if "min" in duration:
                    duration_sec = num * 60
                else:
                    duration_sec = num
            except Exception:
                duration_sec = 15
        duration_sec = min(30, max(5, duration_sec))

        client = Client("facebook/MusicGen")
        full_prompt = f"{genre} music, {mood} mood, {prompt}"
        
        logger.info(f"Submitting prediction to MusicGen: '{full_prompt}' for {duration_sec}s...")
        result = client.predict(
            text=full_prompt,
            melody=None,
            model="melody",
            duration=duration_sec,
            topk=250,
            topp=0.0,
            temperature=1.0,
            cfg_coef=3.0,
            api_name="/predict"
        )
        
        if result and os.path.exists(result):
            logger.info(f"MusicGen generated audio successfully: {result}")
            with open(result, "rb") as f:
                return f.read()
        raise Exception("MusicGen did not return a valid audio file.")
    except Exception as e:
        logger.error(f"MusicGen generation failed: {e}", exc_info=True)
        raise e


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


# ── Audio Serving ─────────────────────────────────────────────────────────────
@app.route("/api/songs/<song_id>/audio", methods=["GET"])
def serve_song_audio(song_id):
    """Serve the song audio from MongoDB decodable from base64."""
    try:
        db = get_db()
        song = db.songs.find_one({"song_id": song_id, "is_deleted": {"$ne": True}})
        if not song:
            return jsonify({"error": "Song not found"}), 404
            
        audio_b64 = song.get("audio_b64")
        if not audio_b64:
            return jsonify({"error": "Audio data not found for this song"}), 404
            
        # Extract binary data from base64 string
        if audio_b64.startswith("data:audio/"):
            parts = audio_b64.split(",", 1)
            if len(parts) == 2:
                mime_type = parts[0].split(";")[0].replace("data:", "")
                b64_data = parts[1]
            else:
                mime_type = "audio/wav"
                b64_data = audio_b64
        else:
            mime_type = "audio/wav"
            b64_data = audio_b64
            
        audio_bytes = base64.b64decode(b64_data)
        
        response = app.response_class(audio_bytes, mimetype=mime_type)
        
        # If download parameter is provided, trigger attachment download
        if request.args.get("download") == "1":
            title = song.get("title", "song").replace('"', '\\"')
            ext = "wav" if "wav" in mime_type else "mp3"
            response.headers["Content-Disposition"] = f'attachment; filename="{title}.{ext}"'
            
        response.headers["Accept-Ranges"] = "bytes"
        return response
    except Exception as e:
        logger.error(f"Serve audio error for song {song_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to serve audio", "detail": str(e)}), 500


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
        serialized_songs = []
        for s in songs:
            s_doc = serialize_doc(s)
            if s_doc:
                s_doc["audio_url"] = f"/api/songs/{s_doc['song_id']}/audio"
                serialized_songs.append(s_doc)
        return jsonify({"songs": serialized_songs, "total": total, "page": page})
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
            "suno_cookie": config.SUNO_COOKIE,
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
        vocal_type = data.get("vocal_type", "female")
        face_id = request.user["face_id"]

        if not prompt:
            return jsonify({"error": "A prompt is required"}), 400

        db = get_db()
        models = _get_live_models(db)
        song_id = str(uuid.uuid4())

        logger.info(f"Generating song for {face_id[:8]}…: '{prompt[:60]}' [Singer: {vocal_type}]")

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
        audio_result = {"audio_url": None, "audio_b64": None}
        audio_bytes = None
        suno_cookie = models.get("suno_cookie")
        
        # Tier 1: Suno.com (if cookie is configured)
        if suno_cookie:
            logger.info("Suno cookie configured. Attempting Suno.com generation...")
            try:
                audio_bytes = generate_suno_song(prompt, genre, mood, lyrics, title, suno_cookie, vocal_type)
                logger.info("Successfully generated song via Suno!")
            except Exception as e:
                logger.warning(f"Suno generation failed: {e}. Falling back to Hugging Face MusicGen...")
        
        # Tier 2: Hugging Face MusicGen
        if not audio_bytes:
            logger.info("Attempting Hugging Face MusicGen generation...")
            try:
                musicgen_prompt = prompt
                if vocal_type == "instrumental":
                    musicgen_prompt += ", instrumental"
                elif vocal_type == "female":
                    musicgen_prompt += ", female singer voice"
                elif vocal_type == "male":
                    musicgen_prompt += ", male singer voice"
                elif vocal_type == "duet":
                    musicgen_prompt += ", duet male female singer voices"
                audio_bytes = generate_musicgen_audio(musicgen_prompt, genre, mood, duration)
                logger.info("Successfully generated audio via Hugging Face MusicGen!")
            except Exception as e:
                logger.warning(f"MusicGen generation failed: {e}. Falling back to procedural synthesizer...")
                
        # Tier 3: Local Procedural Synthesizer (guaranteed fallback)
        if not audio_bytes:
            logger.info("Generating procedural audio...")
            try:
                audio_bytes = generate_procedural_wav(genre, mood, title)
            except Exception as e:
                logger.error(f"Procedural audio generation failed: {e}", exc_info=True)
                
        if audio_bytes:
            is_mp3 = audio_bytes.startswith(b"ID3") or b"lavf" in audio_bytes[:100].lower() or audio_bytes.startswith(b"\xff\xfb")
            mime = "audio/mpeg" if is_mp3 else "audio/wav"
            audio_result["audio_b64"] = f"data:{mime};base64," + base64.b64encode(audio_bytes).decode("utf-8")
            audio_result["audio_url"] = f"/api/songs/{song_id}/audio"

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
            "vocal_type": vocal_type,
            "cover_url": cover_url,
            "audio_url": f"/api/songs/{song_id}/audio" if audio_result.get("audio_b64") else audio_result.get("audio_url"),
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
        serialized_songs = []
        for s in songs:
            s_doc = serialize_doc(s)
            if s_doc:
                s_doc["audio_url"] = f"/api/songs/{s_doc['song_id']}/audio"
                serialized_songs.append(s_doc)

        return jsonify({
            "songs": serialized_songs,
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
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
