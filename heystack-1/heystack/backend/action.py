from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os, re, random
from typing import Dict, Any, List
import json
# Simple retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

user_state = {}


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Al-Atrash — Oud Chatbot (Enhanced Memory + Context)")

# --- Chat memory persistence ---
MEMORY_FILE = os.path.join(APP_ROOT, "chat_memory.json")

# Load past memory if exists
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        try:
            SESSIONS = json.load(f)
        except json.JSONDecodeError:
            SESSIONS = {}  # file was empty or broken — start fresh
else:
    SESSIONS = {}


def save_sessions():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(SESSIONS, f, indent=2)

import atexit
atexit.register(save_sessions)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve static frontend ---
app.mount("/static", StaticFiles(directory=os.path.join(APP_ROOT, "..", "frontend")), name="static")

# --- Memory storage ---
# SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session(sender_id: str) -> Dict[str, Any]:
    # Added last_topic to remember when Farid was discussed
    return SESSIONS.setdefault(sender_id, {"user_name": None,
                                           "awaiting_name": True,
                                           "learning_topic": None,
                                           "last_topic": None})

# --- Knowledge Base setup ---
KB_PATH = os.path.join(APP_ROOT, "data", "oud_knowledge.txt")
if os.path.exists(KB_PATH):
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_text = f.read().strip()
else:
    kb_text = ""

PARAGRAPHS = [p.strip() for p in kb_text.split("\n\n") if p.strip()]
vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(PARAGRAPHS) if PARAGRAPHS else None

def retrieve_best_answer(query: str, top_k: int = 2) -> List[str]:
    if tfidf is None:
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return [PARAGRAPHS[i] for i in top_idx if sims[i] > 0.05]

# --- Keyword Knowledge Map ---
SONG_VIDEO_MAP = {
    "song1": {
        "title": "Noura Ya Noura - An Easy Oud Song by Farid Al-Atrash",
        "url": "https://www.youtube.com/embed/jhVzW8jbhDE?list=RDjhVzW8jbhDE&start_radio=1"
    },
    "song2": {
        "title": "Learn Leila by Farid Al-Atrash on Oud - Easy Oud Songs",
        "url": "https://www.youtube.com/embed/rysHoKWqGAs?list=RDrysHoKWqGAs&start_radio=1"
    }
}

KEYWORD_DATA_MAP = {
    "farid": {
        "aliases": ["al-atrash","al atrash", "alatrash", "the musician", "the oud player"],
        "image": "static/farid-al-atrash.jpeg",
        "facts": [
            "Farid Al-Atrash was known as 'the King of the Oud'.",
            "Syrian-Egyptian singer, actor, and oud master",
            "He moved to Egypt as a child and became one of the most influential figures in Arabic music."
        ],
    },

    "oud_picture": {
        "aliases": [
            "show oud", "picture of oud", "show me oud",
            "how oud looks", "oud photo", "image of oud", "image", "picture", "it's image", "it's picture",
            "show me oud", "picture of oud", "how oud looks like", "it's photo", "photo", "it's picture"
        ],
        # 👇 you’ll write your own image path here
        "images": ["static/images/oud_picture_1.webp"],
        "facts": [
            "The Oud is a pear-shaped string instrument widely used in Middle Eastern music.",
            "It is often considered the ancestor of the European lute."
        ]
    },
    "oud_structure": {
        "aliases": [
            "oud structure", "parts of oud", "oud diagram",
            "oud anatomy", "structure of oud", 'the structure'
        ],
        # 👇 you’ll also write your own structure image path
        "images": ["static/images/oud_picture_2.jfif"],
        "facts": [
            "The Oud’s main parts include the soundboard, soundholes, bridge, neck, and pegbox.",
            "It has 11 strings grouped in 5 or 6 courses and has no frets, allowing smooth slides."
        ]
    },
    "oud_audio": {
        "aliases": [
            "sound of oud", "oud audio", "how oud sounds",
            "oud tone", "listen to oud", "hear oud"
        ],
        "audio_files": ["static/audio/oud_sample.mp3"],
        "facts": [
            "Here’s how the Oud sounds — warm, deep, and expressive. 🎵",
            "Its unique timbre comes from its fretless design and hollow body."
        ]
    },

    "oud_professional": {
        "images": ["static/images/professional.png"],
        "facts": ["A professional Oud usually has a spruce top and a walnut or mahogany body for a rich, deep tone."]
    },
    "oud_beginner": {
        "aliases": ["oud for beginners","oud for beginners", "beginner"],
        "images": ["static/images/beginner.png"],
        "facts": ["A beginner’s Oud has nylon strings and lighter wood, making it easier to play for new learners."]
    },


}

# --- Intent Detection with Context ---
def detect_intent(text: str, session: Dict[str, Any]) -> str:
    txt = text.lower().strip()

    # --- Greetings ---
    if re.search(r"\b(hello|hi|hey|salam)\b", txt):
        return "greet"
    if re.search(r"\b(bye|goodbye|see you|good night)\b", txt):
        return "goodbye"

    # --- Ask about chatbot name or Farid ---
    if re.search(
        r"\b("
        r"why\s+(are|r|did|do|you('| a)?re)\s+(you\s+)?(called|named|have\s+that\s+name)|"
        r"who\s+(named|is|was)\s+(you|al[\s-]?atrash)|"
        r"who\s+(do|are)\s+you\s+named\s+after|"
        r"what\s+does\s+al[\s-]?atrash\s+mean|"
        r"tell\s+me\s+about\s+(al[\s-]?atrash|farid)|"
        r"farid\s+al[\s-]?atrash"
        r")\b",
        txt, re.I
    ):
        return "ask_name_origin"

    # --- Oud topics (looser match) ---
    if re.search(r"(understanding the oud|understanding|understand|understand it|understanding it|understand oud)", txt):
        session["learning_topic"] = "about_oud"  # store context
        return "choose_about_oud"

    if re.search(r"(play|play it|how to play|learn to play)", txt):
        session["learning_topic"] = "play_oud"  # store context
        return "choose_play_oud"

    # --- Subtopics after user picks "about oud" ---
    if txt in ["history", "the history", "tell me history", "its history"]:
        return "show_oud_history"
    if txt in ["structure", "its structure", "about structure", 'the structure']:
        return "show_oud_structure"

    # --- Asking for oud recommendation ---
    if re.search(r"(buy|choose|recommend|select|which).*oud", txt):
        return "ask_oud_recommendation"


    # --- Subtopics after user picks "how to play" ---
    if "tune" in txt or "tuning" in txt:
        return "ask_tuning_oud"
    if "stroke" in txt or "basic" in txt or "practice" in txt:
        return "ask_strokes_oud"

    if "advanced stroke" in txt or "advanced technique" in txt:
        return "show_advanced_strokes"

    # --- Farid follow-up ---
    if ("him" in txt or "he" in txt or "his" in txt) and session.get("last_topic") == "farid":
        return "show_farid_info"

    if re.search(r"\b(show me oud|how oud looks|picture of oud|picture|how oud looks like|it's photo|image|it's image|photo|it's picture)\b", txt):
        return "show_oud_picture"

    if re.search(r"\b(structure|structure of| the structure | parts|parts of|diagram|diagram of|anatomy|anatomy of).*(oud)\b", txt):
        return "show_oud_structure"

    # --- Audio playback intent ---
    if "hear sound" in txt or "hear the sound of string" in txt or "play sound" in txt:
        return "hear_string_audio"

    if re.search(r"(sound|audio|hear|listen)", txt):
        return "show_oud_audio"

    # --- Video intent ---
    if re.search(r"\b(video|watch video|play video|see video|show video|tutorial video|tutorial)\b", txt):
        return "show_video"

    # --- Comparing beginner and professional Oud ---
    if re.search(r"(difference between professional and beginner|difference|difference between them|compare between them|compare|compare between professional and beginner|different between them| different between professional and beginner| different)", txt.lower()):
        return "compare_oud_types"


    # --- Affirmation detection with context ---
    if session.get("awaiting_string_audio") and any(word in txt for word in ["yes", "sure", "ok","okay", "yeah"]):
        return "affirm"  # user likely responding to tuning/audio question

    if session.get("awaiting_oud_buy_offer") and any(word in txt for word in ["yes", "sure", "ok","okay", "yeah"]):
        return "affirm_contain_image"  # user likely responding to tuning/audio question

    if session.get("awaiting_video") and any(word in txt for word in ["yes", "sure", "watch", "video"]):
        return "affirm_video"

    # --- User acknowledgment (neutral affirmations) ---
    if any(word in txt for word in
           ["okay", "ok", "sure", "nice", "great", "thanks", "cool", "good", "alright", "amazing", "wonderful",'what else']):
        return "acknowledge"

    # generic fallback affirm/deny
    if any(word in txt for word in ["yes", "sure", "yeah", "ok","okay"]):
        return "affirm"

    # deny
    if any(word in txt for word in ["no", "not really", "skip","nope"]):
        return "deny"

    # --- User asks for deeper explanation ---
    if any(phrase in txt for phrase in [
        "explain more", "give me more details", "clarify", "more info", "tell me more",
        "more about", "can you elaborate", "go deeper", "explain it more", "more details"
    ]):
        return "explain_more"
    if re.search(r"(beginner|beginners|student)", txt):
        return "show_beginner_oud"
    if re.search(r"(famous song|learn song|songs|oud songs|music by farid)", txt):
        return "choose_song"

    return "question"

# --- Name extraction (graceful handling if skipped) ---
def extract_name(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    lower = text.lower()

    # If user doesn’t want to share name
    if any(p in lower for p in ["don't want", "prefer not", "skip", "no name"]):
        return "friend"

    # 1. Look for patterns like "My name is Adam" or "I am Adam"
    if m := re.search(r"\bmy name is ([A-Za-z][A-Za-z\s'-]{0,30})", text, re.I):
        return m.group(1).strip().title()
    if m := re.search(r"\bi(?:'m| am) ([A-Za-z][A-Za-z\s'-]{0,30})", text, re.I):
        return m.group(1).strip().title()

    # 2. If it's just a single word (like "adam"), take it as the name
    words = text.split()
    if len(words) == 1 and words[0].isalpha():
        return words[0].title()

    return ""

# --- Input model ---
class ChatIn(BaseModel):
    sender: str
    message: str


# --- ROUTES ---
@app.post("/chat")
async def chat(payload: ChatIn):
    sender = payload.sender
    text = payload.message.strip()
    session = get_session(sender)
    responses: List[str] = []

    # First load: empty message → send greeting based on current time
    if text == "":
        hour = datetime.now().hour
        if 5 <= hour < 12:
            greeting = "Good Morning"
        elif 12 <= hour < 18:
            greeting = "Good Afternoon"
        elif 18 <= hour < 24:
            greeting = "Good Evening"
        else:
            greeting = "Are you still waking up yet?"

        responses.append(f"{greeting}! I am Al-Atrash, your guide to the world of the Oud. 🎵")
        responses.append("May I know your name? (You can type 'skip' if you prefer not to share)")
        session["awaiting_name"] = True
        save_sessions()
        return JSONResponse({"recipient": sender, "responses": responses})

    # --- 1. PRIORITY: ASK FOR OR SAVE NAME ---
    # --- 1. PRIORITY: HANDLE NAME ---
    if session.get("awaiting_name"):
        name = extract_name(text)

        if name:
            session["user_name"] = name
            session["awaiting_name"] = False
            session["learning_topic"] = None

            save_sessions()

            if name.lower() == "friend":
                responses = [
                    "No problem! I'll call you my friend 🎵",
                    "Would you like to begin with understanding the Oud or how to play it?"
                ]
            else:
                responses = [
                    f"Nice to meet you, {name}! 🎶",
                    f"Would you like to begin with understanding the Oud or how to play it, {name}?"
                ]

            return JSONResponse({"recipient": sender, "responses": responses})

        responses.append("I didn’t quite catch your name. Could you please tell me again?")
        return JSONResponse({"recipient": sender, "responses": responses})

        # --- 2. HANDLE PENDING CHOICES (Like Song Selection) ---
    if session.get("user_name") and session.get("awaiting_song_choice"):
        choice = text.strip().lower()
        if "noura" in choice or "1" in choice:
            song = SONG_VIDEO_MAP["song1"]
        elif "leila" in choice or "layla" in choice or "2" in choice:
            song = SONG_VIDEO_MAP["song2"]
        else:
            responses.append("Please choose 1 or 2 from the list above 🎶")
            return JSONResponse({"recipient": sender, "responses": responses})

        responses.append(f"Excellent choice! Here's **{song['title']}** 🎵")
        responses.append(
                f'<iframe width="100%" height="200" src="{song["url"]}" frameborder="0" allowfullscreen></iframe>'
            )
        session["awaiting_song_choice"] = False
        return JSONResponse({"recipient": sender, "responses": responses})


    # --- 3. DETECT INTENT (Only after name is confirmed) ---
    intent = detect_intent(text, session)
    session["last_intent"] = intent

    # --- NEW: History & Understanding Logic ---

    # 1. Handle the "history" intent specifically
    if intent == "show_oud_history":
        responses.append("The Oud is one of the oldest string instruments, dating back over 5,000 years. 🎶")
        responses.append("It originated in Mesopotamia and evolved into the modern Oud we know in Arabic music today.")
        session["last_topic"] = "history"
        session["learning_topic"] = "about_oud"
        save_sessions()
        return JSONResponse({"recipient": sender, "responses": responses})

    # 2. Handle the "understanding" keyword flow
    if text.lower().strip() in ["understanding the oud", "understanding", "understand"]:
        session["learning_topic"] = "about_oud"
        responses.append("Let's continue exploring the Oud 🎶")
        responses.append("Would you like to learn its History, Structure, Audio or it's image?")
        save_sessions()
        return JSONResponse({"recipient": sender, "responses": responses})


    acknowledge_patterns = [
        r"\bok(ay)?\b",
        r"\bsure\b",
        r"\byes\b",
        r"\byeah\b",
        r"\byes please\b",
        r"\bof course\b",
        r"\bdefinitely\b",
        r"\babsolutely\b"
    ]


    SESSIONS[sender] = session  # 🔹 save immediately
    save_sessions()  # 🔹 persist immediately
    # --- Reset awaiting_picture if topic changed ---
    if session.get("awaiting_picture") and intent not in ["affirm", "ask_name_origin"]:
        session["awaiting_picture"] = False



    if intent == "choose_song":
        responses.append("Great! Which song would you like to learn? 🎶")
        responses.append(f"1️⃣ {SONG_VIDEO_MAP['song1']['title']}")
        session["video_watched"] = True
        responses.append(f"2️⃣ {SONG_VIDEO_MAP['song2']['title']}")
        session["awaiting_song_choice"] = True
        return JSONResponse({"recipient": sender, "responses": responses})


    if "famous song" in text.lower() or "learn song" in text.lower():
        session["learning_topic"] = "famous_song"
        return JSONResponse({"recipient": sender, "responses": ["Great! Which song would you like to learn? 🎶"]})

    # --- Handle intents ---
    # if intent == "greet":
    #     hour = datetime.now().hour
    #     greeting = (
    #         "Good Morning" if 5 <= hour < 12 else
    #         "Good Afternoon" if 12 <= hour < 18 else
    #         "Good Evening"
    #     )
    #     responses.append(f"{greeting}! I am Al-Atrash, your guide to the world of the Oud. 🎵")
    #     responses.append(f"{greeting}, {session['user_name']}! What would you like to learn today — about the Oud or how to play it?")
    #     return JSONResponse({"recipient": sender, "responses": responses})

    # if intent == "ask_oud_recommendation":
    #     responses.append("That’s a great question! 🎸 Choosing the right Oud can make a big difference.")
    #     responses.append("Would you like me to help you find:")
    #     responses.append("1️⃣ The *best Oud to buy* (for quality & sound), or")
    #     responses.append("2️⃣ The *most suitable Oud for beginners* to learn on?")
    #     session["awaiting_oud_recommendation"] = True
    #     session["last_topic"] = "recommendation"
    #     return JSONResponse({"recipient": sender, "responses": responses})
    #
    if intent == "show_farid_info":
        data = KEYWORD_DATA_MAP["farid"]
        responses.append("Here’s **Farid Al-Atrash**, the legendary King of the Oud! 🎶")
        if data["image"]:
            responses.append(f'<img src="{data["image"]}" alt="Farid Al-Atrash" style="max-width:100%;border-radius:10px;margin-top:10px;">')
        responses.extend(data["facts"])
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "show_oud_picture":
        data = KEYWORD_DATA_MAP["oud_picture"]
        responses.append("Here’s what the Oud looks like 🎵")
        for img in data["images"]:
            responses.append(f'<img src="{img}" alt="Oud" style="max-width:100%;border-radius:10px;margin-top:10px;">')
        responses.extend(data["facts"])
        responses.append(
            "Would you like to *buy an Oud*? I can provide you with helpful information before choosing one 🎸")
        session["awaiting_oud_buy_offer"] = True
        session["last_topic"] = "oud_picture"
        return JSONResponse({"recipient": sender, "responses": responses})

        # ✅ User agreed to buy Oud (after picture)
    if intent == "affirm" and session.get("awaiting_oud_buy_offer"):
        responses.append("That’s a great question! 🎸 Choosing the right Oud can make a big difference.")
        responses.append("Would you like me to help you find:")
        responses.append("1️⃣ The *best Oud to buy* (for quality & sound), or")
        responses.append("2️⃣ The *most suitable Oud for beginners* to learn on?")
        session["awaiting_oud_buy_offer"] = False
        session["awaiting_oud_recommendation"] = True
        session["last_topic"] = "recommendation"
        return JSONResponse({"recipient": sender, "responses": responses})

    # --- User said yes to seeing Farid's picture ---
    if intent == "affirm" and session.get("awaiting_picture"):
        data = KEYWORD_DATA_MAP["farid"]
        responses.append("Here’s **Farid Al-Atrash**, the legendary King of the Oud! 🎶")
        if data["image"]:
            responses.append(f'<img src="{data["image"]}" alt="Farid Al-Atrash" style="max-width:100%;border-radius:10px;margin-top:10px;">')
        responses.extend(data["facts"])
        session["awaiting_picture"] = False
        session["last_topic"] = "farid"
        return JSONResponse({"recipient": sender, "responses": responses})

    # ✅ User said YES to seeing professional Oud
    if intent == "affirm" and session.get("awaiting_professional_oud"):
        data = KEYWORD_DATA_MAP["oud_professional"]
        responses.append("Here’s what a *professional Oud* looks like 🎵")
        for img in data["images"]:
            responses.append(
                f'<img src="{img}" alt="Professional Oud" style="max-width:100%;border-radius:10px;margin-top:10px;">')
        responses.extend(data["facts"])
        session["awaiting_professional_oud"] = False
        return JSONResponse({"recipient": sender, "responses": responses})

    # ✅ User said YES to seeing beginner Oud
    if intent == "affirm" and session.get("awaiting_beginner_oud"):
        data = KEYWORD_DATA_MAP["oud_beginner"]
        responses.append("Here’s what a *beginner’s Oud* looks like 🎶")
        for img in data["images"]:
            responses.append(
                f'<img src="{img}" alt="Beginner Oud" style="max-width:100%;border-radius:10px;margin-top:10px;">')
        responses.extend(data["facts"])
        session["awaiting_beginner_oud"] = False
        return JSONResponse({"recipient": sender, "responses": responses})

    if session.get("awaiting_oud_recommendation"):
        # ✅ User wants the best Oud
        if re.search(r"(best|buy|professional|high quality)", text.lower()):
            responses.append(
                "If you’re looking for the *best Oud to buy*, consider one made of walnut or mahogany for the body and spruce for the soundboard 🎶")
            responses.append("Brands like *Sukar* or *Gawharet El Fan* are well-known for their quality.")
            responses.append("Would you like me to show what a professional Oud looks like?")
            session["awaiting_oud_recommendation"] = False
            session["awaiting_professional_oud"] = True  # 👈 add this flag
            return JSONResponse({"recipient": sender, "responses": responses})

        # ✅ User wants the beginner Oud
        if re.search(r"(beginner|learn|student|easy|beginners)", text.lower()):
            responses.append(
                "If you’re a beginner, look for an Oud with nylon strings — it’s easier on the fingers and great for practice 🎵")
            responses.append(
                "Beginner models from brands like *Sukar* or *Istanbul Oud House* are reliable and affordable.")
            responses.append("Would you like me to show a picture of a beginner’s Oud?")
            session["awaiting_oud_recommendation"] = False
            session["awaiting_beginner_oud"] = True  # 👈 add this flag
            return JSONResponse({"recipient": sender, "responses": responses})

    # If user types "best oud" or "most suitable oud" later in chat
    if re.search(r"\b(best oud)\b", text.lower()):
        data = KEYWORD_DATA_MAP["oud_professional"]
        responses.append("Here’s a *professional Oud* 🎵")
        for img in data["images"]:
            responses.append(
                f'<img src="{img}" alt="Professional Oud" style="max-width:100%;border-radius:10px;margin-top:10px;">')
        responses.extend(data["facts"])
        return JSONResponse({"recipient": sender, "responses": responses})

    if re.search(r"\b(most suitable oud|beginner oud|oud for beginners|oud for beginner)\b", text.lower()):
        data = KEYWORD_DATA_MAP["oud_beginner"]
        responses.append("Here’s a *beginner’s Oud* 🎶")
        for img in data["images"]:
            responses.append(
                f'<img src="{img}" alt="Beginner Oud" style="max-width:100%;border-radius:10px;margin-top:10px;">')
        responses.extend(data["facts"])
        return JSONResponse({"recipient": sender, "responses": responses})

    # --- Show video if user said yes ---
    if intent == "affirm_video" and session.get("awaiting_video"):
        video_url = "https://www.youtube.com/embed/Q0X_Yf9AXAU?si=1dqdP7ISUq2ngSVF"
        responses.append("Great! Here’s a video tutorial on how to play the Oud 🎶")
        responses.append(
            f'<iframe width="100%" height="200" src="{video_url}" frameborder="0" allowfullscreen></iframe>')

        # 👇 Mark that the video has been watched
        session["awaiting_video"] = False
        session["video_watched"] = True
        return JSONResponse({"recipient": sender, "responses": responses})

    # --- User explicitly asks for a video (even later in chat) ---
    if intent == "show_video":
        video_url = "https://www.youtube.com/embed/H4he47X8CY4?list=RDH4he47X8CY4&start_radio=1"
        responses.append("Here’s a video tutorial on how to play the Oud 🎶")
        responses.append(
            f'<iframe width="100%" height="200" src="{video_url}" frameborder="0" allowfullscreen></iframe>')
        session["awaiting_video"] = False
        session["last_topic"] = "video"
        return JSONResponse({"recipient": sender, "responses": responses})

    # --- User said yes to hearing string audios ---
    if intent == "affirm" and session.get("awaiting_string_audio"):
        responses.append("Excellent! Let's play each string sound so you can check if yours matches 🎵")
        responses.append('<audio controls src="static/audio/C2.wav"></audio> C2')
        responses.append('<audio controls src="static/audio/F2.wav"></audio> F2')
        responses.append('<audio controls src="static/audio/A2.wav"></audio> A2')
        responses.append('<audio controls src="static/audio/D3.wav"></audio> D3')
        responses.append('<audio controls src="static/audio/G3.wav"></audio> G3')
        responses.append('<audio controls src="static/audio/C4.wav"></audio> C4')
        session["awaiting_string_audio"] = False
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "hear_string_audio":
        responses.append("Excellent! Let's play each string sound so you can check if yours matches 🎵")
        responses.append('<audio controls src="static/audio/C2.wav"></audio> C2')
        responses.append('<audio controls src="static/audio/F2.wav"></audio> F2')
        responses.append('<audio controls src="static/audio/A2.wav"></audio> A2')
        responses.append('<audio controls src="static/audio/D3.wav"></audio> D3')
        responses.append('<audio controls src="static/audio/G3.wav"></audio> G3')
        responses.append('<audio controls src="static/audio/C4.wav"></audio> C4')
        session["awaiting_string_audio"] = False
        return JSONResponse({"recipient": sender, "responses": responses})


    if intent == "deny" and session.get("awaiting_string_audio"):
        responses.append("No problem! You can always ask me later to play the Oud strings. 🎶")
        session["awaiting_string_audio"] = False
        return JSONResponse({"recipient": sender, "responses": responses})


    if intent == "ask_name_origin":
        responses.append("I'm named **Al-Atrash** after the legendary musician **Farid Al-Atrash** — the King of the Oud. 🎵")
        responses.append("Would you like to see a picture of him?")
        session["last_topic"] = "farid"
        session["awaiting_picture"] = True  # <--- ADD THIS LINE
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "choose_about_oud":
        responses.append("Let's explore the Oud together! 🎶")
        responses.append("Would you like to learn its History, Structure, Audio or it's image?")
        session["learning_topic"] = "about_oud"
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "choose_play_oud":
        responses.append("Wonderful! Let’s begin learning how to play the Oud 🎵")
        responses.append("Would you like to start with *tuning* or *basic strokes*?")

        # # ✅ Only ask about the video if user hasn’t already watched it
        # if session.get("user_name") and not session.get("video_watched", False) :
        #     responses.append("Also, would you like to watch a short video on how to play the Oud? 🎬")
        #     session["awaiting_video"] = True  # set flag only here

        responses.append("Or would you like to learn how to play a *famous song*? 🎵")
        responses.append("For example: 1️⃣ Noura Ya Noura  2️⃣ Leila")
        session["awaiting_song_choice"] = True
        session["learning_topic"] = "play_oud"
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "choose_song":
        responses.append("Farid Al-Atrash is famous for many beautiful Oud songs 🎶")
        responses.append("Would you like to learn one of his songs?")
        responses.append("Here are two popular ones:")
        responses.append(f"1️⃣ {SONG_VIDEO_MAP['song1']['title']}")
        responses.append(f"2️⃣ {SONG_VIDEO_MAP['song2']['title']}")
        responses.append("Please choose 1 or 2 🎵")
        session["awaiting_song_choice"] = True
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "video_watched":
        responses.append("Also, would you like to watch a short video on how to play the Oud? 🎬")
        session["awaiting_video"] = True  # Only set when video not yet watched

        session["learning_topic"] = "play_oud"
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "ask_tuning_oud":
        responses.append("Here’s how you can tune your Oud 🎶")
        responses.append("Arabic tuning: C2 – F2 – A2 – D3 – G3 – C4")
        responses.append("Turkish tuning: E2 – A2 – B2 – E3 – A3 – D4")
        responses.append("Would you like to hear the sound of each string so you can compare your Oud tuning? 🎧")
        session["awaiting_string_audio"] = True
        session["last_topic"] = "tuning"  # <--- ADD THIS
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "show_oud_audio":
        data = KEYWORD_DATA_MAP["oud_audio"]
        responses.extend(data["facts"])
        for audio in data["audio_files"]:
            responses.append(f'<audio controls src="{audio}" style="margin-top:10px;"></audio>')
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "show_oud_structure":
        data = KEYWORD_DATA_MAP["oud_structure"]
        responses.append("Here’s the structure of the Oud 🎶")
        for img in data["images"]:
            responses.append(
                f'<img src="{img}" alt="Oud structure" style="max-width:100%;border-radius:10px;margin-top:10px;">')
        responses.extend(data["facts"])
        session["last_topic"] = "structure"  # <--- ADD THIS
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "ask_strokes_oud":
        responses.append("Let’s start with the basic strokes of the Oud 🎶")
        responses.append("Use a plectrum (risha) and practice alternating up and down strokes on each string.")
        session["last_topic"] = "strokes"  # <--- ADD THIS
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "show_advanced_strokes" or (
            intent == "affirm" and session.get("last_topic") == "strokes"
    ):
        responses.append("Alright! Let’s explore some *advanced stroke techniques* 🎶")
        responses.append("Once you’ve mastered the basic alternating strokes, try these:")
        responses.append("🎵 **Tremolo (Risha Rapid)** — rapid up-down strokes for sustained tone.")
        responses.append("🎵 **Double Downstroke** — two quick downstrokes for accent emphasis.")
        responses.append("🎵 **Sweep Stroke** — lightly gliding across multiple strings for a fluid sound.")
        responses.append("Keep your wrist loose — tension kills rhythm! Relax and feel the groove. ✨")
        session["last_topic"] = "advanced_strokes"
        return JSONResponse({"recipient": sender, "responses": responses})

    # ✅ If user said "yes" right after basic strokes — show advanced strokes instead of video
    if intent in ["affirm"] and session.get("last_topic") == "strokes":
        responses.append("Alright! Let’s explore some *advanced stroke techniques* 🎶")
        responses.append("Once you’ve mastered the basic alternating strokes, try these:")
        responses.append("🎵 **Tremolo (Risha Rapid)** — rapid up-down strokes for sustained tone.")
        responses.append("🎵 **Double Downstroke** — two quick downstrokes for accent emphasis.")
        responses.append("🎵 **Sweep Stroke** — lightly gliding across multiple strings for a fluid sound.")
        responses.append("Keep your wrist loose — tension kills rhythm! Relax and feel the groove. ✨")
        session["last_topic"] = "advanced_strokes"
        session["awaiting_video"] = False  # stop waiting for video
        return JSONResponse({"recipient": sender, "responses": responses})


    if session.get("awaiting_oud_recommendation"):
        if re.search(r"(best|buy|professional|high quality)", text.lower()):
            responses.append("If you’re looking for the *best Oud to buy*, consider one made of walnut or mahogany for the body and spruce for the soundboard 🎶")
            responses.append("Brands like Sukar or Gawharet El Fan are well-respected.")
            responses.append("Would you like me to show you how to identify a high-quality Oud?")
            session["awaiting_oud_recommendation"] = False
            return JSONResponse({"recipient": sender, "responses": responses})

        if re.search(r"(beginner|learn|student|easy)", text.lower()):
            responses.append("If you’re just starting out, a *beginner Oud* with nylon strings and standard tuning (C-F-A-D-G-C) is ideal 🎵")
            responses.append("It’s affordable, easier on the fingers, and great for learning basic strokes.")
            responses.append("Would you like me to show a picture of a typical beginner’s Oud?")
            session["awaiting_oud_recommendation"] = False
            return JSONResponse({"recipient": sender, "responses": responses})

    # --- Handle "explain more" ---
    if intent == "explain_more":
        if session.get("learning_topic") == "play_oud":
            # Check if last topic was "strokes"
            if session.get("last_topic") == "strokes":
                responses.append("Sure! Let’s go into more detail about the *basic strokes*. 🎶")
                responses.append("The key is to relax your wrist and let the risha (plectrum) glide naturally.")
                responses.append(
                    "Start slow, alternate up and down, and keep your rhythm steady — like a heartbeat. ❤️‍🔥")
                responses.append("You can practice on open strings before adding notes or melodies.")
                responses.append("Would you like me to show some *advanced stroke techniques* next?")
            elif session.get("last_topic") == "tuning":
                responses.append("Of course! Here’s more about *tuning* your Oud 🎵")
                responses.append("Make sure you tune the bass strings first — C2 and F2 — to anchor the sound.")
                responses.append("Use a tuner app or match to reference sounds I can play for you.")
                responses.append("Would you like to hear the strings again?")
            else:
                responses.append(
                    "Sure! Could you tell me which part you want me to explain more — *tuning* or *strokes*?")
        elif session.get("learning_topic") == "about_oud":
            if session.get("last_topic") == "structure":
                responses.append("The structure of the Oud is fascinating! 🎶")
                responses.append(
                    "The soundboard (front face) is made from spruce or cedar, giving it that resonant tone.")
                responses.append(
                    "The bowl is made from walnut or mahogany — each type affects the warmth of the sound.")
                responses.append("Would you like to learn about the materials or string setup next?")
            elif session.get("last_topic") == "history":
                responses.append(
                    "Historically, the Oud evolved from the Persian barbat and influenced the European lute. 🎵")
                responses.append("It spread through the Islamic Golden Age and became a cornerstone of Arabic music.")
                responses.append("Would you like me to show a timeline image of the Oud’s history?")
            else:
                responses.append(
                    "Sure! Which topic would you like more details about — *history*, *structure*, or *sound*?")
        else:
            responses.append(
                "I’d love to explain more! Which topic would you like to continue with — the Oud itself or how to play it?")
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "goodbye":
        responses.append("Goodbye! Come back anytime to learn more about the Oud 🎵")
        return JSONResponse({"recipient": sender, "responses": responses})

    # --- Continue learning if user says "learn" or "play" alone ---
    if text.lower().strip() in ["learn", "learn it", "play", "play it", "how to play it"]:
        if session.get("learning_topic") == "about_oud":
            return JSONResponse({
                "recipient": sender,
                "responses": [
                    "Let's continue exploring the Oud 🎶 Would you like to learn its History, Structure, Audio or it's image?"
                ]
            })
        elif session.get("learning_topic") == "play_oud":
            return JSONResponse({
                "recipient": sender,
                "responses": [
                    "Let's continue learning how to play the Oud 🎵 Would you like to start with *tuning* or *basic strokes*?"
                ]
            })

    if session.get("last_topic") == "video":
        responses.append(
            "Would you like me to show another Oud playing tutorial or continue with *tuning* or *strokes*? 🎶")
        return JSONResponse({"recipient": sender, "responses": responses})

    # --- Handle user acknowledgment (okay, nice, etc.) ---
    # --- Handle acknowledgment with topic context ---
    if intent == "acknowledge":
        # If user just saw Oud picture and we offered to buy
        if session.get("last_topic") == "oud_picture" or session.get("awaiting_oud_buy_offer"):
            responses.append(
                "Would you like to *buy an Oud*? I can provide you with helpful information before choosing one 🎸")
            session["awaiting_oud_buy_offer"] = True
            return JSONResponse({"recipient": sender, "responses": responses})

        # If last topic was recommendation
        if session.get("last_topic") == "recommendation":
            responses.append("Would you like to see what a *professional* or *beginner* Oud looks like? 🎵")
            return JSONResponse({"recipient": sender, "responses": responses})

        # Fallback to generic acknowledgment
        if session.get("learning_topic"):
            topic = session["learning_topic"]
            if topic == "about_oud":
                responses.append(
                    "Glad you’re enjoying this! 🎶 Would you like to explore its *history*, *structure*, or *sound* next?")
            elif topic == "play_oud":
                if session.get("video_watched", False):
                    responses.append(
                        "Awesome! 🎵 Would you like to continue with *tuning* or *strokes*?")
                else:
                    responses.append(
                        "Awesome! 🎵 Would you like to continue with *tuning*, *strokes*, or maybe watch a short playing video?")
            else:
                responses.append("Nice! Would you like to learn more about the Oud or how to play it?")
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "show_beginner_oud":
        data = KEYWORD_DATA_MAP["oud_beginner"]
        responses.append("Here’s what a *beginner’s Oud* looks like 🎶")
        for img in data["images"]:
            responses.append(
                f'<img src="{img}" alt="Beginner Oud" style="max-width:100%;border-radius:10px;margin-top:10px;">'
            )
        responses.extend(data["facts"])
        return JSONResponse({"recipient": sender, "responses": responses})

    if intent == "compare_oud_types":
        responses.append("Here’s how a *beginner Oud* differs from a *professional Oud*, both in appearance and performance 🎶")
        responses.append("🎸 **Design & Craftsmanship:** Beginner Ouds have a simple design made from basic woods, while professional ones feature decorative details and high-quality materials like walnut or rosewood.")
        responses.append("🎵 **Sound Quality:** Beginner Ouds produce lighter tones, while professional ones have a deeper, richer, and more resonant sound.")
        responses.append("🎼 **Playability:** Professional Ouds are smoother and more precise to play, while beginner Ouds are easier to maintain but less sensitive to touch.")
        responses.append("💰 **Price:** Beginner Ouds cost around $100–$300, while professional ones range from $700 to over $3000.")
        responses.append('<div><img src="static/images/oud_difference.png" style="max-width:220px;border-radius:10px;"></div>')
        return JSONResponse({"recipient": sender, "responses": responses})

    # # --- Handle question about difference between beginner and professional Oud ---
    # if re.search(r"(difference|compare).*(beginner|professional).*oud", text.lower()):
    #     responses.append(
    #         "Here’s how a *beginner Oud* differs from a *professional Oud*, both in appearance and performance 🎶")
    #     responses.append(
    #         "🎸 **Design & Craftsmanship:** Beginner Ouds have a simple design made from basic woods, while professional ones feature decorative details and high-quality materials like walnut or rosewood.")
    #     responses.append(
    #         "🎵 **Sound Quality:** Beginners produce lighter tones, while professional Ouds have a deeper, richer, and more resonant sound.")
    #     responses.append(
    #         "🎼 **Playability:** Professional Ouds are smoother and more precise to play, while beginner Ouds are easier to maintain but less sensitive to touch.")
    #     responses.append(
    #         "💰 **Price:** Beginner Ouds cost around $100–$300, while professional ones range from $700 to over $3000.")
    #     responses.append("Here’s how they look 👇")
    #     responses.append('<div style="display:flex;flex-wrap:wrap;gap:15px;">')
    #     responses.append(
    #         '<div><p><b>Beginner Oud 🎶</b></p><img src="static/images/oud_beginner.webp" alt="Beginner Oud" style="max-width:220px;border-radius:10px;"><p>Simple body, light tone.</p></div>')
    #     responses.append(
    #         '<div><p><b>Professional Oud 🎵</b></p><img src="static/images/oud_professional.webp" alt="Professional Oud" style="max-width:220px;border-radius:10px;"><p>Decorative design, rich tone.</p></div>')
    #     responses.append('</div>')
    #     return JSONResponse({"recipient": sender, "responses": responses})
    # --- Asking specifically about beginner Oud ---

    # --- When user agrees to buy Oud (yes/ok/sure/etc.) ---
    if intent in ["affirm_contain_image", "acknowledge"] and session.get("awaiting_oud_buy_offer"):
        responses.append("That’s a great question! 🎸 Choosing the right Oud can make a big difference.")
        responses.append(
            "Would you like me to help you find:\n1️⃣ The *best Oud to buy* (for quality & sound), or\n2️⃣ The *most suitable Oud for beginners* to learn on?")
        session["awaiting_oud_buy_offer"] = False
        session["awaiting_oud_recommendation"] = True
        session["last_topic"] = "recommendation"
        return JSONResponse({"recipient": sender, "responses": responses})

    # Fallback
    # --- Final fallback for unmatched intents ---
    if not responses:
        answers = retrieve_best_answer(text)
        if answers:
            responses.extend(answers)
        else:
            responses.append(
                "I’m not sure I understood. Could you rephrase that, or would you like to explore the Oud’s History, Structure, Audio, or Image?"
            )


    SESSIONS[sender] = session  # 🔹 save immediately
    save_sessions()  # 🔹 persist immediately
    return JSONResponse({"recipient": sender, "responses": responses})

