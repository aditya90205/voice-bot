"""
Voice Agent Backend — Powered entirely by Sarvam AI
Stack: Saarika (STT) → Sarvam-M (LLM) → Bulbul (TTS)

Install: pip install fastapi uvicorn httpx python-multipart
Run:     uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import os
import re

app = FastAPI(title="Sarvam Voice Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_SERVER_API_KEY = "sk_1ej736pu_k1I1gZebvYixjojxGcmI9KvT"
SARVAM_API_KEY = (os.getenv("SARVAM_API_KEY") or DEFAULT_SERVER_API_KEY).strip()

SARVAM_STT_URL  = "https://api.sarvam.ai/speech-to-text"
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_TTS_URL  = "https://api.sarvam.ai/text-to-speech"

# In-memory conversation store (keyed by session_id from frontend)
sessions: dict[str, list] = {}

SYSTEM_PROMPT = """You are an expert technical interviewer for a Full Stack Developer (Next.js + Python) role.
Your name is Sarvam Bot.
Follow this interview flow strictly:
1. Introduce yourself briefly and ask the candidate to introduce themselves. Wait for their response.
2. Ask EXACTLY 6 technical questions consecutively (one by one, waiting for the candidate's answer after each). Focus on Next.js, React, Python, APIs, and general full-stack concepts. Do NOT ask all 6 questions at once.
3. After the 6 questions are answered, conclude the interview and provide brief context-aware feedback and the result based on their answers. MUST include the exact text `[INTERVIEW_DONE]` at the very end of your final feedback message so the system knows to stop.
Keep all responses SHORT and CONVERSATIONAL — 1 to 3 sentences maximum.
Speak naturally as if in a video interview. Avoid bullet points, markdown, or long explanations.
Never include chain-of-thought, internal reasoning, or tags like <think>...</think>.
Never include emoji, XML/HTML tags, markdown symbols, or bracketed stage directions.

You STRICTLY understand and respond ONLY in English. If the candidate speaks in another language (e.g. Hindi, Tamil, etc.), politely reply EXACTLY with: 'I cannot understand your language. Please speak in English if possible.' and do not attempt to continue the interview until they do."""


# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_api_key(request: Request, form_key: str = "") -> str:
    """Resolve API key with server default first, then request fallbacks."""
    if SARVAM_API_KEY:
        return SARVAM_API_KEY

    if form_key and form_key.strip():
        return form_key.strip()

    header_key = (request.headers.get("x-sarvam-api-key") or "").strip()
    if header_key:
        return header_key

    subscription_key = (request.headers.get("api-subscription-key") or "").strip()
    if subscription_key:
        return subscription_key

    auth_header = (request.headers.get("authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        bearer_token = auth_header[7:].strip()
        if bearer_token:
            return bearer_token

    return ""


def extract_http_error(prefix: str, err: httpx.HTTPStatusError) -> Exception:
    """Create readable upstream error with response body if available."""
    status = err.response.status_code
    body = (err.response.text or "").strip()
    if status in (401, 403):
        return Exception(
            f"{prefix} auth failed ({status}). Check your Sarvam API key and permissions."
            + (f" Response: {body}" if body else "")
        )
    return Exception(f"{prefix} failed ({status})." + (f" Response: {body}" if body else ""))


def sanitize_assistant_text(text: str) -> str:
    cleaned = (text or "").strip()

    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think[^>]*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"\([^\)]*\)", " ", cleaned)

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE,
    )
    cleaned = emoji_pattern.sub(" ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ── 1. Speech-to-Text (Saarika v2.5) ─────────────────────────────────────────

async def speech_to_text(audio_bytes: bytes, filename: str, api_key: str, content_type: str) -> str:
    """Send native WAV from browser to Sarvam Saarika STT."""
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": api_key},
                files={"file": (filename, audio_bytes, "audio/wav")},
                data={
                    "model": "saarika:v2.5",
                    "language_code": "en-IN",
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise extract_http_error("STT", err) from err
        data = response.json()
        print(f"STT Response: {data}")
        return data.get("transcript", "").strip()


# ── 2. LLM (Sarvam-M) ────────────────────────────────────────────────────────
async def get_llm_response(session_id: str, user_text: str, api_key: str) -> str:
    """Get a response from Sarvam-M, maintaining per-session conversation history."""
    if session_id not in sessions:
        sessions[session_id] = []

    sessions[session_id].append({"role": "user", "content": user_text})

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                SARVAM_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sarvam-m",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *sessions[session_id],
                    ],
                    "max_tokens": 150,
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise extract_http_error("LLM", err) from err
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        assistant_text = sanitize_assistant_text(raw_text)

    if not assistant_text:
        assistant_text = "Sorry, I could not generate a clean response. Please ask again."

    sessions[session_id].append({"role": "assistant", "content": assistant_text})
    return assistant_text


# ── 3. Text-to-Speech (Bulbul v3) ────────────────────────────────────────────
async def text_to_speech(text: str, api_key: str, language_code: str = "en-IN") -> str:
    """Convert text → base64-encoded WAV audio using Sarvam Bulbul v3."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                SARVAM_TTS_URL,
                headers={
                    "api-subscription-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": [text],             # must be an array
                    "target_language_code": language_code,
                    "speaker": "anushka",         # bulbul:v3 speakers (case-sensitive):
                                                  # Female: Anushka, Manisha, Vidya, Arya, Ritu, Priya
                                                  # Male:   Abhilash, Karun, Hitesh, Shubh, Aditya
                    "model": "bulbul:v2",         # or "bulbul:v3" for latest
                    "pace": 1.0,                  # 0.3–3.0 for v2 | 0.5–2.0 for v3
                    "speech_sample_rate": 22050,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise extract_http_error("TTS", err) from err
        data = response.json()
        return data["audios"][0]


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/process-audio")
async def process_audio(
    request: Request,
    file: UploadFile,
    session_id: str = Form(default="default"),
    tts_language: str = Form(default="en-IN"),
    sarvam_api_key: str = Form(default=""),
):
    """
    Main pipeline: audio → STT → LLM → TTS → JSON response
    
    Returns:
        user_text:      what the user said
        assistant_text: LLM response
        audio_b64:      base64 WAV to play in browser
        detected_lang:  language detected by STT
    """
    audio_bytes = await file.read()
    filename = file.filename or "audio.wav"
    api_key = resolve_api_key(request, sarvam_api_key)

    if not api_key:
        return JSONResponse(
            {
                "error": "Sarvam API key is missing on the backend.",
            },
            status_code=400,
        )

    # Step 1: STT
    try:
        user_text = await speech_to_text(audio_bytes, filename, api_key, file.content_type or "audio/wav")
    except Exception as e:
        return JSONResponse({"error": f"STT failed: {str(e)}"}, status_code=500)

    if not user_text:
        return JSONResponse({"error": "Could not understand audio. Please try again."}, status_code=400)

    # Step 2: LLM
    try:
        assistant_text = await get_llm_response(session_id, user_text, api_key)
    except Exception as e:
        return JSONResponse({"error": f"LLM failed: {str(e)}"}, status_code=500)

    # Step 3: TTS
    try:
        audio_b64 = await text_to_speech(assistant_text, api_key, tts_language)
    except Exception as e:
        return JSONResponse({"error": f"TTS failed: {str(e)}"}, status_code=500)

    return {
        "user_text": user_text,
        "assistant_text": assistant_text,
        "audio_b64": audio_b64,
    }


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/health")
async def health():
    return {"status": "ok", "api_key_set": bool(SARVAM_API_KEY)}


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
