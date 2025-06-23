import os
import re
import json
import uuid
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import pyttsx3
import speech_recognition as sr
from pydub import AudioSegment
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===
AudioSegment.converter = r"E:\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
MODEL_PATH = r"C:\Users\ESHANK\.cache\huggingface\hub\models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF\snapshots\52e7645ba7c309695bec7ac98f4f005b139cf465\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
DEP_MODEL_PATH = r"C:/Users/ESHANK/Documents/python 2/models/depression_model.h5"
LOGS_DIR = r"C:/Users/ESHANK/Documents/python 2/DrishtiLogs"
os.makedirs(LOGS_DIR, exist_ok=True)

VALID_API_KEYS = { os.getenv("DRISHTI_API_KEY") } # Replace with env var or DB in production

tts_engine = pyttsx3.init()

llm = CTransformers(
    model=MODEL_PATH,
    model_type="llama",
    lib="avx2",
    config={"temperature": 0.7, "top_k": 50, "max_new_tokens": 60}
)

template = """You are Drishti, a warm, supportive best friend who speaks Hinglish.
ONLY generate Drishti's reply — never user lines. Never mention AI.
Be brief, friendly, caring.

Chat History:
{history}
User: {input}
Drishti:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
memory = ConversationBufferMemory(return_messages=True)
conversation_chain = ConversationChain(llm=llm, prompt=prompt, memory=memory, verbose=False)

# === UTIL FUNCTIONS ===
def clean_response(text):
    text = text.strip().split("\n")[0]
    text = re.split(r"(User:|Drishti:|Me:)", text)[0].strip()
    return text

def detect_sensitive(text):
    return any(w in text.lower() for w in ["suicide", "kill myself", "end my life", "harm myself"])

def user_wants_to_end(text):
    return any(w in text.lower() for w in ["end", "stop", "bye", "goodbye", "exit", "hang up"])

def predict_depression(features):
    model = tf.keras.models.load_model(DEP_MODEL_PATH)
    preds = model.predict(features)
    pred_class = np.argmax(preds)
    classes = ["No Depression", "Mild Depression", "Moderate/Severe Depression"]
    return {
        "class": classes[pred_class],
        "confidence": float(preds[0][pred_class])
    }

# === FASTAPI APP ===
app = FastAPI()

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-KEY")
    if api_key not in VALID_API_KEYS:
        return JSONResponse(status_code=401, content={"detail": "Invalid API Key"})
    return await call_next(request)

@app.post("/drishti/start_session")
async def start_session(data: dict):
    user_inputs = data.get("inputs")
    if not user_inputs or not isinstance(user_inputs, list):
        raise HTTPException(status_code=400, detail="Inputs must be a list of user messages.")

    transcripts = []
    for turn, text in enumerate(user_inputs, start=1):
        print(f"🗣 You: {text}")
        transcripts.append({"turn": turn, "user": text})

        if user_wants_to_end(text):
            response = "Thik hai, bye bye! Take care 💖"
        elif detect_sensitive(text):
            response = "Aisa mat socho yaar. Please baat karo kisi dost ya counselor se. Main tumhare saath hoon."
        else:
            raw = conversation_chain.invoke({"input": text})
            response = clean_response(raw.get("response", raw))

        transcripts[-1]["drishti"] = response

    # === DUMMY FEATURE EXTRACTION ===
    features = np.random.rand(1, 203)  # Replace with actual features
    pred = predict_depression(features)

    session_id = str(uuid.uuid4())
    result = {
        "session_id": session_id,
        "transcript": transcripts,
        "depression_analysis": pred
    }

    json_path = os.path.join(LOGS_DIR, f"drishti_session_{session_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("drishti_api:app", host="127.0.0.1", port=8000, reload=True)

