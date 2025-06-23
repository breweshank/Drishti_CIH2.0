import os
import re
import json
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import pyttsx3
import speech_recognition as sr
from pydub import AudioSegment
import tensorflow as tf
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# === CONFIG ===
AudioSegment.converter = r"E:\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
MODEL_PATH = r"C:\Users\ESHANK\.cache\huggingface\hub\models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF\snapshots\52e7645ba7c309695bec7ac98f4f005b139cf465\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
DEP_MODEL_PATH = r"C:/Users/ESHANK/Documents/python 2/models/depression_model.h5"
LOGS_DIR = r"C:/Users/ESHANK/Documents/python 2/DrishtiLogs"
os.makedirs(LOGS_DIR, exist_ok=True)

tts_engine = pyttsx3.init()

# === LLM SETUP ===
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

# === FUNCTIONS ===
def speak(text):
    print(f"\n🤖 Drishti: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def record_audio(filename, duration=5, fs=16000):
    print("\n🎤 Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio_int16 = np.int16(audio * 32767)
    wav.write(filename, fs, audio_int16)
    print(f"[✔] Saved to {filename}")

def convert_to_pcm_wav(file_path):
    sound = AudioSegment.from_file(file_path)
    pcm_path = file_path.replace(".wav", "_pcm.wav")
    sound.export(pcm_path, format="wav", codec="pcm_s16le")
    return pcm_path

def transcribe_audio(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except:
        return ""

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

# === MAIN ===
def main():
    speak("Hello, I'm Drishti. Baat karo freely. Say 'stop' anytime to end.")
    transcripts = []
    turn = 1

    while True:
        filename = os.path.join(LOGS_DIR, f"voice_{turn}.wav")
        record_audio(filename)

        try:
            pcm_path = convert_to_pcm_wav(filename)
        except:
            speak("Sorry, kuch galat ho gaya. Try again?")
            continue

        text = transcribe_audio(pcm_path)
        if not text:
            speak("Didn't catch that, bolo na dobara?")
            continue

        print(f"🗣 You: {text}")
        transcripts.append({"turn": turn, "user": text})

        if user_wants_to_end(text):
            speak("Thik hai, bye bye! Take care 💖")
            break

        if detect_sensitive(text):
            response = "Aisa mat socho yaar. Please baat karo kisi dost ya counselor se. Main tumhare saath hoon."
        else:
            raw = conversation_chain.invoke({"input": text})
            response = clean_response(raw.get("response", raw))

        transcripts[-1]["drishti"] = response
        speak(response)
        turn += 1

    # === DEPRESSION PREDICTION ===
    # TODO: Replace below with actual feature extraction
    combined_transcript = " ".join([t["user"] for t in transcripts])
    # features = your_real_feature_extraction(...)
    features = np.random.rand(1, 203)  # Dummy placeholder to match model input

    pred = predict_depression(features)

    # === SAVE JSON ===
    result = {
        "transcript": transcripts,
        "depression_analysis": pred
    }
    json_path = os.path.join(LOGS_DIR, "drishti_session.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n📄 Session saved to {json_path}")
    print(f"🤖 Drishti's Analysis: {pred['class']} (Confidence: {pred['confidence']:.2f})")

if __name__ == "__main__":
    main()
