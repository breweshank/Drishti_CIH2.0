import os
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import pyttsx3
import speech_recognition as sr
from pydub import AudioSegment
from huggingface_hub import hf_hub_download 
from ctransformers import AutoModelForCausalLM

# FFmpeg path (ensure correct)
AudioSegment.converter = r"your_local_path/ffmpeg-7.1.1-full_build/ffmpeg-7.1.1-full_build/bin/ffmpeg.exe"
if not os.path.exists(AudioSegment.converter):
    raise EnvironmentError("FFmpeg not found at specified path.")

# Create logs folder
os.makedirs("drishti_logs", exist_ok=True)

# TTS
tts_engine = pyttsx3.init()
def speak(text):
    print("\n🤖 Drishti:", text)
    tts_engine.say(text)
    tts_engine.runAndWait()

# Record user voice
def record_audio(filename, duration=5, fs=16000):
    print("\n🎤 Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio_int16 = np.int16(audio * 32767)
    wav.write(filename, fs, audio_int16)
    print(f"[✔] Saved to {filename}")

# Convert to PCM WAV
def convert_to_pcm_wav(file_path):
    sound = AudioSegment.from_file(file_path)
    pcm_path = file_path.replace(".wav", "_pcm.wav")
    sound.export(pcm_path, format="wav", codec="pcm_s16le")
    return pcm_path

# Transcribe
def transcribe_audio(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except:
        return ""

# Auto-download model to HuggingFace cache
MODEL_ID = "TheBloke/phi-2-GGUF"
MODEL_FILE = "phi-2.Q4_K_M.gguf"

MODEL_PATH = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILE)

# Load LLM
def get_llm_response(conversation_history):
    if not hasattr(get_llm_response, "chatbot"):
        get_llm_response.chatbot = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=MODEL_PATH,
            model_type="phi",
            lib="cuda",  # Use CUDA for GPU acceleration
            max_new_tokens=256,
            temperature=0.7,
            top_k=50
        )

    # Construct prompt
    prompt = ""
    for msg in conversation_history:
        role, content = msg["role"], msg["content"]
        prompt += f"<|{role}|> {content} <|end|>\n"
    prompt += "<|assistant|>"

    output = get_llm_response.chatbot(prompt)
    reply = output.split("<|end|>")[0].replace("<|assistant|>", "").strip()
    return reply

# Check exit
def user_wants_to_end(text):
    return any(word in text.lower() for word in ["end", "exit", "stop the call", "end the call"])

# Main loop
def main():
    speak("Hello, I'm Drishti. I'm here for you. Say 'end the call' when you're done.")

    transcripts = []
    conversation = [
        {
            "role": "system",
            "content": "You are Drishti, a warm, empathetic, emotionally aware voice assistant. You talk like a human—compassionate, supportive, and calming."
        }
    ]
    turn = 1

    while True:
        filename = f"drishti_logs/voice_{turn}.wav"
        record_audio(filename)

        try:
            pcm_path = convert_to_pcm_wav(filename)
        except Exception as e:
            print(f"[Error] {e}")
            speak("I couldn’t process that. Let's try again.")
            continue

        text = transcribe_audio(pcm_path)
        if not text:
            speak("Hmm, I didn’t catch that. Could you say it again?")
            continue

        print(f"🗣️ You: {text}")
        transcripts.append(f"You: {text}")
        conversation.append({"role": "user", "content": text})

        if user_wants_to_end(text):
            speak("Alright. I'm always here whenever you need me. Take care.")
            break

        response = get_llm_response(conversation)
        conversation.append({"role": "assistant", "content": response})
        transcripts.append(f"Drishti: {response}")
        speak(response)

        turn += 1

    with open("drishti_logs/transcript.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(transcripts))

    print("\n📄 Conversation saved to drishti_logs/transcript.txt")
    speak("It was lovely talking with you. Stay strong and kind.")

if __name__ == "__main__":
    main()
