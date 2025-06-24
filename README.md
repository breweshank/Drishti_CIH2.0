# Drishti_CIH2.0
This repository conains tye complete project built up by Team Drishti on Central India Hackaathon 2.0.



# 👁️ Project Drishti

### Developed by **Team Drishti**

---

## 🔍 Overview

**Drishti** is an AI-powered mental wellness companion designed to detect signs of depression through voice-based interactions. Built using speech analysis, Natural Language Processing (NLP), and intelligent backend modeling, the project aims to bridge the gap between individuals and mental health support through seamless, private, and empathetic communication.

---

## 🧠 Key Features

- 🎤 **Voice-Based Interaction**: Real-time voice communication using speech recognition and synthesis.
- 🗣️ **Speech Analysis**: Detects emotional and mental state from voice patterns and content.
- 🧾 **Depression Detection**: Classifies user responses into `Depression`, `False Depression`, or `No Depression`.
- 🧠 **NLP & NLU Integration**: Contextual understanding of user responses.
- 🧪 **Model Trained on DAIC-WOZ Dataset**: Ensures clinically backed performance.
- 📊 **Real-time API Communication**: Interacts with Flutter frontend using Flask APIs.
- 🧾 **Logging System**: Stores conversation logs securely for future analysis.

---

## 🧰 Tech Stack

| Component       | Technology Used                        |
|----------------|----------------------------------------|
| Voice I/O       | `speech_recognition`, `pyttsx3`, `sounddevice`, `pydub` |
| Backend API     | `Flask`, `FastAPI`                    |
| ML/NLP Models   | `Naive Bayes Tree`, `scikit-learn`, `NLTK`, `Transformers` (offline) |
| Dataset         | `DAIC-WOZ` (licensed)                  |
| Frontend        | `Flutter`, `Figma` (UI design)        |
| Model Hosting   | Local / Hugging Face (for offline inference) |
| Audio Processing| `FFmpeg`, `wav`, `numpy`              |


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `ffmpeg` (installed and added to PATH)
- DAIC-WOZ Dataset (licensed)
- Flutter (for frontend integration)

### Clone the Repository

```bash
git clone https://github.com/TeamDrishti/drishti.git
cd drishti
````

### Setup Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Backend

```bash
python voice_mode.py
# Or if using FastAPI
uvicorn main:app --reload
```

---

## 🌐 API Endpoints

| Endpoint                 | Method | Description                                  |
| ------------------------ | ------ | -------------------------------------------- |
| `/drishti/start_session` | POST   | Starts a new voice session                   |
| `/drishti/analyze`       | POST   | Analyzes voice and returns depression status |
| `/drishti/stop`          | POST   | Ends session and saves logs                  |

---

## 📁 Folder Structure

```
drishti/
│
├── app/                      # Flask/FastAPI backend
├── model/                    # Trained ML models
├── data/                     # DAIC-WOZ data (user licensed)
├── logs/                     # User voice session logs
├── assets/                   # Figma design and frontend assets
├── voice_mode.py             # Main script for Drishti voice assistant
├── requirements.txt
└── README.md
```

---

## 🤝 Team Drishti

We are a passionate team of AI/ML developers, mental health enthusiasts, and software engineers dedicated to creating impactful technology for emotional well-being.

**Team Members:**

* \Nandini Jaiswal – AI/ML Developer
* \Nandini Jaiswal – Backend Developer
* \Mohit Rahangdale – Flutter Developer
* \Mohit Rahandale – UI/UX Designer
* \Eshank Ryshbah – AR & Hardware Developer

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

> ⚠️ **Disclaimer**: Drishti is not a replacement for professional mental health services. It is intended for research, support, and early screening purposes only.

---

## 🌟 Acknowledgements

* DAIC-WOZ Dataset by USC Institute for Creative Technologies
* Hugging Face for open model support
* Flutter & Figma community for UI inspiration

---

## 📬 Contact

For queries, collaboration, or suggestions:

📧 [teamdrishti](mailto:teamdrishti.ai@gmail.com)
🔗 [GitHub](https://github.com/breweshank/Drishti_CIH2.0)

---

🧠 *Empathy through AI — that’s the vision of Team Drishti.*

```

---

Would you like this converted into a GitHub-compatible preview or include a project banner/logo?
```
