#  Stress Detection using Speech and Face
### Real-time Emotion & Stress Analysis Web Application

---

##  Project Structure

```
stress_detection/
├── app.py                          ← Main Flask backend
├── requirements.txt                ← Python dependencies
├── models/
│   ├── face_emotion_model.keras    ← YOUR face model (copy here)
│   └── speech_emotion_model.keras ← YOUR speech model (copy here)
├── templates/
│   ├── home.html                   ← Landing page (beige + leaves)
│   ├── login.html                  ← Login page
│   ├── face_detection.html         ← Webcam face emotion page
│   ├── audio_detection.html        ← Speech recording page
│   └── accuracy.html               ← Model accuracy graphs
└── static/
    ├── css/
    ├── js/
    └── images/
```

---

##  Setup Instructions

### Step 1: Copy your models
```
stress_detection/
└── models/
    ├── face_emotion_model.keras     ← copy your file here
    └── speech_emotion_model.keras   ← copy your file here
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the app
```bash
python app.py
```

### Step 4: Open browser
```
http://127.0.0.1:5000
```

---

##  Pages & URLs

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Landing page with project title |
| Login | `/login` | Login form |
| Face Detection | `/face` | Live webcam + emotion detection |
| Audio Detection | `/audio` | Speech recording + emotion analysis |
| Accuracy | `/accuracy` | Model accuracy graphs |

---

##  Stress Calculation Logic

**Stress Score = Average of (Angry% + Fear% + Disgust% + Sad%)**

| Score | Level | Color |
|-------|-------|-------|
| 0–30% |  Low Stress | Green |
| 31–60% |  Medium Stress | Orange |
| 61–100% |  High Stress | Red |

---

##  Emotion Labels

**Face Model:**
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

**Speech Model:**
neutral, calm, happy, sad, angry, fear, disgust, surprise




```bash
# Windows (using chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```
