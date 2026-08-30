# 🔍 InstaFactCheck — Instagram Video & Post Fact-Checker

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/shubhansujee9/InstaFactCheck)

An end-to-end AI fact-checking application that analyzes Instagram Reels, video posts, and carousel/photo posts. It downloads content, transcribes speech, extracts discrete factual assertions using an LLM, verifies evidence, and returns an interactive fact-check report with color-coded verdict badges, explanations, and citations.


---

## 🏗️ Architecture

```mermaid
flowchart TD
    IG["Instagram App (Share Sheet) / Web Input"] -->|"Reel/Post URL or Text"| App["Flutter App / Web UI"]
    App -->|"POST /analyze"| API["FastAPI Backend"]
    API --> DL["yt-dlp: Download Reel/Post + Caption"]
    DL --> TR["ffmpeg: Audio Extraction"]
    TR --> CL["LLM (OmniRoute / GPT-4o): Extract Claims"]
    CL --> VR["LLM / Web Evidence: Fact-check per claim in parallel"]
    VR -->|"JSON Report"| App
    App --> UI["Report Screen: Verdict Badges, Explanations, Sources"]
```

---

## ⚡ Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **AI Gateway & LLM**: [OmniRoute](https://github.com/darianhickman/omniroute) / OpenAI (`auto/best-chat`, `gpt-4o`)
- **Video & Metadata Processing**: `yt-dlp`, `ffmpeg`, `instaloader`
- **Mobile App**: Flutter (Android Share Sheet integration via `receive_sharing_intent`, Material 3 design)
- **Web UI**: Modern dark-mode web application served directly from FastAPI
- **Hosting / Tunneling**: Cloudflare Tunnels (`cloudflared`) & Docker

---

## 🚀 Quick Start

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

Edit `.env` with your API configuration:
```env
OPENAI_API_KEY=your_key_or_omniroute_key
OPENAI_BASE_URL=http://localhost:20128/v1
OPENAI_MODEL=auto/best-chat
```

Run the backend server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Web UI**: Open [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 2. Flutter Mobile App Setup

```bash
cd flutter_app

# Install dependencies
flutter pub get

# Run on connected Android device or emulator
flutter run
```

---

### 3. Docker Deployment

```bash
cd backend
docker-compose up --build
```

---

## 📋 Data Contract (`POST /analyze`)

### Request
```json
{
  "url": "https://www.instagram.com/reel/DcoihYFhlOO/"
}
```
*or direct claim text:*
```json
{
  "text": "Eating raw onions cures bacterial infections within 2 hours without antibiotics."
}
```

### Response
```json
{
  "overall_summary": "This video contained 5 verifiable claims. Of these, 4 verified as true, 1 mostly true.",
  "overall_verdict": "mostly_true",
  "claims": [
    {
      "claim_text": "Ravi Kishan is a Member of Parliament representing the BJP.",
      "verdict": "true",
      "explanation": "Ravi Kishan serves as an MP in the Lok Sabha representing the Gorakhpur constituency.",
      "sources": [
        {
          "title": "Members : Lok Sabha",
          "url": "https://sansad.in/ls/members"
        }
      ]
    }
  ],
  "video_title": "Video by unscripted.india_",
  "transcript_snippet": null
}
```

---

## 🛡️ License
MIT License
