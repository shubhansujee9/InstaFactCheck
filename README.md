# 🔍 InstaFactCheck — Instagram Video & Post Fact-Checker

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/shubhansujee9/InstaFactCheck)

An end-to-end AI fact-checking application that analyzes Instagram Reels, video posts, and carousel/photo posts. It downloads content, transcribes speech, extracts discrete factual assertions using an LLM, verifies evidence, and returns an interactive fact-check report with color-coded verdict badges, explanations, and citations.


---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph IN["1. Ingestion Layer"]
        IG["Instagram App (Share Sheet) / Web Input"] -->|"Reel / Post URL or Claim Text"| APP["Flutter Mobile App / Web UI"]
        APP -->|"POST /analyze"| API["FastAPI Backend (Port 8000)"]
        API --> DL["yt-dlp & instaloader: Download Media & Caption"]
    end

    subgraph MM["2. Multimodal Forensics & Understanding Layer"]
        DL -->|"Video Stream (.mp4)"| GEM_MM["Gemini 2.5 Flash Native Video/Audio API"]
        GEM_MM -->|"Multilingual Transcript (Hindi/English)"| TS["Verbatim Speech & Speaker ID"]
        GEM_MM -->|"Visual Frame Forensics"| VF["Visual Scenes, Meme OCR & AI Deepfake Checks"]
        
        DL -.->|"Offline Fallback"| WHISPER["Local faster-whisper (tiny int8) + FFmpeg Keyframes"]
        WHISPER -.-> TS
        WHISPER -.-> VF
    end

    subgraph CC["3. Cross-Modal Consistency & Claim Extraction Layer"]
        TS & VF & DL -->|"Video Reality vs Caption"| EXTRACTOR["Gemini Flash Claim & Context Engine"]
        EXTRACTOR -->|"Discrepancy Check"| MISMATCH["Caption vs Video Mismatch Warning"]
        EXTRACTOR -->|"Source Identification"| FULL_VID["Original Full Video YouTube Link Discovery"]
        EXTRACTOR -->|"Discrete Assertions"| CLAIMS["Verifiable Factual Claims (Origin Tagged)"]
    end

    subgraph FC["4. Real-time Live Web Evidence & Verification Layer"]
        CLAIMS -->|"Parallel Query Execution"| SEARCH["Free Live Web Search Engine (DuckDuckGo / DDGS)"]
        SEARCH -->|"Real-time News & Fact Articles"| EVIDENCE["Live Web Citations & Fact-Check Evidence"]
        EVIDENCE & CLAIMS -->|"Evaluate"| VERIFIER["Gemini Flash Fact Verifier"]
        VERIFIER -->|"Verdicts & Explanations"| REPORT["Aggregated Fact-Check & Forensics Report"]
    end

    subgraph UI_LAYER["5. Interactive Presentation Layer"]
        REPORT -->|"JSON Response"| APP
        APP --> DASH["Glassmorphic Fact-Check Dashboard\n• 3D Glowing Verdict Centerpiece\n• Video vs Caption Breakdown\n• AI / Deepfake Forensic Badges\n• Live Web Citations & Full Video Button"]
    end

    classDef primary fill:#6366f1,stroke:#4f46e5,stroke-width:2px,color:#fff;
    classDef gemini fill:#0284c7,stroke:#0369a1,stroke-width:2px,color:#fff;
    classDef success fill:#16a34a,stroke:#15803d,stroke-width:2px,color:#fff;
    classDef warn fill:#d97706,stroke:#b45309,stroke-width:2px,color:#fff;

    class GEM_MM,EXTRACTOR,VERIFIER gemini;
    class DASH,API primary;
    class REPORT,EVIDENCE success;
    class MISMATCH warn;
```

---

## ⚡ Tech Stack & Capabilities

- **Backend**: Python 3.11, FastAPI, Uvicorn, Pydantic v2
- **Multimodal Video & Audio Engine**: Google Gemini 2.5 Flash (`google.generativeai` / `google.genai`)
- **Claim Extraction & Fact Verification**: Google Gemini Flash Latest (`gemini-flash-latest` via OpenAI compatibility)
- **Zero-Key Live Web Search**: `ddgs` / `duckduckgo-search` (Real-time live news and fact-check citations)
- **Local Fallback Speech Transcription**: `faster-whisper` (Running `tiny` int8 model locally on CPU)
- **Media Downloaders & Video Processing**: `yt-dlp`, `ffmpeg`, `instaloader`
- **Mobile Client**: Flutter 3.x (Material 3 dark design, Android Share Sheet receiver via `receive_sharing_intent`)
- **Web Client**: Modern Glassmorphic Dark Dashboard with responsive radial gradients and micro-animations
- **Cloud Deployment**: 1-Click Render Docker blueprint (`render.yaml`) & Cloudflare Tunnels (`cloudflared`)

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
