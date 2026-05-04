# TrustVision AI

TrustVision AI is a face-image authenticity analysis project built with:

- `FastAPI` for the backend API
- `React + Vite + Tailwind CSS` for the frontend
- `PyTorch` for image inference
- `SQLite` for scan history
- `ReportLab` for PDF report export

This repository currently supports:

- Image upload analysis
- Webcam frame sampling from the browser
- Authenticity score, prediction, confidence, and risk level
- PDF report download

This repository does **not** currently implement:

- User authentication
- PostgreSQL
- Video file upload and end-to-end video analysis in the UI
- Enterprise dashboards or role-based access control

## Project Layout

```text
TrustVision-AI/
  Backend/              FastAPI routes and PDF reporting
  Frontend/             React application
  database/             SQLAlchemy database setup and models
  ml/                   Core inference logic and ML model weights
  app.py                Gradio UI demo
  demo.py               Command-line interface (CLI) image scan demo
  start_app.py          Starts backend and frontend together
```

## How to Run the Project

### Prerequisites
- **Python 3.10+**
- **Node.js & npm**

### 1. Install Dependencies
Install the Python backend and machine learning dependencies:
```bash
pip install -r requirements.txt
```

Install the React frontend dependencies:
```bash
cd Frontend
npm install
cd ..
```

### 2. Start the Application
You can start both the FastAPI backend and the React frontend simultaneously using the provided startup script. From the repository root, run:
```bash
python start_app.py
```
This script will automatically launch:
- **Backend API:** `http://localhost:8000`
- **Frontend UI:** `http://localhost:5173`

*(Press `Ctrl+C` in the terminal to stop all services.)*

Alternatively, you can run them manually in separate terminal windows:
- **Backend:** `python -m uvicorn Backend.main:app --reload`
- **Frontend:** `cd Frontend && npm run dev`

### Optional Gemini Verification
The app works with the local PyTorch checkpoint by default. To enable the optional Gemini secondary verifier, set one of these environment variables before starting the backend:
```bash
GEMINI_API_KEY=your_key_here
```
For key rotation, set `GEMINI_API_KEYS` to a comma-separated list. Do not hardcode API keys in source files.

## Optional Demos

### Gradio Web UI
A lightweight Gradio web interface is available for quickly testing the model outside the main application:
```bash
python app.py
```

### CLI Image Scan
You can easily scan an image from the terminal with a simple command:
```bash
python demo.py --image path/to/file.jpg
```
Heatmap generation is not currently supported with the new SigLip model integration.

## License

MIT
