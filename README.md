# ⚙️ Code Complexity Predictor

An AI-powered web application that instantly predicts the Big-O Time Complexity of Python and Java code snippets using **GraphCodeBERT**. 

## 🚀 Features
- **Intelligent Analysis:** Powered by Microsoft's GraphCodeBERT fine-tuned on the CodeParrot/CodeComplex dataset.
- **Premium Interface:** A stunning Glassmorphism dark-mode UI with syntax highlighting and micro-animations.
- **Lightning Fast:** Built on a lightweight FastAPI backend for near-instant inference.
- **Cloud-Ready:** Completely containerized with Docker, configured for automatic deploy on Render.com.

## 🛠️ Tech Stack
- **Frontend:** HTML5, Vector CSS (Vanilla), JavaScript, PrismJS
- **Backend:** Python, FastAPI, Uvicorn
- **AI/ML:** PyTorch, HuggingFace Transformers (`GraphCodeBERT`)
- **Deployment:** Docker, Render

## 💻 Running Locally

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Model files**
   Ensure you have configured `download_model.py` with your Google Drive File ID, then run:
   ```bash
   python download_model.py
   ```

3. **Start the Server**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Open the App**
   Navigate to `http://localhost:8000` in your web browser.

---
*Built with ❤️ for algorithmic analysis.*
