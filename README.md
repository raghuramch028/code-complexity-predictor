# ⚡ Code Complexity Predictor

A beautiful, sleek, and developer-focused web application that instantly analyzes any provided source code and accurately predicts its asymptotic Time and Space Complexity (Big-O Notation).

https://code-complexity.onrender.com/

## 🚀 Features

- **Multi-language Support**: Automatically understands Python, Java, C++, JavaScript, Go, and more.
- **Deep Algorithmic Analysis**: Delivers highly accurate Big-O notation for both Time and Space.
- **Explainability**: Provides a concise step-by-step reasoning for the determined complexity.
- **Premium Design**: Features an ultra-modern dark theme, glowing UI elements, beautiful typography, and translucent glassmorphism effects.

## 💻 Tech Stack

- **Frontend**: Vanilla HTML5, CSS3 (Modern features & animations), and JavaScript.
- **Backend**: Node.js & Express.js.
- **Intelligence**: Powered by a state-of-the-art Advanced AI engine via REST backend integration.

## 🛠️ Usage

1. Open the web interface.
2. Paste any code snippet or entire algorithm into the Code Editor block.
3. Click **"Analyze Complexity →"**.
4. The backend will parse the logic and return the formatted Big-O complexities along with reasoning.

## ⚙️ Local Setup

To run this application on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/raghuramch028/code-complexity-predictor.git
   cd code-complexity-predictor
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory and securely configure your Intelligence API credentials:
   ```env
   GEMINI_API_KEY=your_authentication_key_here
   PORT=3000
   ```
   *(Note: The environment variable relies on this exact naming convention for backend logic routing).*

4. **Start the server:**
   ```bash
   npm start
   ```

5. **View the Application:**
   Open your browser and navigate to `http://localhost:3000`.

## 🌐 Deployment

This application is structurally configured and optimized for zero-downtime deployment on platforms like Render, Heroku, or Vercel edge networks. Simply map the respective build/start commands and provide your Environment Variables in the hosting dashboard.
