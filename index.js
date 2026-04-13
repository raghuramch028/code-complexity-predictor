const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { GoogleGenAI } = require('@google/genai');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

console.log("Starting server implementation...");

// Initialize SDK explicitly
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

app.post('/api/predict', async (req, res) => {
    try {
        const { code } = req.body;
        
        if (!code) {
            return res.status(400).json({ error: 'Code is required' });
        }

        const prompt = `
You are an expert algorithm analyzer. Analyze the time and space complexity of the following code.
Return your response in two parts:
1. The overall Time Complexity and Space Complexity in Big-O notation.
2. A brief, exact, and clear reasoning for why.

Code:
${code}
        `;

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        res.json({ result: response.text });
    } catch (error) {
        console.error('Error hitting Gemini API:', error);
        res.status(500).json({ error: 'Failed to analyze code complexity' });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});
