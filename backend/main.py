import os
import torch
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Complexity descriptions 
DESCRIPTIONS = {
    "constant":  ("O(1)", "⚡ Constant Time", "Executes in the same time regardless of input size. Very fast!"),
    "linear":    ("O(n)", "📈 Linear Time", "Execution time grows linearly with input size."),
    "logn":      ("O(log n)", "🔍 Logarithmic Time", "Very efficient! Common in binary search algorithms."),
    "nlogn":     ("O(n log n)", "⚙️ Linearithmic Time", "Common in efficient sorting algorithms like merge sort."),
    "quadratic": ("O(n²)", "🐢 Quadratic Time", "Execution time grows quadratically. Common in nested loops."),
    "cubic":     ("O(n³)", "🦕 Cubic Time", "Triple nested loops. Avoid for large inputs."),
    "np":        ("O(2ⁿ)", "💀 Exponential Time", "NP-Hard complexity. Only feasible for very small inputs."),
}

app = FastAPI(title="Code Complexity Predictor API")

class PredictRequest(BaseModel):
    code: str

# Global state
model = None
tokenizer = None
le = None
device = None

@app.on_event("startup")
def load_resources():
    global model, tokenizer, le, device
    print("Loading resources...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Load label encoder
    if os.path.exists("label_encoder.pkl"):
        le = joblib.load("label_encoder.pkl")
    else:
        print("WARNING: label_encoder.pkl not found!")
        
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=7)
    if os.path.exists("best_model.pt"):
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
    else:
        print("WARNING: best_model.pt not found!")
        
    model.to(device)
    model.eval()
    print("Resources loaded successfully!")

@app.post("/api/predict")
def predict_complexity(request: PredictRequest):
    code = request.code
    if not code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
        
    try:
        inputs = tokenizer(code, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()

        label = le.inverse_transform([pred])[0]
        notation, title, description = DESCRIPTIONS.get(label, (label, label, ""))

        return {
            "notation": notation,
            "title": title,
            "description": description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
