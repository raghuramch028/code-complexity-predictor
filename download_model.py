import gdown
import os

print("---------------------------------------------------------")
print("📥 GOOGLE DRIVE MODEL DOWNLOADER")
print("---------------------------------------------------------")

# IMPORTANT: Replace these strings with the actual File IDs from your Google Drive share links!
# Example link: https://drive.google.com/file/d/1A2B3C4D5E/view?usp=sharing
# The File ID is: 1A2B3C4D5E

# 1. Google Drive File ID for best_model.pt
MODEL_FILE_ID = "1eFqWSvXSl-bcaPXh6LCG2uQmBmEF5GGJ"

# 2. Google Drive File ID for label_encoder.pkl 
ENCODER_FILE_ID = "1S_zqbzkMNVuYnQlopMmrRRujIOeF5v2q"

if not os.path.exists("best_model.pt"):
    if MODEL_FILE_ID == "YOUR_MODEL_FILE_ID_HERE":
        print("❌ ERROR: You did not paste your Google Drive tracking ID in download_model.py!")
        print("Please edit download_model.py, paste your ID into MODEL_FILE_ID, and push to GitHub.")
        exit(1)
        
    print(f"Downloading best_model.pt from Google Drive...")
    gdown.download(id=MODEL_FILE_ID, output="best_model.pt", quiet=False)
    print("✅ Model downloaded successfully!")
else:
    print("✅ best_model.pt already exists locally. Skipping download.")

if not os.path.exists("label_encoder.pkl"):
    if ENCODER_FILE_ID == "YOUR_ENCODER_FILE_ID_HERE":
        print("❌ ERROR: You did not paste your Google Drive tracking ID for label_encoder.pkl!")
        exit(1)
        
    print("Downloading label_encoder.pkl from Google Drive...")
    gdown.download(id=ENCODER_FILE_ID, output="label_encoder.pkl", quiet=False)
    print("✅ Encoder downloaded successfully!")
