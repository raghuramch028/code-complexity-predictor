import gdown
import os

print("---------------------------------------------------------")
print("📥 GOOGLE DRIVE MODEL DOWNLOADER")
print("---------------------------------------------------------")

# IMPORTANT: Replace these strings with the actual File IDs from your Google Drive share links!
# Example link: https://drive.google.com/file/d/1A2B3C4D5E/view?usp=sharing
# The File ID is: 1A2B3C4D5E

# 1. Google Drive File ID for best_model.pt
MODEL_FILE_ID = "YOUR_MODEL_FILE_ID_HERE"

# 2. Google Drive File ID for label_encoder.pkl (if you didn't commit it)
# By default, I am leaving the Hugging Face link here just for the 1KB encoder so you don't have to upload it twice.
# If you want it from GDrive too, uncomment and put the ID!
# ENCODER_FILE_ID = "YOUR_ENCODER_FILE_ID_HERE"

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

# Download tiny encoder either from GitHub or HF (it's only 1KB so it doesn't cause LFS issues)
if not os.path.exists("label_encoder.pkl"):
    import urllib.request
    print("Downloading label_encoder.pkl...")
    url_le = "https://huggingface.co/spaces/raghuram00/code-complexity-predictor/resolve/main/label_encoder.pkl"
    urllib.request.urlretrieve(url_le, "label_encoder.pkl")
    print("✅ Encoder downloaded successfully!")
