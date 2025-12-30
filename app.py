import os
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import joblib
import numpy as np

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "anemia_cnn90.pth"  # ملف الموديل المحفوظ
IMG_SIZE = 128
NUM_CLASSES = 2
# ----------------------------------------

# ---------------- موديل الأصلي ----------------


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
# ---------------- انتهى الموديل ----------------


# ----------------- Load model -----------------
scaler = joblib.load("scaler.joblib")
model = ImprovedCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----------------- Transform -----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ----------------- Flask App -----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/anemia", methods=["GET", "POST"])
def anemia():
    result = None
    filename = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # تصنيف الصورة
            image = Image.open(filepath).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                result = "Anemic" if pred.item() == 0 else "Non_Anemic"

    return render_template("index.html", result=result, filename=filename)


@app.route("/malnutrition", methods=["GET", "POST"])
def malnutrition():
    # تحميل الموديل الصحيح
    malnutrition_model = joblib.load("KNN_Classifier_2.joblib")

    result = None
    if request.method == "POST":
        gender = int(request.form.get("gender"))
        hemoglobin = float(request.form.get("hemoglobin"))
        mch = float(request.form.get("mch"))
        mchc = float(request.form.get("mchc"))
        mcv = float(request.form.get("mcv"))

        features = np.array([[gender, hemoglobin, mch, mchc, mcv]])

        features_scaled = scaler.transform(features)
        pred = malnutrition_model.predict(features_scaled)[0]
        result = "Malnourished" if pred == 1 else "Normal"

    return render_template("malnutrition.html", result=result)


# ----------------- Run -----------------
if __name__ == "__main__":
    app.run(debug=True)
