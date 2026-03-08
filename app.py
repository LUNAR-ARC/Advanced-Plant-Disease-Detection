import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request
import json
import os
import cv2
import numpy as np

# ------------------------------
# Setup
# ------------------------------
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load Model
# ------------------------------
checkpoint = torch.load("plant_model.pth", map_location=device)
class_names = checkpoint["class_names"]

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# ------------------------------
# Image Transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------
# Load Treatment Data
# ------------------------------
with open("treatments.json", "r") as f:
    treatments = json.load(f)

# ------------------------------
# Prediction Function
# ------------------------------
def predict_image(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_percent = round(confidence.item() * 100, 2)

    return predicted_class, confidence_percent

# ------------------------------
# Advanced Severity Detection
# ------------------------------
def calculate_severity(pil_image):

    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Yellow (early infection)
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Brown (moderate infection)
    lower_brown = np.array([5, 50, 20])
    upper_brown = np.array([20, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Dark spots (severe infection)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 40])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    infection_mask = mask_yellow + mask_brown + mask_dark

    infected_pixels = np.sum(infection_mask > 0)
    total_pixels = infection_mask.size

    infection_percentage = (infected_pixels / total_pixels) * 100

    # Severity Levels
    if infection_percentage < 10:
        severity = "Healthy / Very Mild 🌿"
    elif infection_percentage < 25:
        severity = "Mild 🍃"
    elif infection_percentage < 50:
        severity = "Moderate 🍂"
    else:
        severity = "Severe 🔴"

    return round(infection_percentage, 2), severity

# ------------------------------
# Routes
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    recommendation = None

    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")

        predicted_class, confidence = predict_image(image)
        infection_percent, severity = calculate_severity(image)

        if predicted_class in treatments:
            recommendation = treatments[predicted_class]

        result = {
            "disease": predicted_class,
            "confidence": confidence,
            "infection_percent": infection_percent,
            "severity": severity
        }

    return render_template("index.html", result=result, recommendation=recommendation)

# ------------------------------
# Webcam Route
# ------------------------------
@app.route("/webcam")
def webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        predicted_class, confidence = predict_image(pil_img)
        infection_percent, severity = calculate_severity(pil_img)

        display_text = f"{predicted_class} | {severity} ({infection_percent}%)"

        cv2.putText(frame, display_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.imshow("PlantCare AI - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return "Webcam session ended."

# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)

# Nirmata -> Neer Chaudhary ^_^