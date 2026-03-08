import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ------------------------------
# 1. Device Setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 2. Load Saved Model
# ------------------------------
checkpoint = torch.load("plant_model.pth", map_location=device)
class_names = checkpoint["class_names"]

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# ------------------------------
# 3. Image Transform
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
# 4. Get Image Path from User
# ------------------------------
image_path = input("Enter image path: ")

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# ------------------------------
# 5. Predict
# ------------------------------
with torch.no_grad():
    outputs = model(image)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

predicted_class = class_names[predicted.item()]
confidence_percent = confidence.item() * 100

print("\n🌿 Disease Prediction:", predicted_class)
print("🔥 Confidence: {:.2f}%".format(confidence_percent))

# Nirmata -> Neer Chaudhary ^_^