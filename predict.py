import torch
from torchvision import transforms
from PIL import Image
import timm
import os

# -----------------------------
# Load class names
# -----------------------------
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load Model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "saved_model/efficientnet_plant_disease.pth"

model = timm.create_model(
    'efficientnet_b0.ra_in1k',
    pretrained=False,
    num_classes=len(CLASS_NAMES)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# -----------------------------
# Predict Function
# -----------------------------
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)

    disease = CLASS_NAMES[pred_class.item()]
    confidence = confidence.item() * 100

    return disease, confidence

# -----------------------------
# Run prediction
# -----------------------------
if __name__ == "__main__":
    test_image = "test_images/tomato_test.jpg"   # change path if needed
    disease, confidence = predict_image(test_image)

    print("\n=== Prediction Result ===")
    print("Image:", test_image)
    print("Predicted Disease:", disease)
    print("Confidence: {:.2f}%".format(confidence))
