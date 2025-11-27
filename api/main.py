from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import timm
from torchvision import transforms
import io

app = FastAPI()

# Add CORS so Vercel / other frontends can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later change to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load Model ----
num_classes = 38   # UPDATED AFTER TRAINING
model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load("saved_model/efficientnet_efficientnet_plant_disease.pth", map_location="cpu"))
model.load_state_dict(torch.load("saved_model/efficientnet_plant_disease.pth", map_location="cpu"), strict=False)
model.eval()

# ---- Preprocessing ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASS_NAMES = []  # <-- UPDATE AFTER TRAINING FINISHES

# ---- Routes ----
@app.get("/")
def home():
    return {"status": "API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an uploaded image file and returns predicted class and confidence.

    Note: model currently loads on CPU. If you change to GPU runtime, send the
    image tensor to the same device before inference.
    """
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # shape: [1, C, H, W]

    # Ensure tensor is on same device as model
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        predicted = int(torch.argmax(outputs, 1).item())
        confidence = float(torch.softmax(outputs, dim=1)[0][predicted].item())

    return {
        "class_name": CLASS_NAMES[predicted] if CLASS_NAMES else predicted,
        "confidence": round(confidence * 100, 2)
    }
