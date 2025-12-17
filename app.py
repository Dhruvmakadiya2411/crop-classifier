import os
import io
import threading
import gc
import torch
import timm
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware # Added for browser safety
from PIL import Image
from torchvision import transforms

# ======================================================
# SAFETY CONFIG (LOW RAM)
# ======================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

DEVICE = "cpu"

# ======================================================
# FASTAPI APP + CORS SETUP
# ======================================================
app = FastAPI(title="Crop Disease Classifier")

# Add this block to allow your HTML to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================
# MODEL PATHS
# ======================================================
MODEL_PATHS = {
    "corn": os.path.join(BASE_DIR, "models", "corn.pth"),
    "pepper": os.path.join(BASE_DIR, "models", "pepper.pth"),
    "potato": os.path.join(BASE_DIR, "models", "potato.pth"),
    "strawberry": os.path.join(BASE_DIR, "models", "strawberry.pth"),
}

# ======================================================
# SINGLE MODEL CACHE (LAZY)
# ======================================================
_model = None
_classes = None
_loaded_crop = None
_model_lock = threading.Lock()

# ======================================================
# TRANSFORMS
# ======================================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================================
# LAZY MODEL LOADER
# ======================================================
def load_model_lazy(crop: str):
    global _model, _classes, _loaded_crop
    if crop not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Invalid crop")

    with _model_lock:
        if crop == _loaded_crop and _model is not None:
            return _model, _classes

        # Clear memory
        _model = None
        _classes = None
        _loaded_crop = None
        gc.collect()

        try:
            checkpoint = torch.load(MODEL_PATHS[crop], map_location=DEVICE)
            model = timm.create_model(
                checkpoint["model_name"],
                pretrained=False,
                num_classes=len(checkpoint["classes"])
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            _model = model
            _classes = checkpoint["classes"]
            _loaded_crop = crop
            return _model, _classes
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Model load failed")

# ======================================================
# INFERENCE
# ======================================================
def predict_image(image: Image.Image, crop: str):
    model, classes = load_model_lazy(crop)
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, idx = torch.max(probs, 1)
    return classes[idx.item()], round(conf.item() * 100, 2)

# ======================================================
# ROUTES
# ======================================================
@app.get("/")
def serve_ui():
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        return {"error": "index.html not found in project directory"}
    return FileResponse(index_path)

@app.post("/predict")
async def predict(
    crop: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image type")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        label, confidence = predict_image(image, crop)

        # Map 'pepper' to 'Bell Pepper' for the frontend UI display if needed
        display_crop = "Bell Pepper" if crop == "pepper" else crop.capitalize()

        return {
            "crop": display_crop,
            "disease": label,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Prediction process failed")

if __name__ == "__main__":
    import uvicorn
    # This matches the MacBook Pro path and port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)