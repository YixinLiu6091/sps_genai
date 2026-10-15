import torch

from pathlib import Path
from PIL import Image
from torchvision import transforms

from CNN.cnn_model import CNNModel


MODEL_PATH = Path(__file__).resolve().parent / "cnn_model.pth"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


def load_model():
    model = CNNModel()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict_image(image: Image.Image):
    model = load_model()

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(
            probabilities, dim=1
        )

    class_id = predicted.item()

    return {
        "predicted_class": CLASS_NAMES[class_id],
        "class_id": class_id,
        "confidence": round(confidence.item(), 4)
    }
