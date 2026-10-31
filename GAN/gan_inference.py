
import torch

from pathlib import Path
from io import BytesIO
from PIL import Image

from GAN.gan_model import Generator


# Model configuration
LATENT_DIM = 100

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "generator.pth"
)

# Load trained generator
generator = Generator()

generator.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

generator.eval()


def generate_image():

    # Generate random noise
    noise = torch.randn(1, LATENT_DIM)

    # Generate image
    with torch.no_grad():
        fake_image = generator(noise)

    # Convert [-1, 1] to [0, 255]
    image_array = (
        (fake_image.squeeze().clamp(-1, 1) + 1)
        * 127.5
    ).to(torch.uint8).cpu().numpy()

    # Convert to PIL image
    image = Image.fromarray(image_array, mode="L")

    # Convert image to PNG bytes
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    buffer.seek(0)

    return buffer.getvalue()
