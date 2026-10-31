
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from GAN.gan_model import Generator, Discriminator


# ==========================================
# 1. Set up paths
# ==========================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENERATOR_PATH = ROOT / "generator.pth"
DISCRIMINATOR_PATH = ROOT / "discriminator.pth"


# ==========================================
# 2. Select device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# ==========================================
# 3. Training configuration
# ==========================================

EPOCHS = 30
BATCH_SIZE = 64
LATENT_DIM = 100
LEARNING_RATE = 0.0001

torch.manual_seed(42)


# ==========================================
# 4. Load MNIST dataset
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])

train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

print("Training images:", len(train_dataset))


# ==========================================
# 5. Initialize GAN models
# ==========================================

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_g = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

optimizer_d = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

# Fixed noise for comparing generated images
fixed_noise = torch.randn(
    64,
    LATENT_DIM,
    device=device
)

datalogs = []


# ==========================================
# 6. Train GAN
# ==========================================

for epoch in range(EPOCHS):

    generator.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        ncols=110
    )

    for real_images, _ in progress:

        real_images = real_images.to(device)

        batch_size = real_images.size(0)

        # Label smoothing for real images
        real_labels = torch.full(
            (batch_size, 1),
            0.9,
            device=device
        )

        fake_labels = torch.zeros(
            batch_size,
            1,
            device=device
        )

        # ----------------------------------
        # A. Train Discriminator
        # ----------------------------------

        optimizer_d.zero_grad()

        # Real images
        real_predictions = discriminator(
            real_images
        )

        real_loss = criterion(
            real_predictions,
            real_labels
        )

        # Generate fake images
        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            device=device
        )

        fake_images = generator(noise)

        # Detach to avoid updating Generator
        fake_predictions = discriminator(
            fake_images.detach()
        )

        fake_loss = criterion(
            fake_predictions,
            fake_labels
        )

        # Total discriminator loss
        d_loss = real_loss + fake_loss

        if not torch.isfinite(d_loss).item():
            raise RuntimeError(
                "Non-finite discriminator loss detected."
            )

        d_loss.backward()
        optimizer_d.step()


        # ----------------------------------
        # B. Train Generator
        # ----------------------------------

        # Freeze discriminator parameters
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)

        optimizer_g.zero_grad()

        # Generate a fresh batch of images
        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            device=device
        )

        fake_images_for_g = generator(noise)

        # Generator tries to fool discriminator
        generated_predictions = discriminator(
            fake_images_for_g
        )

        # Generator wants predictions close to 1
        generator_targets = torch.ones_like(
            generated_predictions
        )

        g_loss = criterion(
            generated_predictions,
            generator_targets
        )

        if not torch.isfinite(g_loss).item():
            raise RuntimeError(
                "Non-finite generator loss detected."
            )

        g_loss.backward()
        optimizer_g.step()

        # Unfreeze discriminator parameters
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)


        # ----------------------------------
        # C. Track training performance
        # ----------------------------------

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()

        progress.set_postfix({
            "D Loss": f"{d_loss.item():.4f}",
            "G Loss": f"{g_loss.item():.4f}"
        })


    # ======================================
    # 7. Epoch summary
    # ======================================

    avg_d_loss = total_d_loss / len(train_loader)
    avg_g_loss = total_g_loss / len(train_loader)

    datalogs.append({
        "epoch": epoch + 1,
        "discriminator_loss": avg_d_loss,
        "generator_loss": avg_g_loss
    })

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}], "
        f"D Loss: {avg_d_loss:.4f}, "
        f"G Loss: {avg_g_loss:.4f}"
    )


    # ======================================
    # 8. Save generated image samples
    # ======================================

    generator.eval()

    with torch.no_grad():
        samples = generator(
            fixed_noise
        ).cpu()

    save_image(
        samples,
        OUTPUT_DIR / f"epoch_{epoch + 1:02d}.png",
        nrow=8,
        normalize=True,
        value_range=(-1, 1)
    )

    # ======================================
    # 9. Save model weights
    # ======================================

    torch.save(
        generator.state_dict(),
        GENERATOR_PATH
    )

    torch.save(
        discriminator.state_dict(),
        DISCRIMINATOR_PATH
    )

    print(
        f"Epoch {epoch + 1}: "
        "Model weights and sample image saved."
    )


# ==========================================
# 10. Training completed
# ==========================================

print("\nFinished Training!")

print(
    "Generator saved to:",
    GENERATOR_PATH
)

print(
    "Discriminator saved to:",
    DISCRIMINATOR_PATH
)

print(
    "Generated samples saved to:",
    OUTPUT_DIR
)
