
import torch
import torch.nn as nn


# Part 1: Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(100, 7 * 7 * 128)

        self.generator = nn.Sequential(
            # 128 x 7 x 7 -> 64 x 14 x 14
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64 x 14 x 14 -> 1 x 28 x 28
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        x = self.generator(x)
        return x


# Part 2: Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential(
            # 1 x 28 x 28 -> 64 x 14 x 14
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2),

            # 64 x 14 x 14 -> 128 x 7 x 7
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),

            # Real / Fake probability
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)


# Test the GAN architecture
if __name__ == "__main__":

    batch_size = 4

    generator = Generator()
    discriminator = Discriminator()

    # Random noise input
    noise = torch.randn(batch_size, 100)

    # Generate fake images
    fake_images = generator(noise)

    # Discriminator predicts real/fake
    predictions = discriminator(fake_images)

    print("Generator output:", fake_images.shape)
    print("Discriminator output:", predictions.shape)

    assert fake_images.shape == (batch_size, 1, 28, 28)
    assert predictions.shape == (batch_size, 1)

    print("GAN architecture test passed!")
