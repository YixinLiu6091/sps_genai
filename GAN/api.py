
from fastapi import APIRouter
from fastapi.responses import Response

from GAN.gan_inference import generate_image


router = APIRouter(
    prefix="/gan",
    tags=["GAN"]
)


@router.get("/generate")
def gan_generate():

    image_bytes = generate_image()

    return Response(
        content=image_bytes,
        media_type="image/png"
    )
