
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
from io import BytesIO

from CNN.cnn_inference import predict_image


router = APIRouter(prefix="/cnn", tags=["CNN"])


@router.post("/predict")
async def cnn_predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(BytesIO(contents))
        image.load()

        result = predict_image(image)

        return result

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )
