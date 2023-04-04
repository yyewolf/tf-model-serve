from app.background import remove_background
import tensorflow as tf
import numpy as np
from app.model import Model
from fastapi import FastAPI, File, Response
from PIL import Image
import io
import os
import cv2

app = FastAPI()

models = {}


def load_model(name: str):
    models[name] = Model(name)


def load_models():
    for model in os.listdir('models'):
        load_model(model)


def model_predict(model_name: str, img, k: int):
    md = models[model_name]

    test_image = None
    if type(img) == str:
        test_image = tf.keras.preprocessing.image.load_img(
            img, target_size=(md.IMG_HEIGHT, md.IMG_WIDTH))
    elif type(img) == bytes:
        test_image = Image.open(io.BytesIO(img)).convert('RGB')
        test_image = test_image.resize((md.IMG_HEIGHT, md.IMG_WIDTH))
        test_image = np.array(test_image)
    else:
        # resize the image to the input dimensions of the model
        test_image = img.resize((md.IMG_HEIGHT, md.IMG_WIDTH))

    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = tf.keras.applications.mobilenet_v2.preprocess_input(
        test_image)

    r = md.predict(test_image, k)

    # Convert to normal dict
    output = []
    for i in r:
        output.append({"label": i[0], "probability": float(i[1])})
    return output


def background_remove(model_name: str, img):
    md = models[model_name]
    test_image = Image.open(io.BytesIO(img)).convert('RGB')
    test_image = np.array(test_image)
    test_image = remove_background(test_image)
 
    # aimed width is md.IMG_WIDTH, aimed height is md.IMG_HEIGHT
    if test_image.shape[0] > test_image.shape[1]:
        # height is greater than width
        pad_size = (test_image.shape[0] - test_image.shape[1]) // 2
        padded_image = cv2.copyMakeBorder(test_image, 0, 0, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        # width is greater than height
        pad_size = (test_image.shape[1] - test_image.shape[0]) // 2
        padded_image = cv2.copyMakeBorder(test_image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    resized_image = cv2.resize(padded_image, (md.IMG_WIDTH, md.IMG_HEIGHT))
        
    # convert back into bytes to return it
    img = Image.fromarray(resized_image)
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


load_models()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models")
async def list_models():
    return {"models": list(models.keys())}


@app.get("/models/{model}")
async def status(model: str):
    md = models.get(model)
    if md:
        return {"status": "loaded"}
    return {"status": "not loaded"}


@app.post("/models/{model}/predict")
async def predict(model: str, upload: bytes = File(...), k: int = 3):
    # Get the image from the request
    if not upload:
        return {"message": "No upload file sent"}
    return model_predict(model, upload, k)


@app.post("/transforms/{model}/background_removal", responses={
    200: {
        "content": {"image/png": {}}
    }
}, response_class=Response)
async def predict(model: str, upload: bytes = File(...)):
    # Get the image from the request
    if not upload:
        return {"message": "No upload file sent"}
    return Response(content=background_remove(model, upload), media_type="image/png")
