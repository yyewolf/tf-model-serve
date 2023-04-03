FROM armswdev/tensorflow-arm-neoverse:latest

RUN sudo apt-get update && sudo apt-get -y upgrade
RUN sudo apt-get install -y libsm6 libxext6 libxrender-dev ffmpeg
RUN pip install --upgrade pip
RUN pip install numpy Pillow numpy fastapi pydantic uvicorn python-multipart backgroundremover opencv-python-headless

WORKDIR /
COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]