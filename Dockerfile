FROM armswdev/tensorflow-arm-neoverse:latest
RUN pip install --upgrade pip
RUN pip install numpy Pillow numpy fastapi pydantic uvicorn python-multipart

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]