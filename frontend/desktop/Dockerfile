FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir gradio>=4.8.0 httpx>=0.25.0

COPY . .

EXPOSE 7860

CMD ["python", "main.py"]
