FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for pytgcalls/opencv + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxcb-shm0 \
    libxcb-render0 \
    libxcb-xinerama0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY project/requirements.txt /app/project/requirements.txt
RUN pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir -r /app/project/requirements.txt

COPY . /app

CMD ["python", "project/bot.py"]
